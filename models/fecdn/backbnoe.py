from torch.nn.modules.utils import _pair as to_2tuple
from timm.layers import DropPath
import numpy as np
import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    device = x.device  # 获取输入张量 x 的设备
    filters = filters.to(device)  # 将 filters 移动到输入张量所在的设备
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    device = x.device  # 获取输入张量 x 的设备
    filters = filters.to(device)  # 将 filters 移动到输入张量所在的设备
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class Gateatt(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x, y):
        attn1 = x
        attn2 = y
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return attn



class WPConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, wt_levels=1, wt_type='db1'):
        super(WPConv, self).__init__()
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.convs = nn.Sequential(
                nn.Conv2d(in_channels * 4, in_channels * 4, 3, padding=1, stride=1, dilation=1, groups=in_channels * 4),
                nn.BatchNorm2d(in_channels * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels * 4, in_channels, 1, padding=0, stride=1, dilation=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
                )

        self.convs2 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, 3, padding=1, stride=1, dilation=1, groups=in_channels * 4),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 4, in_channels, 1, padding=0, stride=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.att = Gateatt(in_channels)

        self.convs5 = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, padding=0, stride=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))


    def forward(self, x):
        size = x.size()[2:]
        x1 = self.wt_function(x)
        x1_ll = x1[:, :, 0, :, :]
        shape_x1 = x1.shape
        x1 = x1.reshape(shape_x1[0], shape_x1[1] * 4, shape_x1[3], shape_x1[4])
        x1 = self.convs(x1)


        x2 = self.wt_function(x1_ll)
        shape_x2 = x2.shape
        x2 = x2.reshape(shape_x2[0], shape_x2[1] * 4, shape_x2[3], shape_x2[4])
        x2 = self.convs2(x2)


        x1 = F.interpolate(x1, size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size, mode='bilinear', align_corners=True)
        out = self.att(x1, x2) * x

        # out = self.convs5(torch.cat([x, x1, x2], 1))
        return out

class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = WPConv(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        x = self.proj_1(x)
        shorcut = x.clone()
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut



class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x), H, W


class WPCNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[32, 64, 160, 256],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0.1, drop_path_rate=0.1,
                 depths=[3, 3, 5, 2]):
        super().__init__()

        # Stage 1
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0])
        self.block1 = nn.Sequential(*[
            Block(embed_dims[0], mlp_ratios[0], drop_rate, drop_path_rate * i / (depths[0] - 1))
            for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embed_dims[0], eps=1e-6)

        # Stage 2
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.block2 = nn.Sequential(*[
            Block(embed_dims[1], mlp_ratios[1], drop_rate, drop_path_rate * i / (depths[1] - 1))
            for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1], eps=1e-6)

        # Stage 3
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.block3 = nn.Sequential(*[
            Block(embed_dims[2], mlp_ratios[2], drop_rate, drop_path_rate * i / (depths[2] - 1))
            for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2], eps=1e-6)

        # Stage 4
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        self.block4 = nn.Sequential(*[
            Block(embed_dims[3], mlp_ratios[3], drop_rate, drop_path_rate * i / (depths[3] - 1))
            for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embed_dims[3], eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        x = self.block1(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        x = self.block2(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        x = self.block3(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        x = self.block4(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'Loading weights... {idx}/{len(model_dict)} items')
    return model_dict


