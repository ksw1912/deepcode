import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from timm.layers import DropPath


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

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

        return out

class Attention(nn.Module):
    def __init__(self, d_model1, d_model2):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model1, d_model2, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = WPConv(d_model2)
        self.proj_2 = nn.Conv2d(d_model2, d_model2, 1)

    def forward(self, x):
        x = self.proj_1(x)
        shorcut = x.clone()
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut


class Block(nn.Module):
    def __init__(self, dim1, dim2, mlp_ratio=4., drop=0.1, drop_path=0.1, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim1)
        self.norm2 = nn.BatchNorm2d(dim2)
        self.attn = Attention(dim1, dim2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim2 * mlp_ratio)
        self.mlp = Mlp(in_features=dim2, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim2), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim2), requires_grad=True)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x)
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

