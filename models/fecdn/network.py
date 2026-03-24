import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from module.backbone import WPCNet
from thop import profile
from module.fusion import Block
from functools import partial


def SRMLayer():
    q = [4.0, 12.0, 2.0]
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=float) / q[0]
    filter2 = np.asarray(filter2, dtype=float) / q[1]
    filter3 = np.asarray(filter3, dtype=float) / q[2]
    filters = np.asarray(
        [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]])  # shape=(3,3,5,5)
    # print(filters.shape)
    filters = np.repeat(filters, repeats=3, axis=0)
    filters = torch.from_numpy(filters.astype(np.float32))
    # filters = torch.from_numpy(filters)
    # print(filters.shape)
    return filters


class SRM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SRM, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        filters = SRMLayer()
        self.depthwise.weight = nn.Parameter(filters)
        self.depthwise.weight.requires_grad = False

    def forward(self, x):
        out = self.depthwise(x)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x



class MCA(nn.Module):
    def __init__(self, dim, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = 2 * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()

        self.split_indices = (dim, dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * c)
        return (x + shortcut).permute(0, 3, 1, 2)


class SCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1, bias=False),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*4, dim, 1, bias=False))
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)
        return x



class fusionmodule(nn.Module):
    def __init__(self, in_planes1, in_planes2, out_planes):
        super(fusionmodule, self).__init__()
        self.conv11 = BasicConv2d(in_planes1, out_planes * 3 // 2, 1)
        self.conv12 = BasicConv2d(in_planes2, out_planes * 3 // 2, 1)

        self.med_ch = out_planes

        self.SCA = SCA(self.med_ch)
        self.MCA = MCA(self.med_ch)
        self.LCA = Block(self.med_ch, self.med_ch)

        self.conv2 = BasicConv2d(out_planes * 3, out_planes, 1)

    def forward(self, x, y):

        x = self.conv11(x)
        y = self.conv12(y)


        x1, x2, x3 = torch.split(x, x.size(1) // 3, dim=1)
        y1, y2, y3 = torch.split(y, y.size(1) // 3, dim=1)

        fusion1 = self.SCA(torch.cat([x1, y1], dim=1))
        fusion2 = self.MCA(torch.cat([x2, y2], dim=1))
        fusion3 = self.LCA(torch.cat([x3, y3], dim=1))

        out = self.conv2(torch.cat([fusion1, fusion2, fusion3], dim=1))

        return out



class fgdecoder(nn.Module):
    def __init__(self):
        super(fgdecoder, self).__init__()
        self.cbr_fg = nn.Sequential(
            BasicConv2d(128*3, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            BasicConv2d(256, 128, kernel_size=3, padding=1)
        )
        self.cbr_head = nn.Sequential(
            BasicConv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

        self.Upsample2x = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.Upsample4x = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

    def forward(self, f2, f3, f4):
        f4 = self.Upsample4x(f4)
        f3 = self.Upsample2x(f3)
        fcat = torch.cat([f2, f3, f4], dim=1)

        fg_map = self.cbr_fg(fcat)
        fg_head = self.cbr_head(fg_map)

        return fg_map, fg_head

class edgedecoder(nn.Module):
    def __init__(self):
        super(edgedecoder, self).__init__()
        self.cbr_edge = nn.Sequential(BasicConv2d(128, 128, kernel_size=3, padding=1))

        self.cbr_head = nn.Sequential(
            BasicConv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0))

    def forward(self, f1):
        fedge_map = self.cbr_edge(f1)
        fedge_head = self.cbr_head(fedge_map)
        return fedge_map, fedge_head



class Preprocess(nn.Module):
    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()

        for i in range(up_times):
            self.preprocess.add_module(f'conv_{i}', BasicConv2d(out_c, out_c, kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        # x = self.c1(x)
        x = self.preprocess(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # h
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # w

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class FEAGModule(nn.Module):
    def __init__(self, in_c, dim):
        super().__init__()
        self.input_cbr = nn.Sequential(
            BasicConv2d(in_c, dim, kernel_size=3, padding=1),
            BasicConv2d(dim, dim, kernel_size=3, padding=1))


        self.fg_att = nn.Sequential(
            BasicConv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.Conv2d(dim//2, 1, kernel_size=1, padding=0))

        self.edge_att = nn.Sequential(
            BasicConv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.Conv2d(dim // 2, 1, kernel_size=1, padding=0),
            nn.Sigmoid())

        self.output_cbr = nn.Sequential(
            BasicConv2d(dim*3, dim, kernel_size=3, padding=1),
            CoordAtt(dim, dim))

        self.out2 = BasicConv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x, fg, edge):
        x = self.input_cbr(x)

        fg_att = self.fg_att(fg)
        edge_att = self.edge_att(edge)

        x_fg = x * torch.sigmoid(fg_att)
        x_edge = x * edge_att

        feature = torch.cat([x_fg, x_edge, x], dim=1)

        out = self.out2(self.output_cbr(feature) + x)

        return out




class Basenet(nn.Module):
    def __init__(self, ):
        super(Basenet, self).__init__()
        self.extract = WPCNet()  #

        self.hpf_conv = SRM(in_channels=3, out_channels=3, kernel_size=5, padding=2)
        self.conv = nn.Conv2d(9, 3, 3, 1, 1)
        self.backbone_recon = WPCNet()

        # feature fusion
        self.fusion1 = fusionmodule(32, 32, 128)
        self.fusion2 = fusionmodule(64, 64, 128)
        self.fusion3 = fusionmodule(160, 160, 128)
        self.fusion4 = fusionmodule(256, 256, 128)

        # decode Module
        self.up2X = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # nn.AdaptiveAvgPool2d(output_size)  # 自适应平均池化
        # nn.AdaptiveMaxPool2d(output_size)  # 自适应最大池化

        self.fgdecoder = fgdecoder()
        self.edgedecoder = edgedecoder()


        self.downsample = nn.AdaptiveAvgPool2d(8)



        """ Adjust the shape of decouple output """
        self.preprocess_fg4 = Preprocess(128, 128, 8)  # 1/16
        self.preprocess_bg4 = Preprocess(128, 128, 8)  # 1/16

        self.preprocess_fg3 = Preprocess(128, 128, 4)  # 1/8
        self.preprocess_bg3 = Preprocess(128, 128, 4)  # 1/8

        self.preprocess_fg2 = Preprocess(128, 128, 2)  # 1/4
        self.preprocess_bg2 = Preprocess(128, 128, 2)  # 1/4

        self.preprocess_fg1 = Preprocess(128, 128, 1)  # 1/2
        self.preprocess_bg1 = Preprocess(128, 128, 1)  # 1/2


        self.feag4 = FEAGModule(128, 128)
        self.feag3 = FEAGModule(128 + 128, 128)
        self.feag2 = FEAGModule(128 + 128, 128)
        self.feag1 = FEAGModule(128 + 128, 128)


        self.decoder_final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))


    def forward(self, A):
        size = A.size()[2:]
        channels = torch.split(A, split_size_or_sections=1, dim=1)
        new_inputs = torch.cat(channels * 3, dim=1)
        y = self.conv(self.hpf_conv(new_inputs))
        # print(y.shape, "222")
        layer1_A, layer2_A, layer3_A, layer4_A = self.extract(y)

        feature_g = self.backbone_recon(A)


        feature_fusion1 = self.fusion1(layer1_A, feature_g[0])
        feature_fusion2 = self.fusion2(layer2_A, feature_g[1])
        feature_fusion3 = self.fusion3(layer3_A, feature_g[2])
        feature_fusion4 = self.fusion4(layer4_A, feature_g[3])


        ### 特征解码
        """ Decouple Layer """
        fg_map, fg_head = self.fgdecoder(feature_fusion2, feature_fusion3, feature_fusion4)
        edge_map, edge_head = self.edgedecoder(feature_fusion1)


        """ Contrast-Driven Feature Aggregation """
        f_fg4 = self.preprocess_fg4(fg_map)
        f_edge4 = self.preprocess_bg4(edge_map)
        f_fg3 = self.preprocess_fg3(fg_map)
        f_edge3 = self.preprocess_bg3(edge_map)
        f_fg2 = self.preprocess_fg2(fg_map)
        f_edge2 = self.preprocess_bg2(edge_map)
        f_fg1 = self.preprocess_fg1(fg_map)
        f_edge1 = self.preprocess_bg1(edge_map)

        f4 = self.feag4(feature_fusion4, f_fg4, f_edge4)

        f4_up = self.up2X(f4)

        f_4_3 = torch.cat([feature_fusion3, f4_up], dim=1)
        f3 = self.feag3(f_4_3, f_fg3, f_edge3)
        f3_up = self.up2X(f3)
        f_3_2 = torch.cat([feature_fusion2, f3_up], dim=1)
        f2 = self.feag2(f_3_2, f_fg2, f_edge2)
        f2_up = self.up2X(f2)
        f_2_1 = torch.cat([feature_fusion1, f2_up], dim=1)
        f1 = self.feag1(f_2_1, f_fg1, f_edge1)

        output_map = self.decoder_final(f1)
        output_map = F.interpolate(output_map, size, mode='bilinear', align_corners=True)
        fg_head = F.interpolate(fg_head, size, mode='bilinear', align_corners=True)
        edge_head = F.interpolate(edge_head, size, mode='bilinear', align_corners=True)

        return output_map, fg_head, edge_head



if __name__=='__main__':
    net = Basenet().cuda()
    out = net(torch.rand((2, 3, 256, 256)).cuda())[0]
    print(out.shape)

    net = Basenet()  # 替换成你自己的模型类或实例化对象
    device = torch.device("cpu")

    input = torch.randn(1, 3, 256, 256)

    flops, params = profile(net, inputs=(input,))

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))