import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
affine_par = True
from blocks import Bottleneck, _FPM, BR
from Restore import Restoretest

base_dir = os.path.dirname(__file__) #recently file location

class MFLnet(nn.Module):  # only loaclization without detection, FLDCF include localization and detection
    def __init__(self, args, block=Bottleneck, layers=[1, 2, 2, 1], num_classes=2, aux=True):
        self.inplanes = 64
        self.aux = aux
        super(MFLnet, self).__init__()

        self.learned = Restoretest(args)
        if args.dataset == 'Fake-LoveDA':
            name = 'model_lo.pt'
        if args.dataset == 'Fake-Vaihingen':
            name = 'model_vi.pt'
        if args.dataset == 'HRCUS_FAKE':
            name = 'model_hr.pt' ## 추후 데이터셋 학습 필요함!!!!!!!!!!
        self.learned.load_state_dict(
            torch.load(
                os.path.join(base_dir,'MFLnet_model_weights', name),
            ),
            strict=True
        )

        self.handlelern1 = nn.Conv2d(32 * 4, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.handlelern2 = nn.Conv2d(32 * 4, 64, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, affine=affine_par)
        self.conv3 = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, affine=affine_par)

        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.res5_con1x1 = nn.Sequential(
            nn.Conv2d(1024 + 2048, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.fpm1 = _FPM(512, num_classes)
        self.br1 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)

        self.predict1 = self._predict_layer(512 * 6, num_classes)

        dropout = 0.9

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _predict_layer(self, in_channels, num_classes):
        return nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0),
                             nn.BatchNorm2d(256),
                             nn.ReLU(True),
                             nn.Dropout2d(0.1),
                             nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def base_forward(self, x, RDNout):
        # draw_features(64, RDNout[4][0],"{}/0.png".format('./image'))
        # draw_features(64, RDNout[5][0],"{}/1.png".format('./image'))
        # draw_features(64, RDNout[6][0],"{}/2.png".format('./image'))
        # draw_features(64, RDNout[7][0],"{}/3.png".format('./image'))
        rdn1 = self.handlelern1(torch.cat(RDNout[0:4], 1))
        rdn2 = self.handlelern2(torch.cat(RDNout[4:8], 1))
        x = self.relu(self.bn1(self.conv1(x)))
        size_conv1 = x.size()[2:]
        x = self.relu(self.bn2(self.conv2(torch.cat([x, rdn1], dim=1))))
        # draw_features(64, rdn2[0],"{}/rdn.png".format('./image'))
        # draw_features(64, x[0],"{}/x.png".format('./image'))
        x = self.relu(self.bn3(self.conv3(torch.cat([x, rdn2], dim=1))))
        x = self.maxpool(x)
        x = self.layer1(x)
        res2 = x
        x = self.layer2(x)
        res3 = x
        x = self.layer3(x)
        res4 = x
        x = self.layer4(x)
        x = self.res5_con1x1(torch.cat([x, res4], dim=1))

        return x, res3, res2, size_conv1

    def forward(self, x):
        result, RDNout = self.learned(x)
        b, c, w, h = x.size()
        size = x.size()[2:]
        score1, score2, score3, size_conv1 = self.base_forward(x, RDNout)
        score1 = self.fpm1(score1)
        score1 = self.predict1(score1)  # 1/8
        score1 = self.br1(score1)
        score2 = score1

        # second fusion
        size_score3 = score3.size()[2:]
        score3 = F.interpolate(score2, size_score3, mode='bilinear', align_corners=True)
        score3 = self.br5(score3)
        # draw_features(64, score3[:,0],"{}/decoder1.png".format('./image'))

        # upsampling + BR
        score3 = F.interpolate(score3, size_conv1, mode='bilinear', align_corners=True)
        score3 = self.br6(score3)
        # draw_features(64, score3[:,0],"{}/decoder2.png".format('./image'))
        score3 = F.interpolate(score3, size, mode='bilinear', align_corners=True)
        score3 = self.br7(score3)

        return score3, None


if __name__ == "__main__":
    import torch

    class Args:
        dataset = "Fake-LoveDA"  # 원본 코드 분기용. 안 쓰면 아무 값이나 둬도 됨.

    args = Args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MFLnet(args, num_classes=2).to(device)
    model.eval()

    x = torch.randn(1, 3, 256, 256).to(device)

    with torch.no_grad():
        seg, aux = model(x)

    print("=== MFLnet test ===")
    print("input shape :", x.shape)
    print("seg shape   :", seg.shape)
    print("aux output  :", aux)