import torch
import torch.nn as nn

affine_par = True


class Restoretest(nn.Module):
    def __init__(self, args):
        super(Restoretest, self).__init__()
        G0 = 32
        kSize = 3  # 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = 8, 2, 32

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, x):
        with torch.no_grad():
            f__1 = self.SFENet1(x)
            x = self.SFENet2(f__1)  # 64 32 32

            RDBs_out = []
            for i in range(self.D):
                x = self.RDBs[i](x)
                # draw_features(64, x[0],"./image/encoder{}.png".format(i))
                RDBs_out.append(x)  # 64 32 32

            x = self.GFF(torch.cat(RDBs_out, 1))
            # draw_features(64, x[0],"./image/encoder{}.png".format(i))
            x += f__1  # 64 32 32
            result = self.UPNet(x)

        return result, RDBs_out




class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x




class Restore(nn.Module):
    def __init__(self, args):
        super(Restore, self).__init__()
        G0 = 32
        kSize = 3  # 3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = 8, 2, 32

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)  # 64 32 32

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            # draw_features(64, x[0],"./image/encoder{}.png".format(i))
            RDBs_out.append(x)  # 64 32 32

        x = self.GFF(torch.cat(RDBs_out, 1))
        # draw_features(64, x[0],"./image/encoder{}.png".format(i))
        x += f__1  # 64 32 32
        result = self.UPNet(x)

        return result, RDBs_out
