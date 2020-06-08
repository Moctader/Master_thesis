import torch.nn.functional as F
from torch import nn
from collagen.core import Module


class SimpleConvNet(Module):
    def __init__(self, bw=1, drop=0.5, n_cls=10, n_channels=1):
        super(SimpleConvNet, self).__init__()
        self.n_filters_last = bw

        self.conv1 = self.make_layer(n_channels, bw*32)
        self.conv2 = self.make_layer(bw*32, bw)
        self.conv3 = self.make_layer(bw, self.n_filters_last)

        self.classifier = nn.Sequential(nn.Dropout(drop),
                                        nn.Linear(self.n_filters_last, n_cls))

    @staticmethod
    def make_layer(inp, out):
        return nn.Sequential(nn.Conv2d(inp, out, 1),
                             nn.BatchNorm2d(out),
                             nn.ReLU(True))

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)  # 16x16
        x = F.max_pool2d(self.conv2(x), 2)  # 8x8
        x = F.max_pool2d(self.conv3(x), 2)  # 4x4

        #x = F.adaptive_avg_pool2d(x, 1)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        #x = x.view(x.size(0),-1,64,1)


        return x

