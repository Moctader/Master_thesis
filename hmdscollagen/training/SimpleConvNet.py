import torch.nn.functional as F
from torch import nn
from collagen.core import Module


class SimpleConvNet(Module):
    def __init__(self, bw=32, drop=0.5, n_cls=128, n_channels=1):
        super(SimpleConvNet, self).__init__()
        self.n_filters_last = bw*2

        self.conv1 = self.make_layer(n_channels, bw)
        self.conv2 = self.make_layer(bw, bw*2)
        self.conv3 = self.make_layer(bw*2, self.n_filters_last)

        self.classifier = nn.Sequential(nn.Dropout(drop),
                                        nn.Linear(self.n_filters_last, n_cls))

    @staticmethod
    def make_layer(inp, out):
        return nn.Sequential(nn.Conv2d(inp, out, 3, 1, 1),
                             nn.BatchNorm2d(out),
                             nn.ReLU(True))

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)  # 16x16
        x = F.max_pool2d(self.conv2(x), 2)  # 8x8
        x = F.max_pool2d(self.conv3(x), 2)  # 4x4

        x = F.adaptive_avg_pool2d(x, 1)


        x = x.view(x.size(0),-1,1,64)


        return self.classifier(x)

