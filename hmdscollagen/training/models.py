import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from scipy import ndimage, misc

class SimpleNet(nn.Module):
  def __init__(self,
               num_classes=10,
               final_channels=1):

    super(SimpleNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()

    self.pool = nn.MaxPool2d(kernel_size=2)

    self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
    self.relu3 = nn.ReLU()

    self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
    self.relu4 = nn.ReLU()

    self.final_layer = nn.Conv2d(in_channels=24, out_channels=final_channels, kernel_size=3, stride=1, padding=1)

    self.fc = nn.Linear(in_features=64 * 128 * 1, out_features=num_classes)

  def forward(self, input, out_shape=None):

    output = self.conv1(input)
    output = self.relu1(output)

    output = self.conv2(output)
    output = self.relu2(output)

    output = self.pool(output)  # Max pooling

    output = self.conv3(output)
    output = self.relu3(output)

    output = self.conv4(output)
    output = self.relu4(output)

    output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)  # Could be 'bilinear'

    output = self.final_layer(output)

    # output = output.view(-1, 16 * 16 * 24)

   # output = self.fc(output)
    if out_shape is not None:
      return F.interpolate(output, size=out_shape, mode='bicubic', align_corners=True)
    else:
      return output
