from torch.autograd import Variable
import torch.nn.functional as F
import torch

# List all the models here

'''
class SimpleCNN(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 16 * 16)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return (x)
    '''
"""
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class SimpleNet(nn.Module):
      def __init__(self,
                 num_classes=10,
                 final_channels=3,
                 # out_shape=None,
                 # out_shape=(512, 1024)
                 ):

        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.final_layer = nn.Conv2d(in_channels=24, out_channels=final_channels, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)

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
            return F.interpolate(output, size=out_shape, mode='bilinear', align_corners=True)
        else:
            return output.permute(0, 2, 3, 1)

      input = torch.rand((512, 1024))

"""

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout


class SimpleNet(nn.Module):
    def __init__(self,
                 num_classes=10,
                 final_channels=3):

        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.final_layer = nn.Conv2d(in_channels=24, out_channels=final_channels, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)

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

        output = F.interpolate(output, scale_factor=2, mode='bicubic', align_corners=True)  # Could be 'bilinear'

        output = self.final_layer(output)

        # output = output.view(-1, 16 * 16 * 24)

        # output = self.fc(output)
        if out_shape is not None:
            return F.interpolate(output, size=out_shape, mode='bicubic', align_corners=True)
        else:
            return output.permute(0, 2, 3, 1)


"""

class Net(nn.Module):
  def __init__(self,
               num_classes=10,
               final_channels=3):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()

    self.pool = nn.MaxPool2d(kernel_size=2)

    self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
    self.relu3 = nn.ReLU()

    self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
    self.relu4 = nn.ReLU()

    self.final_layer = nn.Conv2d(in_channels=24, out_channels=final_channels, kernel_size=3, stride=1, padding=1)

    self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)

  def forward(self, x, out_shape=None):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(self.conv2(x)), 2))
    x = x.view(x.size(0), -1)  # Flatten layer
    x = F.dropout(x, training=self.training)
    if out_shape is not None:
      return F.interpolate(x, size=out_shape, mode='bilinear', align_corners=True)
    else:
      return x.permute(0, 2, 3, 1)



import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

class SimpleNet(nn.Module):
      def __init__(self,
                 num_classes=10,
                 final_channels=3):

        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.final_layer = nn.Conv2d(in_channels=24, out_channels=final_channels, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)

      def forward(self, output, out_shape=None):

        output = self.conv1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool(output)  # Max pooling

        output = self.conv3(output)
        output = self.relu3(output)

        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)  # Could be 'bilinear'

        output = self.final_layer(output)

        # output = output.view(-1, 16 * 16 * 24)

        # output = self.fc(output)
        if out_shape is not None:
            return F.interpolate(output, size=out_shape, mode='bilinear', align_corners=True)
        else:
            return output.permute(0, 2, 3, 1)
"""
