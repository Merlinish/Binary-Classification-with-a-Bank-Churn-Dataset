import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
from tensorboardX import SummaryWriter
import numpy as np


class ResNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=num_classes, kernel_size=(6, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(num_classes)

        self.conv2 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=(3, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        self.conv3 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=(3, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(num_classes)

        self.conv4 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_classes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        return x


if __name__ == '__main__':
    # size = 14
    net = ResNet(input_size=1, num_classes=1).to(device="cuda")
    summary(net, (1, 10, 1), batch_size=1, device="cuda")

    rand_input = torch.rand(1, 1, 10, 1)

    model1 = net()
    with SummaryWriter(logdir="../model/tensorboard/model", comment="resnet") as w:
        w.add_graph(net, (rand_input,))
