import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv2d(self.channel, self.channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.relu = nn.ELU()
        self.conv2 = nn.Conv2d(self.channel, self.channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.channel)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = residual + x
        x = self.relu(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, num_classes = 20):
        super(SimpleNet, self).__init__()
        self.dropout = nn.Dropout2d(0.2)
        self.conv1 = nn.Conv2d(3,32,5,2,padding = 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 2, padding = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, padding = 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.res3 = ResidualBlock(128)
        self.conv4 = nn.Conv2d(128, 256, 5, 2, padding = 2)
        self.bn4 = nn.BatchNorm2d(256)
        self.res4 = ResidualBlock(256)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.res1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.res2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.res3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = self.res4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x