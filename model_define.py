import os
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class DownConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.maxpooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.down_conv(x)
        x = self.maxpooling(x1)
        return x, x1


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Encoder4(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = DoubleConv(in_channels, out_channels)
        self.conv1 = DoubleConv(out_channels, out_channels)
        self.conv2 = DoubleConv(out_channels, out_channels)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # 卷积块，6层
        x = self.conv0(x)
        x1 = self.conv1(x)
        x1 = x1 + x
        x1 = self.conv2(x1)
        # 池化
        x = self.maxpooling(x1)
        return x, x1


class Encoder5(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = DoubleConv(in_channels, out_channels)
        self.conv1 = DoubleConv(out_channels, out_channels)
        self.conv2 = DoubleConv(out_channels, out_channels)
        self.conv3 = DoubleConv(out_channels, out_channels)
        self.conv4 = DoubleConv(out_channels, out_channels)
        self.conv5 = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        # 卷积块，12层
        x = self.conv0(x)
        x1 = self.conv1(x)
        x1 = x1 + x
        x2 = self.conv2(x1)
        x2 = x2 + x1
        x3 = self.conv3(x2)
        x3 = x3 + x2
        x4 = self.conv4(x3)
        x4 = x4 + x3
        x4 = self.conv5(x4)
        return x4


class UpConv(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=18):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.encoder1 = DownConv(3, 32)  # 256^2 -> 128^2
        self.encoder2 = DownConv(32, 64)  # 128^2 -> 64^2
        self.encoder3 = DownConv(64, 128)  # 64^2 -> 32^2
        self.encoder4 = Encoder4(128, 256)  # 32^2 -> 16^2
        self.encoder5 = Encoder5(256, 512)  # 16^2 -> 16^2

        self.decoder4 = UpConv(512 + 256, 256)
        self.decoder3 = UpConv(256 + 128, 128)
        self.decoder2 = UpConv(128 + 64, 64)
        self.decoder1 = UpConv(64 + 32, 32)

        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x, x1 = self.encoder1(x)
        x, x2 = self.encoder2(x)
        x, x3 = self.encoder3(x)
        x, x4 = self.encoder4(x)
        x = self.encoder5(x)
        x = self.decoder4(x, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)

        logits = self.outc(x)
        return logits


def init_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    model = UNet(3, 18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_state = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in model_state["model_state_dict"].items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
