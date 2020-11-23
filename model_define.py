import os
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class ASPP(nn.Module):
    # have bias and relu, no bn
    def __init__(self, in_channel=512, depth=256):
        super().__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                  nn.ReLU(inplace=True))

        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                           nn.ReLU(inplace=True))
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3),
            nn.ReLU(inplace=True))
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),
            nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channel, depth, 3, 1, padding=9, dilation=9),
            nn.ReLU(inplace=True))

        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 5, depth, 1, 1),
                                             nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features,
                                       size=size,
                                       mode='bilinear',
                                       align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(
            torch.cat([
                image_features, atrous_block1, atrous_block6, atrous_block12,
                atrous_block18
            ],
                      dim=1))
        return net


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=18,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Deeplab_v3(nn.Module):
    # in_channel = 3 fine-tune
    def __init__(self, class_number=18):
        super().__init__()
        encoder = resnet50()
        self.start = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)

        self.maxpool = encoder.maxpool

        self.low_feature1 = nn.Sequential(nn.Conv2d(64, 32, 1, 1),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(inplace=True))
        self.low_feature3 = nn.Sequential(nn.Conv2d(256, 64, 1, 1),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True))
        self.low_feature4 = nn.Sequential(nn.Conv2d(512, 128, 1, 1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(inplace=True))

        self.layer1 = encoder.layer1  #256
        self.layer2 = encoder.layer2  #512
        self.layer3 = encoder.layer3  #1024
        self.layer4 = encoder.layer4  #2048

        self.aspp = ASPP(in_channel=2048, depth=256)

        self.conv_cat4 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, 3, 1, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv_cat3 = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, 1, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.Conv2d(256, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv_cat1 = nn.Sequential(nn.Conv2d(64 + 32, 64, 3, 1, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 18, 3, 1, padding=1))

    def forward(self, x):
        size0 = x.shape[2:]  # need upsample input size
        x1 = self.start(x)  # 64,  128*128
        x2 = self.maxpool(x1)  # 64,  64*64
        x3 = self.layer1(x2)  # 256, 64*64
        x4 = self.layer2(x3)  # 512, 32*32
        x5 = self.layer3(x4)  # 1024,16*16
        x = self.layer4(x5)  # 2048,8*8
        x = self.aspp(x)  # 256, 8*8

        low_feature1 = self.low_feature1(x1)  # 64,  128*128
        # low_feature2 = self.low_feature2(x2) # 64,  64*64
        low_feature3 = self.low_feature3(x3)  # 256, 64*64
        low_feature4 = self.low_feature4(x4)  # 512, 32*32 -> 128, 32*32
        # low_feature5 = self.low_feature5(x5) # 1024,16*16

        size1 = low_feature1.shape[2:]
        # size2 = low_feature2.shape[2:]
        size3 = low_feature3.shape[2:]
        size4 = low_feature4.shape[2:]
        # size5 = low_feature5.shape[2:]

        decoder_feature4 = F.interpolate(x,
                                         size=size4,
                                         mode='bilinear',
                                         align_corners=True)
        x = self.conv_cat4(torch.cat([low_feature4, decoder_feature4], dim=1))

        decoder_feature3 = F.interpolate(x,
                                         size=size3,
                                         mode='bilinear',
                                         align_corners=True)
        x = self.conv_cat3(torch.cat([low_feature3, decoder_feature3], dim=1))

        decoder_feature1 = F.interpolate(x,
                                         size=size1,
                                         mode='bilinear',
                                         align_corners=True)
        x = self.conv_cat1(torch.cat([low_feature1, decoder_feature1], dim=1))

        score = F.interpolate(x,
                              size=size0,
                              mode='bilinear',
                              align_corners=True)

        return score


def init_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    model = Deeplab_v3()
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
