import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.LeakyReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid())
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SENetBottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None, reduction=16):
        super(SENetBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.se = SELayer3D(planes * self.expansion, reduction)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SENetDilatedBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(SENetDilatedBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.se = SELayer3D(planes * self.expansion, reduction=16)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SENet3D(nn.Module):
    def __init__(self, block, layers, channel_list, stride_list, in_channels=2,
                 shortcut_type='B', cardinality=32, num_classes=2):
        super(SENet3D, self).__init__()
        self.inplanes = channel_list[0]

        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=stride_list[0], padding=(3, 3, 3), bias=False)
        self.gn1 = nn.GroupNorm(8, self.inplanes)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=stride_list[1], padding=1)
        self.layer1 = self._make_layer(block, channel_list[1], layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, channel_list[2], layers[1], shortcut_type, cardinality, stride=stride_list[2])
        self.layer3 = self._make_layer(block, channel_list[3], layers[2], shortcut_type, cardinality, stride=stride_list[3])
        self.layer4 = self._make_layer(SENetDilatedBottleneck, channel_list[4], layers[3], shortcut_type, cardinality, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def senet3d10(**kwargs):
    """Constructs a SENet3D-10 model."""
    model = SENet3D(SENetBottleneck, [1, 1, 1, 1], **kwargs)
    return model


def senet3d18(**kwargs):
    """Constructs a SENet3D-18 model."""
    model = SENet3D(SENetBottleneck, [2, 2, 2, 2], **kwargs)
    return model


def senet3d34(**kwargs):
    """Constructs a SENet3D-34 model."""
    model = SENet3D(SENetBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def senet3d50(**kwargs):
    """Constructs a SENet3D-50 model."""
    model = SENet3D(SENetBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def senet3d101(**kwargs):
    """Constructs a SENet3D-101 model."""
    model = SENet3D(SENetBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def senet3d152(**kwargs):
    """Constructs a SENet3D-152 model."""
    model = SENet3D(SENetBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def senet3d200(**kwargs):
    """Constructs a SENet3D-200 model."""
    model = SENet3D(SENetBottleneck, [3, 24, 36, 3], **kwargs)
    return model


class BackBone3D(nn.Module):
    def __init__(self, in_channels, channel_list, stride_list):
        super(BackBone3D, self).__init__()
        net = SENet3D(SENetBottleneck, [3, 4, 6, 3], num_classes=2,
                      in_channels=in_channels,
                      channel_list=channel_list, stride_list=stride_list)
        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3:5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4

if __name__ == '__main__':
    img = torch.randn(2, 2, 160, 192, 160)
    net = BackBone3D(in_channels=2, channel_list=[64, 128, 256, 512, 512],  stride_list=[2, 2, 2, 2])
    out = net(img)
    print(out.shape)

    img = torch.randn(2, 2, 80, 96, 80)
    net = BackBone3D(in_channels=2, channel_list=[16, 32, 64, 128, 128],  stride_list=[2, 2, 2, 1])
    out = net(img)
    print(out.shape)

    img = torch.randn(2, 2, 40, 40, 40)
    net = BackBone3D(in_channels=2, channel_list=[16, 32, 64, 128, 128],  stride_list=[2, 2, 1, 1])
    out = net(img)
    print(out.shape)

    img = torch.randn(2, 2, 20, 24, 20)
    net = BackBone3D(in_channels=2, channel_list=[16, 32, 64, 128, 128], stride_list=[2, 1, 1, 1])
    out = net(img)
    print(out.shape)

    img = torch.randn(2, 2, 10, 12, 10)
    net = BackBone3D(in_channels=2, channel_list=[16, 32, 64, 128, 128], stride_list=[1, 1, 1, 1])
    out = net(img)
    print(out.shape)