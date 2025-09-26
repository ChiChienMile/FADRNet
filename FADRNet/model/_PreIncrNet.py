import torch
from torch import nn

from model.Backbone.SENet import SENet3D
from model.Backbone.SENet import SENetBottleneck

# from Backbone.SENet import SENet3D
# from Backbone.SENet import SENetBottleneck

class BackBone3D(nn.Module):
    def __init__(self, in_channels=2, stride_list=[2, 2, 2, 2], channel_list=[64, 128, 256, 512, 512]):
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

class PreIncrNet(nn.Module):
    def __init__(self, basic_task, in_channels=1, n_classes=2):
        super(PreIncrNet, self).__init__()
        self.name = 'PreIncrNet_' + basic_task
        self.backbone = BackBone3D(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.CLSmodel = nn.Linear(1024, n_classes)

    def flatten(self, features):
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return features

    def forward(self, x, label, loss_ce):
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        features = self.flatten(layer4)
        logits = self.CLSmodel(features)
        loss = loss_ce(logits, label)
        return loss, logits

    def predictcls(self, x):
        with torch.no_grad():
            layer0 = self.backbone.layer0(x)
            layer1 = self.backbone.layer1(layer0)
            layer2 = self.backbone.layer2(layer1)
            layer3 = self.backbone.layer3(layer2)
            layer4 = self.backbone.layer4(layer3)
            features = self.flatten(layer4)
            logits = self.CLSmodel(features)
            return logits

    def get_features(self, x):
        with torch.no_grad():
            layer0 = self.backbone.layer0(x)
            layer1 = self.backbone.layer1(layer0)
            layer2 = self.backbone.layer2(layer1)
            layer3 = self.backbone.layer3(layer2)
            layer4 = self.backbone.layer4(layer3)
            features = self.flatten(layer4)
            return features


if __name__ == "__main__":
    imcls = torch.randn(1, 1, 160, 192, 160)
    model = PreIncrNet(basic_task='IDH', in_channels=1, n_classes=2)
    logist = model.predictcls(imcls)
    print(logist.size())