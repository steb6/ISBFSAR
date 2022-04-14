import torch
import torchvision.models as models


class MutualGazeDetectorHeads(torch.nn.Module):
    def __init__(self):
        super(MutualGazeDetectorHeads, self).__init__()
        self.backbone = ResNetModel()
        self.classifier = BinaryClassifier(1000)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class MutualGazeDetectorOPE(torch.nn.Module):
    def __init__(self):
        super(MutualGazeDetectorOPE, self).__init__()
        self.backbone = ResNetModel()
        self.classifier = BinaryClassifier(1000 + 57)

    def forward(self, x, f):
        eye_features = self.backbone(x)
        features = torch.concat((eye_features, f), dim=1)
        return self.classifier(features)


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 256)
        self.drop1 = torch.nn.Dropout(0.2)
        self.act1 = torch.nn.ReLU()
        # self.layer2 = torch.nn.Linear(256, 64)
        # self.drop2 = torch.nn.Dropout(0.2)
        # self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(256, 1)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.layer1(x)
        y = self.drop1(y)
        y = self.act1(y)
        # y = self.layer2(y)
        # y = self.drop2(y)
        # y = self.act2(y)
        y = self.layer3(y)
        return self.out(y)


class ResNetModel(torch.nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        # self.model = models.resnet18(pretrained=True)
        self.model = models.mobilenet_v3_small(pretrained=True)
        # self.model = models.vgg19(pretrained=True)
        # self.model = models.inception_v3(pretrained=True)

    def forward(self, x):
        return self.model(x)
