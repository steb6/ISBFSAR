import torch
import torchvision.models as models


class MutualGazeDetector(torch.nn.Module):
    def __init__(self):
        super(MutualGazeDetector, self).__init__()
        self.backbone = ResNetModel()
        # self.norm = torch.nn.LayerNorm(1000)  # TODO MMM
        self.classifier = BinaryClassifier(self.backbone.model.fc.out_features)

    def forward(self, x):
        features = self.backbone(x)
        # features = self.norm(features)  # TODO MMM
        return self.classifier(features)


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 250)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(250, 64)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(64, 1)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.layer1(x)
        y = self.act1(y)
        y = self.layer2(y)
        y = self.act2(y)
        y = self.layer3(y)
        return self.out(y)


class ResNetModel(torch.nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.model(x)
