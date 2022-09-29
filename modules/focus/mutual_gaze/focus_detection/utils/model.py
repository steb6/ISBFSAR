import torch
import torchvision.models as models
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1


class JustOpenPose(torch.nn.Module):
    def __init__(self):
        super(JustOpenPose, self).__init__()
        self.layer1 = torch.nn.Linear(14*2, 128)
        self.act1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.1)
        self.layer2 = torch.nn.Linear(128, 64)
        self.act2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(0.1)
        self.layer3 = torch.nn.Linear(64, 1)
        self.act3 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.layer3(x)
        x = self.act3(x)
        return x


class MutualGazeDetectorHeads(torch.nn.Module):
    def __init__(self, model, pretrained=True):
        super(MutualGazeDetectorHeads, self).__init__()
        self.backbone = BackBone(model, pretrained)
        self.classifier = BinaryClassifier(512)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class MutualGazeDetectorOPE(torch.nn.Module):
    def __init__(self, model, pretrained):
        super(MutualGazeDetectorOPE, self).__init__()
        self.backbone = BackBone(model, pretrained)
        self.classifier = BinaryClassifier(1000 + 57)

    def forward(self, x, f):
        eye_features = self.backbone(x)
        features = torch.concat((eye_features, f), dim=1)
        return self.classifier(features)


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 256)
        self.drop1 = torch.nn.Dropout(0.1)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(256, 64)
        self.drop2 = torch.nn.Dropout(0.1)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(64, 1)
        self.out = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.layer1(x)
        y = self.drop1(y)
        y = self.act1(y)
        y = self.layer2(y)
        y = self.drop2(y)
        y = self.act2(y)
        y = self.layer3(y)
        return self.out(y)


class BackBone(torch.nn.Module):
    def __init__(self, model, pretrained=True):
        super(BackBone, self).__init__()
        if model == "resnet":
            self.model = models.resnet50(pretrained=pretrained)
        elif model == "mnet":
            self.model = models.mobilenet_v3_small(pretrained=pretrained)
        elif model == "facenet":
            self.model = InceptionResnetV1(pretrained='vggface2')

    def forward(self, x):
        return self.model(x)
