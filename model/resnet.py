from torchvision import models
import torch.nn as nn

from model.component import Flatten, DebugLayer


class ResNet(nn.Module):
  def __init__(self, classes, bottleneck=False, pretrained=False):
    super(ResNet, self).__init__()

    model = models.resnet50(pretrained=pretrained)
    self.feature_extractor = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool,
        Flatten(),
    )
    if bottleneck:
      feature_dim = 256
      self.bottleneck = nn.Sequential(
          # Bottleneck layer
          nn.Linear(2048, 256),
          nn.ReLU(inplace=True),
          nn.Dropout(),
      )
    else:
      feature_dim = 2048

    self.classifier = nn.Sequential(
        nn.Linear(feature_dim, classes),
    )