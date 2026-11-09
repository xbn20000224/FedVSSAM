import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
"""Network definitions used by the training entry point."""


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x




class CNNCifar10(nn.Module):
    def __init__(self, args):
        super(CNNCifar10, self).__init__()
        self.n_cls = 10


        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # pool -> 16x5x5

        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 16, 5, 5]
        x = x.view(-1, 16*5*5)                 # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, num_groups=16):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_CIFAR100(nn.Module):
    def __init__(self, args=None, num_classes=10):
        super(ResNet18_CIFAR100, self).__init__()
        resnet18 = models.resnet18()
        resnet18.fc = nn.Linear(512, num_classes)

        # Change BN to GN
        resnet18.bn1 = nn.GroupNorm(num_groups=16, num_channels=64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=16, num_channels=64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=16, num_channels=64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=16, num_channels=64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=16, num_channels=64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=16, num_channels=128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=16, num_channels=128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=16, num_channels=128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=16, num_channels=128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=16, num_channels=128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=16, num_channels=256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=16, num_channels=256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=16, num_channels=256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=16, num_channels=256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=16, num_channels=256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=16, num_channels=512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=16, num_channels=512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=16, num_channels=512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=16, num_channels=512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=16, num_channels=512)

        assert len(dict(resnet18.named_parameters()).keys()) == len(
            resnet18.state_dict().keys()), 'More BN layers are there...'

        self.model = resnet18

    def forward(self, x):

        x = self.model(x)

        return x




