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
        resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

        assert len(dict(resnet18.named_parameters()).keys()) == len(
            resnet18.state_dict().keys()), 'More BN layers are there...'

        self.model = resnet18

    def forward(self, x):

        x = self.model(x)

        return x




class WideBasicBlockGN(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0.0, num_groups=16):
        super(WideBasicBlockGN, self).__init__()

        assert in_planes % num_groups == 0, "in_planes must be divisible by num_groups"
        assert out_planes % num_groups == 0, "out_planes must be divisible by num_groups"

        # Pre-activation: GN -> ReLU -> Conv
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.dropout_rate = dropout_rate
        self.equal_in_out = (in_planes == out_planes) and (stride == 1)

        # Projection shortcut when shape changes
        self.shortcut = None
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
            )

    def forward(self, x):
        out = self.relu1(self.gn1(x))

        if self.equal_in_out:
            shortcut = x
            out = self.conv1(out)
        else:
            shortcut = self.shortcut(out)
            out = self.conv1(out)

        out = self.relu2(self.gn2(out))
        if self.dropout_rate > 0.0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)

        out = out + shortcut
        return out


class NetworkBlockGN(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0, num_groups=16):
        super(NetworkBlockGN, self).__init__()

        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    in_planes=in_planes if i == 0 else out_planes,
                    out_planes=out_planes,
                    stride=stride if i == 0 else 1,
                    dropout_rate=dropout_rate,
                    num_groups=num_groups,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WRN28_4_CIFAR100(nn.Module):
    def __init__(self, args=None, num_classes=100, depth=28, widen_factor=4,
                 dropout_rate=0.0, num_groups=16):
        super(WRN28_4_CIFAR100, self).__init__()

        assert (depth - 4) % 6 == 0, "For WRN, depth should satisfy (depth - 4) % 6 == 0"
        n = (depth - 4) // 6   # WRN-28-* => n = 4

        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        # For WRN-28-4: [16, 64, 128, 256]

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.block1 = NetworkBlockGN(
            nb_layers=n,
            in_planes=n_channels[0],
            out_planes=n_channels[1],
            block=WideBasicBlockGN,
            stride=1,
            dropout_rate=dropout_rate,
            num_groups=num_groups
        )

        self.block2 = NetworkBlockGN(
            nb_layers=n,
            in_planes=n_channels[1],
            out_planes=n_channels[2],
            block=WideBasicBlockGN,
            stride=2,
            dropout_rate=dropout_rate,
            num_groups=num_groups
        )

        self.block3 = NetworkBlockGN(
            nb_layers=n,
            in_planes=n_channels[2],
            out_planes=n_channels[3],
            block=WideBasicBlockGN,
            stride=2,
            dropout_rate=dropout_rate,
            num_groups=num_groups
        )

        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)       # [B, 16, 32, 32]
        out = self.block1(out)    # [B, 64, 32, 32]
        out = self.block2(out)    # [B, 128, 16, 16]
        out = self.block3(out)    # [B, 256, 8, 8]
        out = self.relu(self.gn(out))
        out = self.avgpool(out)   # [B, 256, 1, 1]
        out = torch.flatten(out, 1)
        out = self.fc(out)        # [B, num_classes]
        return out
