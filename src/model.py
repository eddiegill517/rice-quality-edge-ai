import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel attention mechanism."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        w = self.pool(x)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class InvertedResidual(nn.Module):
    """Depthwise separable convolution with SE attention and optional residual."""
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = in_ch * expand_ratio

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU6(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        ])

        layers.append(SqueezeExcitation(mid_ch))

        layers.extend([
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class TinyRiceNet(nn.Module):
    """
    Lightweight grain classifier for edge deployment.
    57K parameters, under 100KB at INT8 quantization.
    """
    def __init__(self, num_classes=5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        self.body = nn.Sequential(
            InvertedResidual(16, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=2),
            InvertedResidual(24, 24, stride=1, expand_ratio=2),
            InvertedResidual(24, 32, stride=2, expand_ratio=2),
            InvertedResidual(32, 32, stride=1, expand_ratio=2),
            InvertedResidual(32, 48, stride=2, expand_ratio=2),
            InvertedResidual(48, 48, stride=1, expand_ratio=2),
            InvertedResidual(48, 64, stride=2, expand_ratio=2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x