import torch
import torch.nn as nn
from torchvision import models


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn2(self.conv2(self.prelu(self.bn1(self.conv1(x)))))



class Generator(nn.Module):
    def __init__(self, n_res_blocks=16):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(n_res_blocks)])
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
       
        up_blocks = []
        for _ in range(2):
            up_blocks += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*up_blocks)
        self.block3 = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.res_blocks(x1)
        x3 = self.block2(x2)
        x = x1 + x3
        x = self.upsample(x)
        return torch.tanh(self.block3(x))



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_ch, out_ch, stride=1):
            return [
                nn.Conv2d(in_ch, out_ch, 3, stride, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            *block(64, 64, 2),
            *block(64, 128),
            *block(128, 128, 2),
            *block(128, 256),
            *block(256, 256, 2),
            *block(256, 512),
            *block(512, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:36]
        self.feature_extractor = vgg.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, img):
        return self.feature_extractor(img)
