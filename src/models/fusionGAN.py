import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class FusionGANDiscriminator(nn.Module):
    def __init__(self):
        super(FusionGANDiscriminator, self).__init__()
        # input_size 480*640
        # Layer 1
        self.conv1 = spectral_norm(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layer 2
        self.conv2 = spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3
        self.conv3 = spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(128)
        
        # Layer 4
        self.conv4 = spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))
        self.bn4 = nn.BatchNorm2d(256)

        # Layer 5
        self.conv5 = spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1))
        self.bn5 = nn.BatchNorm2d(512)

        self.conv_layers = nn.Sequential(
            self.conv1, self.bn1, nn.LeakyReLU(),
            self.conv2, self.bn2, nn.LeakyReLU(),
            self.conv3, self.bn3, nn.LeakyReLU(),
            self.conv4, self.bn4, nn.LeakyReLU(),
            self.conv5, self.bn5, nn.LeakyReLU(),
        )

        self.MP = nn.AdaptiveAvgPool2d((1,1))
        
        # Linear Layer
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feature_map = self.conv_layers(x)
        compressed = self.MP(feature_map)
        compressed = torch.flatten(compressed, start_dim=1, end_dim=-1)
        return self.fc(compressed)

class FusionGANGenerator(nn.Module):
    def __init__(self):
        super(FusionGANGenerator, self).__init__()
        # Layer 1: 5x5 conv, 256 output channels
        self.conv1 = spectral_norm(nn.Conv2d(2, 256, kernel_size=5, stride=1, padding=2))
        self.bn1 = nn.BatchNorm2d(256)
        
        # Layer 2: 5x5 conv, 128 output channels
        self.conv2 = spectral_norm(nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2))
        self.bn2 = nn.BatchNorm2d(128)
        
        # Layer 3: 3x3 conv, 64 output channels
        self.conv3 = spectral_norm(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1))
        self.bn3 = nn.BatchNorm2d(64)
        
        # Layer 4: 3x3 conv, 32 output channels
        self.conv4 = spectral_norm(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        self.bn4 = nn.BatchNorm2d(32)
        
        # Layer 5: 1x1 conv, 1 output channel (final output)
        self.conv5 = spectral_norm(nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0))

        self.fusion_layers = nn.Sequential(
            self.conv1, self.bn1, nn.LeakyReLU(),
            self.conv2, self.bn2, nn.LeakyReLU(),
            self.conv3, self.bn3, nn.LeakyReLU(),
            self.conv4, self.bn4, nn.LeakyReLU(),
            self.conv5, nn.Sigmoid()
        )

    def forward(self, vis, ir):
        return self.fusion_layers(torch.concat([vis, ir], dim=1))


if __name__ == "__main__":
    input_tensor_1 = torch.randn(8, 1, 480, 640)
    model = FusionGANDiscriminator()
    feature = input_tensor_1
    for conv in model.conv_layers:
        feature = conv(feature)
        print(feature.size())
    compressed = model.MP(feature)
    print(compressed.size())
    feature = model.fc(torch.flatten(compressed, start_dim=1, end_dim=-1))
    print(feature.size())
    print('*********************************************************')
    model = FusionGANGenerator()
    feature = torch.randn(8, 2, 120, 120)
    for conv in model.fusion_layers:
        feature = conv(feature)
        print(feature.size())
