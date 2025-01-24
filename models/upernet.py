import torch
import torch.nn as nn

class UPerNet(nn.Module):
    def __init__(self, decoder_pyramid_channels=256, decoder_segmentation_channels=128):
        super(UPerNet, self).__init__()
        self.decoder_pyramid_channels = decoder_pyramid_channels
        self.decoder_segmentation_channels = decoder_segmentation_channels
        self.decoder = nn.Conv2d(512, self.decoder_pyramid_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.decoder(x)
        x = nn.Conv2d(self.decoder_pyramid_channels, self.decoder_segmentation_channels, kernel_size=1)(x)
        return x
