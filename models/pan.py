import torch
import torch.nn as nn

class PAN(nn.Module):
    def __init__(self, encoder_output_stride=16, upsampling=4):
        super(PAN, self).__init__()
        self.encoder_output_stride = encoder_output_stride
        self.upsampling = upsampling
        self.decoder = nn.Conv2d(512, 256, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.decoder(x)
        x = nn.functional.interpolate(x, scale_factor=self.upsampling, mode='bilinear', align_corners=False)
        return x
