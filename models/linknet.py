import torch
import torch.nn as nn
from torchvision import models

class LinkNet(nn.Module):
    def __init__(self, encoder_name='resnet34'):
        super(LinkNet, self).__init__()
        self.encoder = models.resnet34(pretrained=True)
        self.decoder = nn.Conv2d(512, 256, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
