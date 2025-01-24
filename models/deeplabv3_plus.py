import torch
import torch.nn as nn
from torchvision import models

class DeepLabV3Plus(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet'):
        super(DeepLabV3Plus, self).__init__()
        # 使用预训练的ResNet34作为编码器
        self.encoder = models.resnet34(pretrained=True)
        # 可以加入ASPP模块、解码器等
        self.decoder_channels = 256  # 假设解码器的卷积通道数
        self.decoder = nn.Conv2d(512, self.decoder_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
