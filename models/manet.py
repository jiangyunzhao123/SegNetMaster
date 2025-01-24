import torch
import torch.nn as nn

class MAnet(nn.Module):
    def __init__(self, decoder_channels=256, pab_channels=128):
        super(MAnet, self).__init__()
        self.decoder_channels = decoder_channels
        self.pab_channels = pab_channels
        # 定义PAB和MFAB模块
        self.decoder = nn.Conv2d(512, self.decoder_channels, kernel_size=3, padding=1)
        # 模拟注意力模块（具体实现可以根据需要来定制）
        self.attention = nn.Conv2d(self.decoder_channels, self.pab_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.decoder(x)
        x = self.attention(x)
        return x
