import segmentation_models_pytorch as smp
import torch.nn as nn

class UNetModel(nn.Module):
    def __init__(self, arch='UNet', encoder_name='resnet34', in_channels=3, out_classes=1, **kwargs):
        super(UNetModel, self).__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, logits, mask):
        return self.loss_fn(logits, mask)
