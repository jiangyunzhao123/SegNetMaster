from .pet_model import PetModel
from .unet_model import UNetModel
from models.deeplabv3_plus import DeepLabV3Plus
from models.linknet import LinkNet
from models.manet import MAnet
from models.pan import PAN
from models.upernet import UPerNet

def get_model(arch='FPN', encoder_name='resnet34', in_channels=3, out_classes=1):
    """
    This function returns a model based on the architecture name provided.

    Parameters:
    - arch (str): The architecture of the model to use. Defaults to 'FPN'.
    - encoder_name (str): The encoder architecture name to use, default is 'resnet34'.
    - in_channels (int): The number of input channels, default is 3 (for RGB images).
    - out_classes (int): The number of output classes, default is 1.

    Returns:
    - model: The initialized model.
    """
    
    # Check the architecture type and return the corresponding model
    if arch == 'FPN':
        return PetModel(arch=arch, encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    elif arch == 'UNet':
        return UNetModel(arch=arch, encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    elif arch == 'deeplabv3_plus':
        return DeepLabV3Plus(encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    elif arch == 'linknet':
        return LinkNet(encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    elif arch == 'manet':
        return MAnet(encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    elif arch == 'pan':
        return PAN(encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    elif arch == 'upernet':
        return UPerNet(encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    else:
        raise ValueError(f"Model architecture {arch} is not supported.")
