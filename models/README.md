# **Segmentation Models - README**

该文件夹包含了多种常用的语义分割模型，您可以在您的项目中直接使用它们进行训练和推理。以下是该文件夹中模型的简要介绍及其用法。

### 1. **UNet**

UNet 是一种用于图像语义分割的全卷积神经网络架构。它由两部分组成：

- **Encoder (下采样路径)**: 提取逐渐抽象的特征。
- **Decoder (上采样路径)**: 恢复空间细节。

关键在于使用了跳跃连接 (skip connections)，将编码器和解码器之间的特征图连接起来，从而确保解码器能够访问编码器中的细粒度信息，有助于生成更精确的分割掩码。

#### 使用示例：

```
python


复制编辑
import torch
import segmentation_models_pytorch as smp

# 初始化UNet模型
model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", classes=5)
model.eval()

# 生成随机输入
images = torch.rand(2, 3, 256, 256)

# 进行推理
with torch.inference_mode():
    mask = model(images)

print(mask.shape)
# torch.Size([2, 5, 256, 256])
```

#### 主要参数：

- `encoder_name`: 用作编码器的分类模型（如 `resnet34`，`resnet18` 等）。
- `encoder_weights`: 编码器预训练的权重（如 `imagenet`）。
- `classes`: 输出掩码的类别数。
- `activation`: 最后卷积层后的激活函数。

------

### 2. **UNet++**

UNet++ 是一种改进的 UNet 网络结构，其解码器比传统的 UNet 更复杂。UNet++ 在跳跃连接的基础上增加了更多的连接，增强了细粒度的特征恢复能力。

#### 使用示例：

```
python


复制编辑
import torch
import segmentation_models_pytorch as smp

# 初始化UNet++模型
model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", classes=5)
model.eval()

# 生成随机输入
images = torch.rand(2, 3, 256, 256)

# 进行推理
with torch.inference_mode():
    mask = model(images)

print(mask.shape)
# torch.Size([2, 5, 256, 256])
```

#### 主要参数：

- `encoder_name`: 用作编码器的分类模型。
- `encoder_weights`: 编码器预训练的权重。
- `decoder_channels`: 解码器中卷积层的通道数。
- `activation`: 激活函数。







### 3. **FPN (Feature Pyramid Networks)**:

- **Purpose**: Uses a feature pyramid network for semantic segmentation. It captures features at multiple scales and merges them to make predictions.

- Key Parameters

  :

  - `encoder_name`: Backbone model for feature extraction (e.g., `resnet34`).
  - `encoder_depth`: Controls the depth of the encoder.
  - `decoder_pyramid_channels`: Number of convolution filters in the feature pyramid.
  - `decoder_segmentation_channels`: Filters in the segmentation blocks.
  - `decoder_merge_policy`: Method to merge features (`'add'` or `'cat'`).
  - `in_channels`: Number of input channels (default is 3 for RGB).
  - `classes`: Number of output classes for the segmentation mask.
  - `activation`: Activation function for the final layer.
  - `upsampling`: Upsampling factor (default is 4).

### 4. **PSPNet (Pyramid Scene Parsing Network)**:

- **Purpose**: Incorporates a spatial pyramid pooling layer that captures context at multiple scales, beneficial for large-scale scene parsing.

- Key Parameters

  :

  - `psp_out_channels`: Filters in the spatial pyramid.
  - `psp_use_batchnorm`: Whether to apply batch normalization in the spatial pyramid.
  - `psp_dropout`: Dropout rate in the pyramid.
  - `in_channels`: Number of input channels.
  - `classes`: Number of output classes.
  - `activation`: Activation function to apply after the final convolution.
  - `upsampling`: Upsampling factor for preserving input-output spatial identity.

### 5. **DeepLabV3**:

- **Purpose**: Uses atrous (dilated) convolutions to improve receptive fields, which is beneficial for segmenting objects at various scales.

- Key Parameters

  :

  - `encoder_name`: Backbone used for feature extraction.
  - `encoder_output_stride`: Downsampling factor for last encoder features.
  - `decoder_channels`: Number of convolution filters in the ASPP module.
  - `decoder_atrous_rates`: Dilation rates for ASPP (Atrous Spatial Pyramid Pooling).
  - `decoder_aspp_separable`: Whether to use separable convolutions in ASPP.
  - `decoder_aspp_dropout`: Dropout rate in ASPP module.
  - `in_channels`: Number of input channels.
  - `classes`: Number of output classes for segmentation.
  - `activation`: Activation function for the final output.
  - `upsampling`: Final upsampling factor.

### Use Cases:

- **FPN**: Ideal for tasks that require multi-scale feature extraction, such as object detection or segmentation of scenes with various object sizes.
- **PSPNet**: Best for applications needing context aggregation from different scales, typically for scene parsing tasks where spatial understanding across the entire image is crucial.
- **DeepLabV3**: Designed to handle large-scale objects and small details by using atrous convolutions, making it well-suited for urban scene segmentation or tasks that require high precision in pixel-level segmentation.







### 6. **DeepLabV3+**:

- **Model Overview**: DeepLabV3+ is a state-of-the-art model for semantic image segmentation, combining DeepLabV3 with an improved decoder. It uses dilated convolutions in the ASPP (Atrous Spatial Pyramid Pooling) module and has a powerful encoder-decoder structure.

- Key Parameters

  :

  - `encoder_name`: Defines the backbone classification model (e.g., ResNet).
  - `encoder_depth`: Number of stages in the encoder (3 to 5).
  - `decoder_channels`: Number of convolution filters in the decoder.
  - `decoder_atrous_rates`: Dilated convolution rates for ASPP.
  - `classes`: Number of output classes for segmentation.
  - `activation`: Activation function applied after the final convolution.

### 7. **LinkNet**:

- **Model Overview**: LinkNet is a fully convolutional network that employs an encoder-decoder architecture with skip connections to produce pixel-wise classification maps.

- Key Parameters

  :

  - `decoder_use_batchnorm`: Whether to use batch normalization in the decoder.
  - `aux_params`: Auxiliary output parameters if you want an additional classification head.
  - `encoder_weights`: Pre-trained weights (e.g., ImageNet).

### 8. **MAnet**:

- **Model Overview**: Multi-scale Attention Network (MAnet) uses attention mechanisms to capture rich contextual dependencies, including spatial and channel-wise dependencies. It utilizes Position-wise Attention Blocks (PAB) and Multi-scale Fusion Attention Blocks (MFAB).

- Key Parameters

  :

  - `decoder_channels`: List specifying channels for decoder convolutions.
  - `decoder_pab_channels`: Number of channels for the Position-wise Attention Block.
  - `classes`: Number of output classes.

### 9. **PAN**:

- **Model Overview**: Pyramid Attention Network (PAN) enhances feature representation by using pyramid attention and multi-scale fusion to improve segmentation accuracy.

- Key Parameters

  :

  - `encoder_output_stride`: Downsampling factor for the encoder.
  - `decoder_channels`: Number of filters for convolution in the decoder.
  - `upsampling`: Final upsampling factor to maintain input-output spatial shape identity.

### 10. **UPerNet**:

- **Model Overview**: UPerNet (Unified Perceptual Parsing Network) uses a unified perceptual parsing strategy that combines features from various spatial resolutions in a hierarchical manner to improve segmentation accuracy.

- Key Parameters

  :

  - `decoder_pyramid_channels`: Number of channels for the pyramid module.
  - `decoder_segmentation_channels`: Channels used in the segmentation module.