import os
import torch
import onnx
import onnxruntime
import numpy as np
import segmentation_models_pytorch as smp

# 确保保存目录存在
output_dir = './saved_models'
os.makedirs(output_dir, exist_ok=True)

# 创建随机模型（可以替换为加载你的训练模型）
model = smp.Unet("resnet34", encoder_weights="imagenet", classes=1)
model.eval()  # 将模型设置为推理模式

# 定义动态轴，以便在推理时支持不同的输入大小
dynamic_axes = {0: "batch_size", 2: "height", 3: "width"}

# 设置ONNX模型文件名
onnx_model_name = os.path.join(output_dir, "unet_resnet34.onnx")

# 导出模型为ONNX格式
onnx_model = torch.onnx.export(
    model,  # 被导出的模型
    torch.randn(1, 3, 224, 224),  # 模型输入样本
    onnx_model_name,  # 保存路径
    export_params=True,  # 存储模型的训练参数
    opset_version=17,  # ONNX的版本
    do_constant_folding=True,  # 是否执行常量折叠优化
    input_names=["input"],  # 输入名称
    output_names=["output"],  # 输出名称
    dynamic_axes={  # 动态长度的轴
        "input": dynamic_axes,
        "output": dynamic_axes,
    },
)

# 检查ONNX模型
onnx_model = onnx.load(onnx_model_name)
onnx.checker.check_model(onnx_model)

# 使用onnxruntime进行推理
# 创建一个不同batch大小和尺寸的样本进行测试
sample = torch.randn(2, 3, 512, 512)

# 加载ONNX模型并进行推理
ort_session = onnxruntime.InferenceSession(
    onnx_model_name, providers=["CPUExecutionProvider"]
)

# 进行ONNX推理并输出结果
ort_inputs = {"input": sample.numpy()}
ort_outputs = ort_session.run(output_names=None, input_feed=ort_inputs)

# 与PyTorch模型的输出进行比较
with torch.no_grad():
    torch_out = model(sample)

# 比较ONNX Runtime和PyTorch的结果
np.testing.assert_allclose(torch_out.numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)

print("导出的模型已经通过ONNXRuntime进行测试，结果一致！")
