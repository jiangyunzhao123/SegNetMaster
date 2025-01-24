import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(images, masks, pred_masks, save_dir):
    for i in range(len(images)):
        plt.figure(figsize=(10, 5))

        # 显示图像
        plt.subplot(1, 3, 1)
        plt.imshow(images[i].transpose(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        # 显示真实标签
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i].squeeze(0))
        plt.title("Ground truth")
        plt.axis("off")

        # 显示预测结果
        plt.subplot(1, 3, 3)
        plt.imshow(pred_masks[i].squeeze(0))
        plt.title("Prediction")
        plt.axis("off")

        # 保存图像
        save_path = f"{save_dir}/sample_{i}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
