import torch
import os
from models.model_utils import get_model
from src.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.trainer import train_model

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置数据路径
image_dir = "data/voc/JPEGImages"
annotation_dir = "data/voc/SegmentationClass"
train_image_files = os.listdir(image_dir)
train_annotation_files = os.listdir(annotation_dir)

# 数据集和Dataloader
train_dataset = CustomDataset(image_dir, annotation_dir, train_image_files, train_annotation_files)
train_dataloader = DataLoader(train_dataset, batch_size
