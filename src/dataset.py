import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, image_list, annotation_list, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = image_list
        self.annotations = annotation_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.annotation_dir, self.annotations[idx])

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        sample = dict(image=image, mask=mask)
        if self.transform:
            sample = self.transform(**sample)

        # resize
        image = np.array(Image.fromarray(sample["image"]).resize((1024, 1024)))
        mask = np.array(Image.fromarray(sample["mask"]).resize((1024, 1024)))

        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)

        return sample
