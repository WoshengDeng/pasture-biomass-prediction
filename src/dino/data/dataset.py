# src/data/dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dino.config import CFG  # Import the configuration class from the project configuration module


class BiomassDataset(Dataset):
    """
    Custom dataset for biomass regression.

    Each sample:
    - Loads a full-width image and splits it into left/right halves
    - Applies transforms independently to each half
    - Returns paired inputs with multi-target labels

    Args:
        df (pd.DataFrame): Input dataframe (one row per image)
        img_dir (str): Root directory for images
        transform (callable, optional): Albumentations transform
    """

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.paths = df['image_path'].values
        self.labels = df[CFG.TARGET_COLS].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        img_name = os.path.basename(self.paths[idx])
        path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(path)

        # Fallback to a zero image to avoid training interruption on read failure
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split image into left/right halves (task-specific input design)
        h, w, _ = img.shape
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]

        if self.transform:
            left = self.transform(image=left)['image']
            right = self.transform(image=right)['image']

        label = torch.from_numpy(self.labels[idx])
        return left, right, label