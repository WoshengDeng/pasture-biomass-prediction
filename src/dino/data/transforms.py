# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Import the configuration class from the project configuration module
from src.dino.config import CFG


def get_train_transforms() -> Compose:
    """
    Build training data augmentation pipeline.

    Includes:
    - Geometric augmentations (flip, rotation, scale/shift)
    - Color augmentation for robustness to lighting variations
    - Standard ImageNet normalization

    Returns:
        albumentations.Compose: Training transform pipeline.
    """
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),  # Standardize input resolution
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Match pretrained backbone distribution
        ToTensorV2()
    ])


def get_val_transforms() -> Compose:
    """
    Build validation preprocessing pipeline.

    - No augmentation (deterministic evaluation)
    - Resize to model input size
    - Apply ImageNet normalization

    Returns:
        albumentations.Compose: Validation transform pipeline.
    """
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])