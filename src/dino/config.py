# config.py
import os
import torch

class CFG:
    """Configuration class for all hyperparameters and paths."""

    BASE_PATH = ''  # Root data directory 
    TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    TEST_CSV = os.path.join(BASE_PATH, 'test.csv')
    TEST_IMAGE_DIR = os.path.join(BASE_PATH, 'test')

    MODEL_DIR = ''  # Directory to save model weights
    OUTPUT_DIR = ''  # Directory for logs and summaries

    MODEL_NAME = 'vit_huge_plus_patch16_dinov3.lvd1689m'  # DINOv3 ViT-Huge

    SEED = 42
    N_FOLDS = 4
    FOLDS_TO_TRAIN = [0, 1, 2, 3]  # All folds by default

    IMG_SIZE = 512  # Input resolution per half-image
    BATCH_SIZE = 6  # Small due to ViT-Huge memory constraints
    NUM_WORKERS = 0  # Kaggle/Windows often use 0

    EPOCHS = 210  # Max epochs (early stopping will cut short)
    WARMUP_EPOCHS = 2  # Warmup in steps
    LR_BACKBONE = 1e-5  # Lower LR for backbone (stable fine-tuning)
    LR_HEAD = 5e-4  # Higher LR for head (quick adaptation)
    WD = 1e-2  # Weight decay

    CLIP_GRAD_NORM = 1.0  # Gradient clipping threshold
    DROPOUT = 0.2  # Dropout probability in the head

    EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for this many epochs

    TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Ensure necessary directories exist (consistent with original code)
os.makedirs(CFG.MODEL_DIR, exist_ok=True)