# src/data/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from src.dino.config import CFG  # Import the configuration class from the project configuration module


def load_train_data() -> pd.DataFrame:
    """
    Load and preprocess training data.

    Pipeline:
    - Read long-format annotations (one row per image-target pair)
    - Convert to wide format (one row per image)
    - Ensure all target columns exist
    - Create stratification bins based on total biomass
    - Assign StratifiedGroupKFold splits to prevent data leakage

    Returns:
        pd.DataFrame: Training dataframe in wide format with fold assignments.
    """
    df = pd.read_csv(CFG.TRAIN_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Convert long → wide format (multi-target regression setup)
    df_wide = df.pivot_table(
        index=['image_id', 'image_path'],
        columns='target_name',
        values='target',
        aggfunc='first'
    ).reset_index()

    # Ensure consistent target schema (robust to missing labels)
    for col in CFG.TARGET_COLS:
        if col not in df_wide.columns:
            df_wide[col] = 0.0

    # Stratification based on total biomass distribution
    df_wide['total_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=5, labels=False, duplicates='drop')

    # Stratified + grouped CV to balance targets and avoid leakage
    sgkf = StratifiedGroupKFold(
        n_splits=CFG.N_FOLDS,
        shuffle=True,
        random_state=CFG.SEED
    )
    df_wide['fold'] = -1

    for fold, (_, val_idx) in enumerate(
            sgkf.split(df_wide, df_wide['total_bin'], groups=df_wide['image_id'])
    ):
        df_wide.loc[val_idx, 'fold'] = fold

    print(f"✓ Loaded {len(df_wide)} training images")
    print(f"Fold distribution:\n{df_wide['fold'].value_counts().sort_index()}")

    return df_wide


def load_test_data() -> pd.DataFrame:
    """
    Load and preprocess test data.

    - Reads long-format test annotations
    - Extracts unique image entries for inference
    - Keeps only image_id and image_path

    Returns:
        pd.DataFrame: Unique test images (one row per image).
    """
    df = pd.read_csv(CFG.TEST_CSV)
    df['image_id'] = df['sample_id'].str.split('__').str[0]

    # Deduplicate to one entry per image (test data is provided in long format)
    df_unique = df.drop_duplicates('image_id')[['image_id', 'image_path']].reset_index(drop=True)

    print(f"✓ Loaded {len(df_unique)} test images")
    return df_unique