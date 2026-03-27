# src/siglip/config.py
"""
Configuration for SigLIP-based feature extraction and GBDT training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union


@dataclass
class SiglipConfig:
    """
    Configuration for SigLIP-based GBDT training and inference.

    Attributes:
        data_path (Path): Root directory of competition images.
        split_path (Path): CSV file defining train/val splits and fold indices.
        siglip_model_path (str): Path to pretrained SigLIP model (HuggingFace format).
        output_dir (Path): Directory to save trained models and logs.
        patch_size (int): Size of square patches for image splitting.
        overlap (int): Overlap between consecutive patches.
        embedding_dim (int): Dimension of SigLIP image embeddings (for validation).
        targets (List[str]): All prediction targets required.
        target_names (List[str]): Output order of targets for model predictions.
        target_max (Dict[str, float]): Maximum values for target normalization.
        skip_targets (List[str]): Targets not predicted by SigLIP (handled by DINO).
        pca_n_components (Union[float, int]): PCA variance retention or number of components.
        pls_n_components (int): Number of PLS components (supervised).
        gmm_n_components (int): Number of Gaussian mixture components.
        use_semantic_features (bool): Include semantic features (greenness ratio, clover ratio).
        random_state (int): Random seed for PCA, PLS, GMM.
        seed (int): Global random seed.
        n_folds (int): Number of folds for cross-validation.
        fold_column (str): Column name in split CSV indicating fold index.
        save_models (bool): Whether to save trained models (as joblib/pickle).
        device (str): Device for feature extraction ('cuda' or 'cpu').
        mixed_precision (bool): Use mixed precision for feature extraction.
    """

    # Paths
    data_path: Path = Path("/data/raw")
    split_path: Path = Path("/data/processed/csiro_data_split.csv)
    siglip_model_path: str = "/pretrained/google-siglip-so400m-patch14-384"
    output_dir: Path = Path("outputs/siglip_models")

    # Data
    patch_size: int = 520
    overlap: int = 16
    embedding_dim: int = 1152  # for validation only

    # Targets
    targets: List[str] = field(default_factory=lambda: [
        "Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"
    ])
    target_names: List[str] = field(default_factory=lambda: [
        "Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"
    ])
    target_max: Dict[str, float] = field(default_factory=lambda: {
        "Dry_Clover_g": 71.7865,
        "Dry_Dead_g": 83.8407,
        "Dry_Green_g": 157.9836,
        "Dry_Total_g": 185.70,
        "GDM_g": 157.9836,
    })
    skip_targets: List[str] = field(default_factory=lambda: ["Dry_Clover_g"])

    # Feature engineering
    pca_n_components: Union[float, int] = 0.80
    pls_n_components: int = 8
    gmm_n_components: int = 6
    use_semantic_features: bool = True
    random_state: int = 42

    # Training
    seed: int = 42
    n_folds: int = 4
    fold_column: str = "fold"

    # Output
    save_models: bool = True

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True

    def __post_init__(self):
        """Convert path strings to Path objects if necessary."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        if isinstance(self.split_path, str):
            self.split_path = Path(self.split_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]):
        """
        Load configuration from a YAML file.

        Args:
            yaml_path (Union[str, Path]): Path to YAML configuration file.

        Returns:
            SiglipConfig: Configuration instance.
        """
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Flatten the YAML structure if needed (e.g., paths.data_path)
        # This expects the YAML to have top-level keys matching dataclass fields.
        return cls(**config_dict)