# inference/inference.py
"""
Main inference script orchestrating DINO and SigLIP predictions,
ensemble, and submission generation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from inference.dino_inference import run_dino_inference
from inference.siglip_inference import run_siglip_inference
from src.ensemble.ensemble import weighted_ensemble, enforce_physical_constraints, create_submission


def run_inference(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute the complete inference pipeline.

    Args:
        config: Configuration dictionary loaded from YAML.
            Must contain all necessary sections: paths, dino, siglip, ensemble.

    Returns:
        Submission DataFrame with columns ['sample_id', 'target'].
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("CSIRO BIOMASS INFERENCE - VIT_HUGE_PLUS + SIGLIP")
    logger.info("=" * 60)

    # 1. Load test data
    data_path = Path(config['paths']['data_path'])
    test_df_raw = pd.read_csv(data_path / 'test.csv')
    test_wide = test_df_raw[['image_path']].drop_duplicates().reset_index(drop=True)
    logger.info(f"[1/6] Test images: {len(test_wide)}")

    # 2. Run DINO inference
    logger.info("[2/6] Running DINO HUGE inference...")
    dino_df = run_dino_inference(config, test_wide, device=config['device'])
    logger.info(f"DINO predictions shape: {dino_df.shape}")

    # 3. Run SigLIP inference (including training on the fly)
    logger.info("[3/6] Running SigLIP inference...")
    siglip_df = run_siglip_inference(config, test_wide)
    logger.info(f"SigLIP predictions shape: {siglip_df.shape}")

    # 4. Ensemble predictions
    logger.info("[4/6] Creating ensemble...")
    weights = config['ensemble']['weights']
    all_targets = config['ensemble']['all_targets']
    clover_only_dino = config['ensemble'].get('clover_only_dino', True)

    final_df = weighted_ensemble(
        dino_df, siglip_df, weights, all_targets, clover_only_dino
    )
    logger.info(f"Ensemble weights: DINO={weights['dino']}, SigLIP={weights['siglip']}")

    # 5. Enforce physical constraints
    if config['ensemble'].get('enforce_physical_constraints', True):
        final_df = enforce_physical_constraints(final_df, all_targets, recompute_derived=True)
        logger.info("Physical constraints enforced (non‑negative, mass balance).")

    # 6. Create submission
    logger.info("[5/6] Creating submission...")
    submission_targets = config['paths'].get('submission_targets', config['dino']['targets'])
    submission = create_submission(
        final_df,
        targets=submission_targets,
        image_path_col='image_path',
        sample_id_format="{image_id}__{target}"
    )

    # 7. Save submission
    output_dir = Path(config['paths'].get('output_dir', '.'))
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / config['paths'].get('submission_filename', 'submission.csv')
    submission.to_csv(submission_path, index=False)
    logger.info(f"[6/6] Submission saved: {submission_path}")

    # Optional: display statistics
    logger.info("\nSubmission stats:\n" + submission['target'].describe().to_string())

    return submission


if __name__ == "__main__":
    # Example usage when run as script:
    # python -m inference.inference --config configs/inference.yaml
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/inference.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_inference(config)