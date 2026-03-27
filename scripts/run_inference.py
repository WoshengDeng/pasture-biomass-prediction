#!/usr/bin/env python
"""
Inference script for CSIRO Biomass Prediction.

This script loads the configuration and runs the complete inference pipeline
(DINO + SigLIP + ensemble) to generate a submission file.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description="Run biomass prediction inference")
    parser.add_argument('--config', type=str, default='configs/inference.yaml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Optional: override device from command line if needed
    # parser.add_argument('--device', type=str, default=None)
    # if args.device:
    #     config['device'] = args.device

    # Run inference
    run_inference(config)


if __name__ == "__main__":
    main()