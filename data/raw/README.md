# Dataset: CSIRO Pasture Biomass Prediction

This dataset is from the Kaggle competition:

https://www.kaggle.com/competitions/csiro-biomass

It contains pasture images and structured metadata for predicting biomass components critical for grazing and feed management.

---

## Objective

Predict five pasture biomass components from top-view field images:

- Dry_Green_g (dry green vegetation excluding clover)
- Dry_Dead_g (dry dead material)
- Dry_Clover_g (dry clover biomass)
- GDM_g (green dry matter)
- Dry_Total_g (total dry biomass)

Each prediction corresponds to a (image, target_name) pair.

---

## Data Structure

### train.csv

Contains labeled training samples:

- `sample_id`: Unique sample identifier
- `image_path`: Path to training image
- `Sampling_Date`: Collection date
- `State`: Australian state of sampling
- `Species`: Pasture species composition
- `Pre_GSHH_NDVI`: Vegetation index (GreenSeeker)
- `Height_Ave_cm`: Average pasture height (cm)
- `target_name`: Biomass component type
- `target`: Ground-truth biomass value (grams)

---

### train/

Folder containing training images referenced by `train.csv`.

---

### test.csv

Contains test-time prediction queries:

- `sample_id`: Unique identifier for prediction row
- `image_path`: Path to test image
- `target_name`: Biomass component to predict  
  (Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g)

---

### test/

Folder containing test images (available at inference time).

---

### sample_submission.csv

Submission format:

- `sample_id`: Matches test.csv
- `target`: Predicted biomass value (grams)

Each row corresponds to one (image, target_name) pair.

---

## Evaluation

Predictions are evaluated using a **weighted R² score**, where different biomass components have different importance weights.  
In particular, `Dry_Total_g` contributes the highest weight (50%).

---

## Citation

If you use this dataset, please cite:

@misc{liao2025estimatingpasturebiomasstopview,
  title={Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture},
  author={Qiyu Liao and Dadong Wang and Rebecca Haling and Jiajun Liu and Xun Li and Martyna Plomecka and Andrew Robson and Matthew Pringle and Rhys Pirie and Megan Walker and Joshua Whelan},
  year={2025},
  eprint={2510.22916},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.22916}
}