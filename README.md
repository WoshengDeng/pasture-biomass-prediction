## 1. Project Overview

A top 2% Kaggle solution (82/3802, Silver Medal) for intracranial aneurysm detection. An end-to-end machine learning system that predicts pasture biomass from field images to support real-world grazing decisions, achieving top 2% ranking in the CSIRO Kaggle competition.

Kaggle write-up (brief overview of approach):
https://kaggle.com/competitions/csiro-biomass/writeups/csiro-image2biomass-prediction-my-approach-to

## 2. Highlights

- Achieved top 2% ranking (82/3802) in the CSIRO Kaggle competition with a high-performance multi-target regression system for pasture biomass prediction.

- Built a constraint-aware multi-task learning framework by predicting fundamental components (Green / Dead / Clover) and deriving GDM and Total via deterministic relationships, improving consistency on high-weight targets.

- Leveraged DINOv3 ViT-Huge (1B+ parameters) with transfer learning, combined with a dual-view (left-right split) input strategy and lightweight feature fusion to better capture spatial heterogeneity in wide-field images.

- Designed a hybrid inference pipeline with 4-fold model ensembling and a complementary SigLIP + semantic feature + GBDT branch to improve robustness under distribution shifts and extreme cases.

- Implemented a full training pipeline with stratified group k-fold validation, mixed precision training, differential learning rates, and early stopping for stable and reproducible performance.

- Applied domain-informed post-processing and target re-composition to enforce physical consistency and reduce prediction variance on critical outputs.

## 3. Problem and Motivation

Accurate estimation of pasture biomass is critical for grazing management, as it directly impacts livestock feeding decisions, pasture recovery, and long-term agricultural sustainability. However, traditional methods such as manual clipping and weighing are labor-intensive and not scalable, while sensor-based or remote sensing approaches often lack reliability or fine-grained component-level information.

This project addresses the problem of predicting multiple pasture biomass components directly from field images, using supervised learning with ground-truth measurements. The task involves estimating five interrelated targets (Green, Dead, Clover, GDM, Total), where strong physical relationships exist between outputs, making consistency and structured modeling essential for reliable predictions.

The goal is to build a robust and scalable vision-based system that can generalize across diverse environmental conditions and support real-world decision-making in precision agriculture.

## 4. System Architecture

The overall system follows a modular pipeline that covers data processing, model training, and inference with ensembling and post-processing:

```
Raw Data (Images + Metadata)
        ↓
Data Preprocessing (pivot to multi-target format, stratified group k-fold)
        ↓
Feature Extraction (DINOv3 ViT-Huge backbone)
        ↓
Dual-View Input (left/right image split)
        ↓
Feature Fusion (lightweight sequence modeling blocks)
        ↓
Multi-Task Prediction (Green / Dead / Clover)
        ↓
Deterministic Composition (GDM = Green + Clover, Total = GDM + Dead)
        ↓
Cross-Validation Training (4-fold ensemble)
        ↓
Inference Pipeline
   ├── DINO Ensemble Branch
   ├── SigLIP + Semantic Features + GBDT Branch
   └── Weighted Fusion
        ↓
Post-processing (domain constraints, clipping, rebalancing)
        ↓
Final Predictions
```

The architecture is designed with the following principles:

- **Modularity**: Clear separation between data processing, modeling, and inference components.
- **Consistency-aware modeling**: Enforcing structural relationships between targets to reduce prediction conflicts.
- **Robustness**: Combining deep learning and tree-based models to improve generalization under distribution shifts.
- **Scalability**: Using cross-validation, ensembling, and efficient inference strategies to ensure stable performance.

## 5. Technical Stack

- **Deep Learning**: PyTorch, timm (DINOv3 ViT-Huge)
- **Classical ML**: LightGBM, CatBoost, Scikit-learn (GBDT, PLS, PCA, GMM)
- **Computer Vision**: OpenCV, PIL, Albumentations
- **Data Processing**: Pandas, NumPy
- **Training & Optimization**: Mixed Precision (AMP), AdamW, Cosine LR Scheduler, Early Stopping
- **Validation Strategy**: Stratified Group K-Fold Cross Validation
- **Modeling Techniques**: Multi-task Learning, Constraint-aware Modeling, Model Ensembling
- **Feature Engineering**: Semantic Feature Extraction (SigLIP), Embedding-based Features

## 6. Results

### Leaderboard Performance

- **Final Ranking**: 82 / 3802 (Top 2%)
- **Evaluation Metric**: Weighted R² ≈ 0.62

This result places the solution among the top-performing entries in a large-scale competition with diverse real-world agricultural data.

---

### Evaluation Metric

The competition uses a **globally weighted R² score** computed over all (image, target) pairs. Different biomass components are assigned different importance weights, with **Dry_Total_g contributing 50% of the final score**.

This makes the task particularly challenging, as models must not only achieve high accuracy on individual components but also maintain **consistency across interdependent targets**, especially for high-weight aggregate outputs.

---

### Key Insights

- **Structured prediction improves stability**: Modeling fundamental components (Green, Dead, Clover) and deriving aggregate targets (GDM, Total) helps reduce inconsistency and improves performance on heavily weighted targets.

- **Ensembling enhances robustness**: Combining multiple folds and heterogeneous models (DINO + GBDT) reduces variance and improves generalization under diverse environmental conditions.

- **Semantic features provide complementary signals**: Incorporating concept-based features (e.g., greenness, density) improves performance in edge cases where pure visual regression is less reliable.

---

### Task Challenges

- **Multi-target dependency**: Outputs are physically correlated, requiring consistency-aware modeling rather than independent prediction.

- **High variability in real-world data**: Images span different regions, seasons, and species compositions, introducing significant distribution shifts.

- **Imbalanced importance across targets**: The heavy weighting on total biomass increases sensitivity to small inconsistencies in component predictions.

## 7. How to Run

Code and environment setup details are omitted as the project is intended for demonstration purposes.


## 8. Project Structure

The repository is organized to separate data, model branches, training scripts, and inference logic, supporting reproducibility and clarity.

```text
pasture-biomass-prediction/
├── configs/                     # Configuration files (dino_train.yaml, siglip_train.yaml, inference.yaml)
├── data/                        # Raw and processed data
│   ├── raw/                     # Original competition data (images, train.csv, test.csv)
│   └── processed/               # Derived data (e.g., csiro_data_split.csv with embeddings and folds)
├── src/                         # Core source code
│   ├── dino/                    # DINOv3 training and model
│   ├── siglip/                  # SigLIP feature extraction and GBDT training
│   └── ensemble/                # Ensemble logic for final predictions
├── scripts/                     # Entry points: run_dino_training.py, run_siglip_train.py, run_inference.py
├── inference/                   # Inference modules (dino_inference.py, siglip_inference.py)
├── outputs/                     # Saved models, logs, and submission files (gitignored)
├── pretrained/                  # Local pretrained models (e.g., SigLIP)
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git ignore rules
```