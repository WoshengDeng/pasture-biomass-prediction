# src/siglip/features/embedding_extractor.py
"""
Feature extraction utilities for SigLIP model.

Provides functions to compute image embeddings and semantic features
using a pretrained SigLIP model.
"""

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

from src.siglip.data.transforms import split_image


def compute_siglip_embeddings(
    model_path: Union[str, Path],
    df: pd.DataFrame,
    img_dir: Union[str, Path],
    device: str = "cuda",
    patch_size: int = 520,
    overlap: int = 16,
    embedding_dim: int = 1152,
) -> np.ndarray:
    """
    Compute image embeddings using a pretrained SigLIP model.

    Args:
        model_path (Union[str, Path]): Path to the pretrained SigLIP model.
        df (pd.DataFrame): DataFrame containing image paths (must have 'image_path' column).
        img_dir (Union[str, Path]): Root directory of images.
        device (str): Device to run inference on ("cpu" or "cuda").
        patch_size (int): Size of square patches for image splitting.
        overlap (int): Overlap between consecutive patches.
        embedding_dim (int): Expected embedding dimension (for fallback vectors).

    Returns:
        np.ndarray: Array of image embeddings with shape [N, embedding_dim].
    """
    print(f"Computing SigLIP embeddings for {len(df)} images...")

    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(model_path)

    embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img_path = Path(img_dir) / row["image_path"]
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Split image into overlapping patches
            patches = split_image(img, patch_size=patch_size, overlap=overlap)
            images = [Image.fromarray(p) for p in patches]

            # Extract patch features
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)

            # Average patch features to get image embedding
            embeddings.append(features.mean(dim=0).cpu().numpy())

        except Exception as e:
            print(f"Error processing {row['image_path']}: {e}")
            embeddings.append(np.zeros(embedding_dim))  # Fallback zero vector

    del model
    torch.cuda.empty_cache()

    return np.stack(embeddings)


def generate_semantic_features(
    embeddings: np.ndarray,
    model_path: Union[str, Path],
    device: str = "cuda",
) -> np.ndarray:
    """
    Generate semantic features by computing similarity between image embeddings
    and predefined textual concepts using a SigLIP model.

    Args:
        embeddings (np.ndarray): Image embeddings from SigLIP, shape [N, D].
        model_path (Union[str, Path]): Path to pretrained SigLIP model.
        device (str): Device to run inference on ("cpu" or "cuda").

    Returns:
        np.ndarray: Semantic features array of shape [N, 10] (8 concept scores + 2 ratios).
    """
    model = AutoModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Predefined textual concepts relevant to biomass prediction
    concepts = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass"],
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"]
    }

    # Compute text embeddings for each concept
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concepts.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # L2 normalize
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)  # Average across prompts

    # Normalize image embeddings
    img_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)

    # Compute similarity scores (dot product) with each concept
    scores = {
        name: (img_tensor @ vec.T).cpu().numpy().flatten()
        for name, vec in concept_vectors.items()
    }

    # Create DataFrame and compute derived ratios
    df_scores = pd.DataFrame(scores)
    df_scores["ratio_greenness"] = df_scores["green"] / (df_scores["green"] + df_scores["dead"] + 1e-6)
    df_scores["ratio_clover"] = df_scores["clover"] / (df_scores["clover"] + df_scores["grass"] + 1e-6)

    del model
    torch.cuda.empty_cache()

    return df_scores.values