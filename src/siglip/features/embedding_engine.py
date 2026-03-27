# src/siglip/features/embedding_engine.py
"""
Feature transformation engine for SigLIP embeddings.

This module provides the SupervisedEmbeddingEngine class which performs
structured dimensionality reduction and feature enhancement on high-dimensional
embeddings for downstream GBDT training.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture


class SupervisedEmbeddingEngine:
    """
    Performs structured dimensionality reduction and feature enhancement on
    high-dimensional embeddings for downstream GBDT training.

    The transformation pipeline includes:
        - Standard scaling
        - PCA for unsupervised dimensionality reduction
        - PLS regression for supervised signal (optional)
        - Gaussian Mixture Model for clustering structure
        - Optional inclusion of semantic features

    Attributes:
        scaler (StandardScaler): Standardizes features.
        pca (PCA): PCA for unsupervised dimensionality reduction.
        pls (PLSRegression): PLS regression for supervised signal.
        gmm (GaussianMixture): GMM for clustering structure.
        pls_fitted_ (bool): Flag to track if PLS was fitted.
    """

    def __init__(
        self,
        pca_n_components: float = 0.80,
        pls_n_components: int = 8,
        gmm_n_components: int = 6,
        random_state: int = 42,
    ):
        """
        Initialize the embedding engine.

        Args:
            pca_n_components: Variance retention (float) or number of components (int) for PCA.
            pls_n_components: Number of PLS components.
            gmm_n_components: Number of Gaussian mixture components.
            random_state: Random seed for reproducibility.
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_n_components, random_state=random_state)
        self.pls = PLSRegression(n_components=pls_n_components, scale=False)
        self.gmm = GaussianMixture(
            n_components=gmm_n_components, covariance_type="diag", random_state=random_state
        )
        self.pls_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray = None, X_semantic: np.ndarray = None):
        """
        Fit the pipeline on training data.

        Args:
            X (np.ndarray): Feature embeddings [n_samples, n_features].
            y (np.ndarray, optional): Target labels for supervised PLS.
            X_semantic (np.ndarray, optional): Semantic features (not used in fit).

        Returns:
            self: Fitted instance.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        if y is not None:
            self.pls.fit(X_scaled, y)
            self.pls_fitted_ = True
        return self

    def transform(self, X: np.ndarray, X_semantic: np.ndarray = None) -> np.ndarray:
        """
        Transform new data using the fitted pipeline.

        Args:
            X (np.ndarray): Feature embeddings [n_samples, n_features].
            X_semantic (np.ndarray, optional): Semantic features to include.

        Returns:
            np.ndarray: Transformed features [n_samples, n_transformed_features].
        """
        X_scaled = self.scaler.transform(X)
        features = [self.pca.transform(X_scaled)]

        if self.pls_fitted_:
            features.append(self.pls.transform(X_scaled))

        features.append(self.gmm.predict_proba(X_scaled))  # GMM probability features

        if X_semantic is not None:
            # Normalize semantic features and include
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            features.append(sem_norm)

        return np.hstack(features)