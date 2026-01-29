"""
Data loading and preprocessing for Breast Cancer Wisconsin (Diagnostic) dataset.

This module provides functions to load and preprocess the UCI Breast Cancer dataset
for quantum machine learning experiments.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional


def load_breast_cancer_data(
    test_size: float = 0.2,
    random_state: int = 42,
    n_components: int = 4,
    preserve_variance: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA, StandardScaler]:
    """
    Load and preprocess Breast Cancer Wisconsin (Diagnostic) dataset.
    
    Args:
        test_size: Proportion of dataset to include in test split (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        n_components: Number of principal components for PCA (default: 4)
        preserve_variance: If specified, use enough components to preserve this variance
        
    Returns:
        Tuple containing:
            - X_train: Training features (n_samples_train, n_features)
            - X_test: Test features (n_samples_test, n_features)
            - y_train: Training labels (n_samples_train,)
            - y_test: Test labels (n_samples_test,)
            - pca: Fitted PCA transformer
            - scaler: Fitted StandardScaler transformer
            
    Example:
        >>> X_train, X_test, y_train, y_test, pca, scaler = load_breast_cancer_data()
        >>> print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        Training samples: 455, Features: 4
    """
    print("Loading Breast Cancer Wisconsin (Diagnostic) dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    class_names = data.target_names
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {class_names[0]} ({np.sum(y==0)}), {class_names[1]} ({np.sum(y==1)})")
    
    # Apply PCA
    if preserve_variance is not None:
        pca = PCA()
        pca.fit(X)
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= preserve_variance) + 1
        print(f"Using {n_components} components to preserve {preserve_variance:.1%} variance")
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)
    
    X_reduced = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"PCA: {X.shape[1]} features -> {X_reduced.shape[1]} components "
          f"({explained_variance:.1%} variance preserved)")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    # Map to rotation angles for quantum encoding: [-π, π]
    X_final = np.arctan(X_scaled) * 2
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, pca, scaler


def preprocess_data(
    X: np.ndarray,
    pca: PCA,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Preprocess new data using fitted transformers.
    
    Args:
        X: Raw feature data (n_samples, n_features_original)
        pca: Fitted PCA transformer
        scaler: Fitted StandardScaler transformer
        
    Returns:
        Preprocessed features ready for quantum encoding (n_samples, n_components)
    """
    X_reduced = pca.transform(X)
    X_scaled = scaler.transform(X_reduced)
    X_final = np.arctan(X_scaled) * 2
    return X_final
