"""
Classical machine learning baseline models for comparison.

This module provides implementations of classical ML models used as baselines.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb


class ClassicalBaselines:
    """
    Collection of classical machine learning baseline models.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize classical baseline models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_state
            ),
            'GradientBoosting': GradientBoostingClassifier(
                random_state=random_state
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=random_state
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(128, 128),
                max_iter=500,
                random_state=random_state
            ),
            'DecisionTree': DecisionTreeClassifier(
                random_state=random_state
            )
        }
        self.trained_models = {}
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        print("Training classical baseline models...")
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        return self.trained_models
    
    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of accuracy scores
        """
        results = {}
        for name, model in self.trained_models.items():
            accuracy = model.score(X_test, y_test)
            results[name] = accuracy
            print(f"  {name}: {accuracy:.4f}")
        return results
    
    def get_model(self, name: str):
        """Get a specific trained model."""
        return self.trained_models.get(name)
