"""
ML models module for fraud detection system.

This module implements three model architectures:
1. Logistic Regression - Simple baseline with L1 regularization
2. XGBoost - Tree-based model with early stopping  
3. Deep Neural Network - Multi-branch architecture (ResNeXt-inspired)

All models are optimized for <100ms inference time.
"""

from .logistic_model import LogisticRegressionModel
from .xgboost_model import XGBoostModel
from .neural_network import DeepNeuralNetworkModel
from .model_comparison import ModelComparison

__all__ = [
    "LogisticRegressionModel",
    "XGBoostModel",
    "DeepNeuralNetworkModel", 
    "ModelComparison"
]