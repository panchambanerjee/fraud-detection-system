"""
Model explainability module for fraud detection system.

This module provides SHAP integration, risk insights generation,
and human-readable explanations for fraud decisions.
"""

from .model_explainer import (
    ModelExplainer,
    RiskInsight,
    FeatureContribution
)

__all__ = [
    "ModelExplainer",
    "RiskInsight",
    "FeatureContribution"
]