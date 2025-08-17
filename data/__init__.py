"""
Data generation and management module for fraud detection system.

This module handles synthetic data generation with realistic fraud patterns,
data validation, and data pipeline management.
"""

from .generate_synthetic_data import (
    FraudDataGenerator,
    TransactionData,
    FraudPattern,
    UserProfile
)

__all__ = [
    "FraudDataGenerator",
    "TransactionData", 
    "FraudPattern",
    "UserProfile"
]