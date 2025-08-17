"""
FastAPI application module for fraud detection system.

This module provides production-ready API endpoints for real-time fraud scoring,
model management, and system monitoring.
"""

from .fraud_api import app, FraudAPI

__all__ = ["app", "FraudAPI"]