"""
Logistic Regression model for fraud detection.

This module implements a simple but effective logistic regression model
with L1 regularization, optimized for <100ms inference time.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from pathlib import Path
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    true_positive_rate: float
    inference_time_ms: float

class LogisticRegressionModel:
    """
    Logistic Regression model with L1 regularization for fraud detection.
    
    This model serves as a baseline and is optimized for:
    - Fast inference (<100ms)
    - Interpretability
    - L1 regularization for feature selection
    - Production deployment
    """
    
    def __init__(self, model_name: str = "logistic_regression_fraud"):
        """
        Initialize the logistic regression model.
        
        Args:
            model_name: Name for the model
        """
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_time = 0.0
        self.last_training_date = None
        
        # Model hyperparameters
        self.hyperparameters = {
            'C': 1.0,  # Inverse of regularization strength
            'penalty': 'l1',  # L1 regularization (Lasso)
            'solver': 'liblinear',  # Fast solver for L1
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'  # Handle imbalanced classes
        }
        
        logger.info("Logistic Regression model initialized: %s", model_name)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training results dictionary
        """
        start_time = time.time()
        logger.info("Starting training for %s", self.model_name)
        
        try:
            # Store feature names
            self.feature_names = list(X_train.columns)
            logger.info("Training with %d features", len(self.feature_names))
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            
            # Initialize and train model
            self.model = LogisticRegression(**self.hyperparameters)
            self.model.fit(X_train_scaled, y_train)
            
            # Training time
            self.training_time = time.time() - start_time
            self.is_trained = True
            self.last_training_date = pd.Timestamp.now()
            
            # Evaluate on training set
            train_predictions = self.model.predict(X_train_scaled)
            train_probabilities = self.model.predict_proba(X_train_scaled)[:, 1]
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(y_train, train_predictions, train_probabilities)
            
            # Evaluate on validation set if provided
            val_metrics = None
            if X_val is not None and y_val is not None:
                val_predictions = self.model.predict(X_val_scaled)
                val_probabilities = self.model.predict_proba(X_val_scaled)[:, 1]
                val_metrics = self._calculate_metrics(y_val, val_predictions, val_probabilities)
            
            # Feature importance
            feature_importance = self._get_feature_importance()
            
            # Training results
            results = {
                'model_name': self.model_name,
                'training_time_seconds': self.training_time,
                'training_date': self.last_training_date.isoformat(),
                'feature_count': len(self.feature_names),
                'hyperparameters': self.hyperparameters,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'feature_importance': feature_importance,
                'model_size_mb': self._get_model_size()
            }
            
            logger.info("Training completed in %.2f seconds", self.training_time)
            logger.info("Training ROC-AUC: %.4f", train_metrics['roc_auc'])
            
            if val_metrics:
                logger.info("Validation ROC-AUC: %.4f", val_metrics['roc_auc'])
            
            return results
            
        except Exception as e:
            logger.error("Training failed: %s", str(e))
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions (0 = legitimate, 1 = fraud).
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability array [P(legitimate), P(fraud)]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def score_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Score a single transaction for fraud detection.
        
        Args:
            features: Dictionary of feature names and values
            
        Returns:
            Scoring results with prediction and probability
        """
        start_time = time.time()
        
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before scoring")
            
            # Convert features to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            missing_features = set(self.feature_names) - set(feature_df.columns)
            if missing_features:
                # Fill missing features with 0
                for feature in missing_features:
                    feature_df[feature] = 0.0
            
            # Reorder columns to match training order
            feature_df = feature_df[self.feature_names]
            
            # Make prediction
            prediction = self.predict(feature_df)[0]
            probability = self.predict_proba(feature_df)[0][1]  # P(fraud)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Performance check
            if inference_time > 100:
                logger.warning("Inference time %.2fms exceeds 100ms target", inference_time)
            
            results = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_score': float(probability),
                'inference_time_ms': round(inference_time, 2),
                'model_name': self.model_name,
                'feature_count': len(self.feature_names),
                'confidence': self._calculate_confidence(probability)
            }
            
            return results
            
        except Exception as e:
            logger.error("Error scoring transaction: %s", str(e))
            raise
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelPerformance:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            ModelPerformance object with metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        start_time = time.time()
        
        # Make predictions
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='binary', zero_division=0
        )
        
        roc_auc = roc_auc_score(y_test, probabilities)
        
        # Calculate FPR and TPR at 1% FPR threshold
        fpr, tpr, thresholds = self._calculate_roc_curve(y_test, probabilities)
        fpr_1pct_idx = np.argmin(np.abs(fpr - 0.01))
        false_positive_rate = fpr[fpr_1pct_idx]
        true_positive_rate = tpr[fpr_1pct_idx]
        
        performance = ModelPerformance(
            roc_auc=roc_auc,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            inference_time_ms=inference_time
        )
        
        logger.info("Model evaluation completed:")
        logger.info("ROC-AUC: %.4f", roc_auc)
        logger.info("Precision: %.4f", precision)
        logger.info("Recall: %.4f", recall)
        logger.info("F1-Score: %.4f", f1)
        logger.info("Inference time: %.2fms", inference_time)
        
        return performance
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        roc_auc = roc_auc_score(y_true, y_prob)
        
        return {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': (y_true == y_pred).mean()
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model coefficients."""
        if not self.is_trained:
            return {}
        
        # Get coefficients (absolute values for L1 regularization)
        coefficients = np.abs(self.model.coef_[0])
        
        # Create feature importance dictionary
        feature_importance = {}
        for feature, coef in zip(self.feature_names, coefficients):
            feature_importance[feature] = float(coef)
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), 
                  key=lambda x: x[1], reverse=True)
        )
        
        return feature_importance
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calculate confidence level based on probability."""
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"
    
    def _calculate_roc_curve(self, y_true: pd.Series, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ROC curve points."""
        from sklearn.metrics import roc_curve
        return roc_curve(y_true, y_prob)
    
    def _get_model_size(self) -> float:
        """Get model size in MB."""
        if not self.is_trained:
            return 0.0
        
        # Estimate model size
        n_features = len(self.feature_names)
        n_coefficients = n_features + 1  # +1 for intercept
        
        # 8 bytes per float64
        size_bytes = n_coefficients * 8
        size_mb = size_bytes / (1024 * 1024)
        
        return round(size_mb, 4)
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'hyperparameters': self.hyperparameters,
                'training_time': self.training_time,
                'last_training_date': self.last_training_date,
                'model_name': self.model_name
            }
            
            joblib.dump(model_data, filepath)
            logger.info("Model saved to %s", filepath)
            
        except Exception as e:
            logger.error("Error saving model: %s", str(e))
            raise
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.hyperparameters = model_data['hyperparameters']
            self.training_time = model_data['training_time']
            self.last_training_date = model_data['last_training_date']
            self.model_name = model_data['model_name']
            self.is_trained = True
            
            logger.info("Model loaded from %s", filepath)
            logger.info("Model trained on %s with %d features", 
                       self.last_training_date, len(self.feature_names))
            
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': self.model_name,
            'model_type': 'Logistic Regression',
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'hyperparameters': self.hyperparameters,
            'training_time_seconds': self.training_time,
            'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
            'model_size_mb': self._get_model_size(),
            'feature_names': self.feature_names[:10] + ['...'] if len(self.feature_names) > 10 else self.feature_names
        }
    
    def update_hyperparameters(self, new_params: Dict[str, Any]):
        """
        Update model hyperparameters.
        
        Args:
            new_params: New hyperparameter values
        """
        # Validate hyperparameters
        valid_params = set(self.hyperparameters.keys())
        invalid_params = set(new_params.keys()) - valid_params
        
        if invalid_params:
            raise ValueError(f"Invalid hyperparameters: {invalid_params}")
        
        # Update hyperparameters
        self.hyperparameters.update(new_params)
        logger.info("Hyperparameters updated: %s", new_params)
        
        # Reset training status
        self.is_trained = False
        self.training_time = 0.0
        self.last_training_date = None