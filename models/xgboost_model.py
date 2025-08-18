"""
XGBoost model for fraud detection.

This module implements an XGBoost model with early stopping and
optimization for <100ms inference time, providing better performance
than logistic regression while maintaining interpretability.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from pathlib import Path
import time
from dataclasses import dataclass
import warnings

# Suppress XGBoost warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

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

class XGBoostModel:
    """
    XGBoost model for fraud detection with early stopping.
    
    This model provides better performance than logistic regression
    and is optimized for:
    - High accuracy (target: ROC-AUC > 0.90)
    - Fast inference (<100ms)
    - Feature importance analysis
    - Early stopping to prevent overfitting
    """
    
    def __init__(self, model_name: str = "xgboost_fraud"):
        """
        Initialize the XGBoost model.
        
        Args:
            model_name: Name for the model
        """
        self.model_name = model_name
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.training_time = 0.0
        self.last_training_date = None
        self.best_iteration = 0
        
        # Model hyperparameters optimized for fraud detection
        self.hyperparameters = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,  # Prevent overfitting
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores
            'tree_method': 'hist',  # Fast tree method
            'early_stopping_rounds': 50
        }
        
        logger.info("XGBoost model initialized: %s", model_name)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              num_boost_round: int = 1000) -> Dict[str, Any]:
        """
        Train the XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (required for early stopping)
            y_val: Validation labels (required for early stopping)
            num_boost_round: Maximum number of boosting rounds
            
        Returns:
            Training results dictionary
        """
        start_time = time.time()
        logger.info("Starting XGBoost training for %s", self.model_name)
        
        if X_val is None or y_val is None:
            raise ValueError("Validation data is required for XGBoost training with early stopping")
        
        try:
            # Store feature names
            self.feature_names = list(X_train.columns)
            logger.info("Training with %d features", len(self.feature_names))
            
            # Prepare DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            
            # Training parameters
            params = self.hyperparameters.copy()
            
            # Train model with early stopping
            evals = [(dtrain, 'train'), (dval, 'eval')]
            
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                verbose_eval=False,
                early_stopping_rounds=params['early_stopping_rounds']
            )
            
            # Get best iteration
            self.best_iteration = self.model.best_iteration
            
            # Training time
            self.training_time = time.time() - start_time
            self.is_trained = True
            self.last_training_date = pd.Timestamp.now()
            
            # Evaluate on training set
            train_predictions = self.predict(X_train)
            train_probabilities = self.predict_proba(X_train)
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(y_train, train_predictions, train_probabilities)
            
            # Evaluate on validation set
            val_predictions = self.predict(X_val)
            val_probabilities = self.predict_proba(X_val)
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
                'best_iteration': self.best_iteration,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'feature_importance': feature_importance,
                'model_size_mb': self._get_model_size()
            }
            
            logger.info("Training completed in %.2f seconds", self.training_time)
            logger.info("Best iteration: %d", self.best_iteration)
            logger.info("Training ROC-AUC: %.4f", train_metrics['roc_auc'])
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
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Make predictions
        probabilities = self.model.predict(dtest, iteration_range=(0, self.best_iteration))
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions
    
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
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Get fraud probabilities
        fraud_probs = self.model.predict(dtest, iteration_range=(0, self.best_iteration))
        
        # Convert to [P(legitimate), P(fraud)] format
        legitimate_probs = 1 - fraud_probs
        probabilities = np.column_stack([legitimate_probs, fraud_probs])
        
        return probabilities
    
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
                'confidence': self._calculate_confidence(probability),
                'best_iteration': self.best_iteration
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
        
        # For binary classification, we need the positive class probability (fraud)
        if y_prob.ndim > 1 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]  # Take fraud probability
        roc_auc = roc_auc_score(y_true, y_prob)
        
        return {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': (y_true == y_pred).mean()
        }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost model."""
        if not self.is_trained:
            return {}
        
        # Get feature importance scores
        importance_scores = self.model.get_score(importance_type='gain')
        
        # Create feature importance dictionary
        feature_importance = {}
        for feature in self.feature_names:
            feature_importance[feature] = float(importance_scores.get(feature, 0.0))
        
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
        
        # Estimate model size based on number of trees and features
        n_trees = self.best_iteration
        n_features = len(self.feature_names)
        
        # Rough estimate: each tree stores feature indices and thresholds
        # 4 bytes per feature index + 8 bytes per threshold per tree
        size_bytes = n_trees * (n_features * 4 + 8)
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
                'feature_names': self.feature_names,
                'hyperparameters': self.hyperparameters,
                'training_time': self.training_time,
                'last_training_date': self.last_training_date,
                'model_name': self.model_name,
                'best_iteration': self.best_iteration
            }
            
            joblib.dump(model_data, filepath)
            logger.info("XGBoost model saved to %s", filepath)
            
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
            self.feature_names = model_data['feature_names']
            self.hyperparameters = model_data['hyperparameters']
            self.training_time = model_data['training_time']
            self.last_training_date = model_data['last_training_date']
            self.model_name = model_data['model_name']
            self.best_iteration = model_data['best_iteration']
            self.is_trained = True
            
            logger.info("XGBoost model loaded from %s", filepath)
            logger.info("Model trained on %s with %d features, best iteration: %d", 
                       self.last_training_date, len(self.feature_names), self.best_iteration)
            
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': self.model_name,
            'model_type': 'XGBoost',
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'hyperparameters': self.hyperparameters,
            'training_time_seconds': self.training_time,
            'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
            'best_iteration': self.best_iteration,
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
        self.best_iteration = 0
    
    def get_feature_importance_plot(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Get feature importance data for plotting.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature names and importance scores
        """
        if not self.is_trained:
            return {}
        
        feature_importance = self._get_feature_importance()
        
        # Get top N features
        top_features = dict(list(feature_importance.items())[:top_n])
        
        return {
            'feature_names': list(top_features.keys()),
            'importance_scores': list(top_features.values()),
            'total_features': len(self.feature_names)
        }