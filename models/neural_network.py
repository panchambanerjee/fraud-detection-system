"""
Deep Neural Network model for fraud detection.

This module implements a multi-branch neural network architecture
inspired by ResNeXt, optimized for <100ms inference time and
high accuracy (target: ROC-AUC > 0.92).
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from pathlib import Path
import time
from dataclasses import dataclass
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

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

class DeepNeuralNetworkModel:
    """
    Deep Neural Network model with multi-branch architecture for fraud detection.
    
    This model implements a ResNeXt-inspired architecture that combines:
    - Memorization power (like XGBoost)
    - Generalization ability (deep learning)
    - Fast inference (<100ms)
    - High accuracy (target: ROC-AUC > 0.92)
    """
    
    def __init__(self, model_name: str = "deep_neural_network_fraud"):
        """
        Initialize the deep neural network model.
        
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
        self.training_history = None
        
        # Model hyperparameters
        self.hyperparameters = {
            'input_dim': None,  # Will be set during training
            'embedding_dim': 64,
            'num_branches': 4,
            'branch_depth': 3,
            'branch_width': 32,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 100,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 10,
            'class_weight': {0: 1.0, 1: 10.0}  # Handle class imbalance
        }
        
        logger.info("Deep Neural Network model initialized: %s", model_name)
    
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build the multi-branch neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        self.hyperparameters['input_dim'] = input_dim
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,))
        
        # Initial dense layer with batch normalization
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.hyperparameters['dropout_rate'])(x)
        
        # Multi-branch architecture (ResNeXt-inspired)
        x = self._build_multi_branch_block(x)
        
        # Global average pooling and final layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.hyperparameters['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.hyperparameters['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        logger.info("Built DNN model with %d input features", input_dim)
        return model
    
    def _build_multi_branch_block(self, x: tf.Tensor) -> tf.Tensor:
        """
        Build multi-branch block for feature learning.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with multi-branch processing
        """
        num_branches = self.hyperparameters['num_branches']
        branch_depth = self.hyperparameters['branch_depth']
        branch_width = self.hyperparameters['branch_width']
        
        # Create multiple branches
        branches = []
        for i in range(num_branches):
            branch = x
            
            # Process through branch layers
            for j in range(branch_depth):
                branch = layers.Dense(branch_width, activation='relu')(branch)
                branch = layers.BatchNormalization()(branch)
                branch = layers.Dropout(self.hyperparameters['dropout_rate'])(branch)
            
            branches.append(branch)
        
        # Concatenate branches
        if len(branches) > 1:
            x = layers.Concatenate()(branches)
        else:
            x = branches[0]
        
        # Add residual connection if dimensions match
        if x.shape[-1] == self.hyperparameters['input_dim']:
            x = layers.Add()([x, inputs])
        
        return x
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the deep neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training results dictionary
        """
        start_time = time.time()
        logger.info("Starting DNN training for %s", self.model_name)
        
        try:
            # Store feature names
            self.feature_names = list(X_train.columns)
            input_dim = len(self.feature_names)
            logger.info("Training with %d features", input_dim)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            
            # Build model
            self.model = self.build_model(input_dim)
            
            # Prepare callbacks
            callbacks_list = self._prepare_callbacks(X_val_scaled, y_val)
            
            # Training data
            train_data = (X_train_scaled, y_train)
            validation_data = (X_val_scaled, y_val) if X_val is not None else None
            
            # Train model
            history = self.model.fit(
                X_train_scaled, y_train,
                batch_size=self.hyperparameters['batch_size'],
                epochs=self.hyperparameters['epochs'],
                validation_data=validation_data,
                callbacks=callbacks_list,
                class_weight=self.hyperparameters['class_weight'],
                verbose=1
            )
            
            # Store training history
            self.training_history = history.history
            
            # Training time
            self.training_time = time.time() - start_time
            self.is_trained = True
            self.last_training_date = pd.Timestamp.now()
            
            # Evaluate on training set
            train_predictions = self.predict(X_train)
            train_probabilities = self.predict_proba(X_train)
            
            # Calculate training metrics
            train_metrics = self._calculate_metrics(y_train, train_predictions, train_probabilities)
            
            # Evaluate on validation set if provided
            val_metrics = None
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val)
                val_probabilities = self.predict_proba(X_val)
                val_metrics = self._calculate_metrics(y_val, val_predictions, val_probabilities)
            
            # Training results
            results = {
                'model_name': self.model_name,
                'training_time_seconds': self.training_time,
                'training_date': self.last_training_date.isoformat(),
                'feature_count': len(self.feature_names),
                'hyperparameters': self.hyperparameters,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_size_mb': self._get_model_size(),
                'training_history': self.training_history
            }
            
            logger.info("Training completed in %.2f seconds", self.training_time)
            logger.info("Training ROC-AUC: %.4f", train_metrics['roc_auc'])
            
            if val_metrics:
                logger.info("Validation ROC-AUC: %.4f", val_metrics['roc_auc'])
            
            return results
            
        except Exception as e:
            logger.error("Training failed: %s", str(e))
            raise
    
    def _prepare_callbacks(self, X_val: np.ndarray, y_val: pd.Series) -> List[callbacks.Callback]:
        """Prepare training callbacks."""
        callbacks_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.hyperparameters['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.hyperparameters['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            'best_dnn_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        return callbacks_list
    
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
        probabilities = self.model.predict(X_scaled, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()
        
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
        
        X_scaled = self.scaler.transform(X)
        fraud_probs = self.model.predict(X_scaled, verbose=0).flatten()
        
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
        
        # Get model parameters count
        total_params = self.model.count_params()
        
        # Estimate size (4 bytes per float32 parameter)
        size_bytes = total_params * 4
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
            # Save Keras model
            model_path = filepath.replace('.pkl', '_keras.h5')
            self.model.save(model_path)
            
            # Save other components
            model_data = {
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'hyperparameters': self.hyperparameters,
                'training_time': self.training_time,
                'last_training_date': self.last_training_date,
                'model_name': self.model_name,
                'training_history': self.training_history,
                'keras_model_path': model_path
            }
            
            joblib.dump(model_data, filepath)
            logger.info("DNN model saved to %s and %s", filepath, model_path)
            
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
            # Load other components
            model_data = joblib.load(filepath)
            
            # Load Keras model
            keras_model_path = model_data['keras_model_path']
            self.model = keras.models.load_model(keras_model_path)
            
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.hyperparameters = model_data['hyperparameters']
            self.training_time = model_data['training_time']
            self.last_training_date = model_data['last_training_date']
            self.model_name = model_data['model_name']
            self.training_history = model_data['training_history']
            self.is_trained = True
            
            logger.info("DNN model loaded from %s", filepath)
            logger.info("Model trained on %s with %d features", 
                       self.last_training_date, len(self.feature_names))
            
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_name': self.model_name,
            'model_type': 'Deep Neural Network',
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
        self.training_history = None
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history for plotting."""
        if not self.is_trained or self.training_history is None:
            return {}
        
        return self.training_history