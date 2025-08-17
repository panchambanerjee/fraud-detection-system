"""
Comprehensive test suite for fraud detection models.

This module tests all components of the fraud detection system:
- Data generation
- Feature engineering
- Model training and inference
- API endpoints
- Explainability tools
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_synthetic_data import FraudDataGenerator, TransactionData
from features.feature_engineering import FeatureEngineer, FeatureSet
from features.feature_store import InMemoryFeatureStore, FeatureStoreManager
from models.logistic_model import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from models.neural_network import DeepNeuralNetworkModel
from models.model_comparison import ModelComparison
from explainability.model_explainer import ModelExplainer

class TestDataGeneration:
    """Test synthetic data generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = FraudDataGenerator(seed=42)
    
    def test_generator_initialization(self):
        """Test data generator initialization."""
        assert self.generator is not None
        assert len(self.generator.fraud_patterns) > 0
        assert len(self.generator.user_profiles) == 1000
        assert len(self.generator.merchant_categories) > 0
    
    def test_transaction_generation(self):
        """Test transaction data generation."""
        # Generate small dataset for testing
        df = self.generator.generate_transactions(
            days=1,
            transactions_per_day=100,
            fraud_rate=0.01
        )
        
        assert len(df) == 100
        assert 'is_fraud' in df.columns
        assert df['is_fraud'].sum() > 0  # Should have some fraud
        assert df['is_fraud'].mean() < 0.05  # But not too many
        
        # Check required columns
        required_columns = [
            'user_id', 'amount', 'merchant_id', 'ip_address', 'email',
            'country', 'city', 'latitude', 'longitude'
        ]
        for col in required_columns:
            assert col in df.columns
    
    def test_fraud_patterns(self):
        """Test fraud pattern generation."""
        df = self.generator.generate_transactions(
            days=1,
            transactions_per_day=1000,
            fraud_rate=0.1  # Higher fraud rate for testing
        )
        
        fraud_transactions = df[df['is_fraud'] == True]
        assert len(fraud_transactions) > 0
        
        # Check fraud types
        fraud_types = fraud_transactions['fraud_type'].unique()
        expected_types = ['card_testing', 'stolen_card', 'account_takeover', 'merchant_fraud']
        
        for fraud_type in fraud_types:
            assert fraud_type in expected_types

class TestFeatureEngineering:
    """Test feature engineering pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_engineer = FeatureEngineer()
        
        # Create sample transaction data
        self.sample_transaction = {
            'user_id': 'test_user_123',
            'amount': 150.00,
            'merchant_id': 'merchant_test_456',
            'timestamp': '2024-01-15T10:30:00',
            'ip_address': '192.168.1.100',
            'email': 'test@example.com',
            'merchant_category': 'electronics',
            'country': 'US',
            'city': 'New York',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'device_type': 'desktop',
            'browser': 'chrome',
            'is_fraud': False
        }
    
    def test_feature_engineering_initialization(self):
        """Test feature engineering pipeline initialization."""
        assert self.feature_engineer is not None
        assert hasattr(self.feature_engineer, 'velocity_features')
        assert hasattr(self.feature_engineer, 'behavioral_features')
        assert hasattr(self.feature_engineer, 'network_features')
        assert hasattr(self.feature_engineer, 'geographic_features')
        assert hasattr(self.feature_engineer, 'temporal_features')
    
    def test_feature_computation(self):
        """Test feature computation for a transaction."""
        feature_set = self.feature_engineer.compute_features(self.sample_transaction)
        
        assert isinstance(feature_set, FeatureSet)
        assert feature_set.computed_at is not None
        
        # Check that features were computed
        assert len(feature_set.velocity_features) > 0
        assert len(feature_set.behavioral_features) > 0
        assert len(feature_set.network_features) > 0
        assert len(feature_set.geographic_features) > 0
        assert len(feature_set.temporal_features) > 0
    
    def test_feature_vector_generation(self):
        """Test feature vector generation."""
        feature_set = self.feature_engineer.compute_features(self.sample_transaction)
        feature_vector = self.feature_engineer.get_feature_vector(feature_set)
        
        assert isinstance(feature_vector, dict)
        assert len(feature_vector) > 0
        
        # Check feature names
        feature_names = self.feature_engineer.get_feature_names()
        assert len(feature_names) > 0
    
    def test_performance_target(self):
        """Test that feature computation meets performance target."""
        start_time = pd.Timestamp.now()
        
        # Compute features multiple times to test performance
        for _ in range(10):
            feature_set = self.feature_engineer.compute_features(self.sample_transaction)
        
        end_time = pd.Timestamp.now()
        total_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
        
        # Average time per computation should be < 100ms
        avg_time = total_time / 10
        assert avg_time < 100, f"Average feature computation time {avg_time:.2f}ms exceeds 100ms target"

class TestFeatureStore:
    """Test feature store functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_store = InMemoryFeatureStore(max_size=100)
        self.sample_features = {'feature1': 1.0, 'feature2': 2.0}
    
    def test_feature_storage(self):
        """Test storing and retrieving features."""
        key = "test_key"
        
        # Store features
        self.feature_store.set(key, self.sample_features, ttl=60)
        
        # Check if exists
        assert self.feature_store.exists(key)
        
        # Retrieve features
        retrieved_features = self.feature_store.get(key)
        assert retrieved_features['features'] == self.sample_features
    
    def test_feature_expiration(self):
        """Test feature expiration."""
        key = "expire_key"
        
        # Store with short TTL
        self.feature_store.set(key, self.sample_features, ttl=1)
        
        # Should exist initially
        assert self.feature_store.exists(key)
        
        # Wait for expiration (simulate)
        import time
        time.sleep(0.1)  # Small delay
        
        # Clean up expired features
        self.feature_store.cleanup_expired()
        
        # Should not exist after cleanup
        assert not self.feature_store.exists(key)
    
    def test_lru_eviction(self):
        """Test LRU eviction when store is full."""
        # Fill store beyond capacity
        for i in range(150):
            key = f"key_{i}"
            self.feature_store.set(key, self.sample_features, ttl=3600)
        
        # Store should not exceed max size
        assert len(self.feature_store.store) <= self.feature_store.max_size

class TestLogisticRegressionModel:
    """Test logistic regression model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LogisticRegressionModel()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create synthetic labels with some fraud
        fraud_rate = 0.1
        self.y_train = pd.Series(
            np.random.binomial(1, fraud_rate, n_samples),
            index=self.X_train.index
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert self.model.model_name == "logistic_regression_fraud"
        assert not self.model.is_trained
        assert self.model.hyperparameters['penalty'] == 'l1'
    
    def test_model_training(self):
        """Test model training."""
        # Train model
        results = self.model.train(self.X_train, self.y_train)
        
        assert self.model.is_trained
        assert results['training_time_seconds'] > 0
        assert results['feature_count'] == 20
        assert 'train_metrics' in results
        
        # Check training metrics
        train_metrics = results['train_metrics']
        assert 'roc_auc' in train_metrics
        assert 'precision' in train_metrics
        assert 'recall' in train_metrics
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Train model first
        self.model.train(self.X_train, self.y_train)
        
        # Test single prediction
        sample_features = {f'feature_{i}': np.random.randn() for i in range(20)}
        result = self.model.score_single(sample_features)
        
        assert 'prediction' in result
        assert 'probability' in result
        assert 'inference_time_ms' in result
        assert result['inference_time_ms'] < 100  # Performance target
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Train model
        self.model.train(self.X_train, self.y_train)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            self.model.save_model(tmp_file.name)
            
            # Create new model instance
            new_model = LogisticRegressionModel()
            assert not new_model.is_trained
            
            # Load model
            new_model.load_model(tmp_file.name)
            assert new_model.is_trained
            assert new_model.feature_names == self.model.feature_names
            
            # Clean up
            os.unlink(tmp_file.name)

class TestXGBoostModel:
    """Test XGBoost model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = XGBoostModel()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create synthetic labels
        fraud_rate = 0.1
        self.y_train = pd.Series(
            np.random.binomial(1, fraud_rate, n_samples),
            index=self.X_train.index
        )
        
        # Create validation data
        self.X_val = pd.DataFrame(
            np.random.randn(200, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_val = pd.Series(
            np.random.binomial(1, fraud_rate, 200),
            index=self.X_val.index
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert self.model.model_name == "xgboost_fraud"
        assert not self.model.is_trained
        assert self.model.hyperparameters['objective'] == 'binary:logistic'
    
    def test_model_training(self):
        """Test model training with validation."""
        # Train model
        results = self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        assert self.model.is_trained
        assert results['training_time_seconds'] > 0
        assert results['best_iteration'] > 0
        assert 'val_metrics' in results
        
        # Check validation metrics
        val_metrics = results['val_metrics']
        assert 'roc_auc' in val_metrics
        assert val_metrics['roc_auc'] > 0.5  # Should be better than random
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        # Train model
        self.model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        
        # Get feature importance
        importance = self.model._get_feature_importance()
        assert len(importance) > 0
        
        # Check importance plot data
        plot_data = self.model.get_feature_importance_plot(top_n=10)
        assert 'feature_names' in plot_data
        assert 'importance_scores' in plot_data

class TestDeepNeuralNetworkModel:
    """Test deep neural network model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = DeepNeuralNetworkModel()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create synthetic labels
        fraud_rate = 0.1
        self.y_train = pd.Series(
            np.random.binomial(1, fraud_rate, n_samples),
            index=self.X_train.index
        )
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert self.model.model_name == "deep_neural_network_fraud"
        assert not self.model.is_trained
        assert 'embedding_dim' in self.model.hyperparameters
    
    def test_model_building(self):
        """Test model architecture building."""
        input_dim = 20
        model = self.model.build_model(input_dim)
        
        assert model is not None
        assert hasattr(model, 'layers')
        assert model.input_shape == (None, input_dim)
        assert model.output_shape == (None, 1)
    
    @pytest.mark.skip(reason="TensorFlow training takes too long for unit tests")
    def test_model_training(self):
        """Test model training (skipped for unit tests)."""
        # This test would be run in integration tests
        pass

class TestModelComparison:
    """Test model comparison framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparison = ModelComparison()
        
        # Create mock models
        self.mock_lr = Mock()
        self.mock_lr.is_trained = True
        self.mock_lr.evaluate.return_value = Mock(
            roc_auc=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            false_positive_rate=0.01,
            true_positive_rate=0.75,
            inference_time_ms=50
        )
        self.mock_lr.get_model_info.return_value = {
            'model_size_mb': 0.1,
            'training_time_seconds': 10
        }
        
        self.mock_xgb = Mock()
        self.mock_xgb.is_trained = True
        self.mock_xgb.evaluate.return_value = Mock(
            roc_auc=0.90,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            false_positive_rate=0.01,
            true_positive_rate=0.80,
            inference_time_ms=80
        )
        self.mock_xgb.get_model_info.return_value = {
            'model_size_mb': 0.5,
            'training_time_seconds': 30
        }
    
    def test_model_addition(self):
        """Test adding models to comparison."""
        self.comparison.add_model("logistic_regression", self.mock_lr)
        self.comparison.add_model("xgboost", self.mock_xgb)
        
        assert len(self.comparison.models) == 2
        assert "logistic_regression" in self.comparison.models
        assert "xgboost" in self.comparison.models
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Add models
        self.comparison.add_model("logistic_regression", self.mock_lr)
        self.comparison.add_model("xgboost", self.mock_xgb)
        
        # Create mock test data
        X_test = pd.DataFrame(np.random.randn(100, 10))
        y_test = pd.Series(np.random.binomial(1, 0.1, 100))
        
        # Compare models
        results = self.comparison.compare_models(X_test, y_test)
        
        assert len(results) == 2
        assert "logistic_regression" in results
        assert "xgboost" in results
        
        # Check that XGBoost has better ROC-AUC
        lr_result = results["logistic_regression"]
        xgb_result = results["xgboost"]
        assert xgb_result.roc_auc > lr_result.roc_auc
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add models and compare
        self.comparison.add_model("logistic_regression", self.mock_lr)
        self.comparison.add_model("xgboost", self.mock_xgb)
        
        X_test = pd.DataFrame(np.random.randn(100, 10))
        y_test = pd.Series(np.random.binomial(1, 0.1, 100))
        
        self.comparison.compare_models(X_test, y_test)
        summary_df = self.comparison.create_performance_summary()
        
        assert len(summary_df) == 2
        assert 'ROC-AUC' in summary_df.columns
        assert 'Inference Time (ms)' in summary_df.columns
        
        # Should be sorted by ROC-AUC
        assert summary_df.iloc[0]['ROC-AUC'] >= summary_df.iloc[1]['ROC-AUC']

class TestModelExplainer:
    """Test model explainability."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.explainer = ModelExplainer()
        
        # Sample features
        self.sample_features = {
            'transactions_1h': 5,
            'amount_zscore': 2.5,
            'ip_fraud_rate': 0.3,
            'location_anomaly': 200,
            'rapid_transaction': 1
        }
        
        # Sample scoring result
        self.sample_result = {
            'risk_score': 0.75,
            'confidence': 'high'
        }
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        assert self.explainer is not None
        assert len(self.explainer.risk_templates) > 0
        assert len(self.explainer.feature_descriptions) > 0
    
    def test_risk_insights_generation(self):
        """Test risk insights generation."""
        insights = self.explainer.generate_risk_insights(
            self.sample_features,
            self.sample_result,
            'test_model'
        )
        
        assert 'risk_level' in insights
        assert 'risk_score' in insights
        assert 'top_risk_factors' in insights
        assert 'business_explanation' in insights
        assert 'recommendations' in insights
    
    def test_risk_factor_identification(self):
        """Test risk factor identification."""
        risk_factors = self.explainer._identify_top_risk_factors(
            self.sample_features,
            0.75
        )
        
        assert len(risk_factors) > 0
        assert len(risk_factors) <= 5  # Top 5 factors
        
        # Check risk factor structure
        for factor in risk_factors:
            assert 'feature_name' in factor
            assert 'risk_contribution' in factor
            assert 'description' in factor
            assert 'risk_type' in factor
    
    def test_business_explanation_generation(self):
        """Test business explanation generation."""
        risk_factors = self.explainer._identify_top_risk_factors(
            self.sample_features,
            0.75
        )
        
        explanation = self.explainer._generate_business_explanation(
            'high_risk',
            risk_factors,
            self.sample_features
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    def test_explanation_summary(self):
        """Test explanation summary generation."""
        summary = self.explainer.get_explanation_summary(
            self.sample_features,
            self.sample_result,
            'test_model'
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'HIGH RISK' in summary or 'MEDIUM RISK' in summary or 'LOW RISK' in summary

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])