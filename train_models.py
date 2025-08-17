#!/usr/bin/env python3
"""
Model training script for fraud detection system.

This script demonstrates how to train all three models (Logistic Regression,
XGBoost, and Deep Neural Network) on synthetic data and save them for
production use.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our modules
from data.generate_synthetic_data import FraudDataGenerator
from features.feature_engineering import FeatureEngineer
from models.logistic_model import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from models.neural_network import DeepNeuralNetworkModel
from models.model_comparison import ModelComparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_training_data():
    """Generate synthetic training data."""
    logger.info("Generating synthetic training data...")
    
    generator = FraudDataGenerator(seed=42)
    
    # Generate 30 days of data
    df = generator.generate_transactions(
        days=30,
        transactions_per_day=10000,
        fraud_rate=0.001  # 0.1% fraud rate
    )
    
    logger.info("Generated %d transactions (%d fraud cases, %.3f%% fraud rate)",
               len(df), df['is_fraud'].sum(), df['is_fraud'].mean() * 100)
    
    return df

def engineer_features(df):
    """Engineer features for all transactions."""
    logger.info("Engineering features for %d transactions...", len(df))
    
    feature_engineer = FeatureEngineer()
    all_features = []
    
    # Process transactions in batches for memory efficiency
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        batch_features = []
        for _, transaction in batch.iterrows():
            try:
                # Convert to dictionary
                transaction_dict = transaction.to_dict()
                
                # Compute features
                feature_set = feature_engineer.compute_features(transaction_dict)
                feature_vector = feature_engineer.get_feature_vector(feature_set)
                
                batch_features.append(feature_vector)
                
            except Exception as e:
                logger.warning("Error computing features for transaction %s: %s", 
                             transaction.get('transaction_id', 'unknown'), str(e))
                # Fill with zeros if feature computation fails
                batch_features.append({})
        
        all_features.extend(batch_features)
        
        if (i + batch_size) % 5000 == 0:
            logger.info("Processed %d/%d transactions", min(i + batch_size, len(df)), len(df))
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Fill NaN values with 0
    features_df = features_df.fillna(0)
    
    logger.info("Feature engineering completed. Feature matrix shape: %s", features_df.shape)
    
    return features_df

def prepare_training_data(features_df, labels):
    """Prepare training and validation data."""
    logger.info("Preparing training and validation data...")
    
    # Remove any rows with infinite values
    features_df = features_df.replace([np.inf, -np.inf], 0)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features_df, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    logger.info("Training set: %d samples (%d fraud)", 
               len(X_train), y_train.sum())
    logger.info("Validation set: %d samples (%d fraud)", 
               len(X_val), y_val.sum())
    
    return X_train, X_val, y_train, y_val

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train logistic regression model."""
    logger.info("Training Logistic Regression model...")
    
    model = LogisticRegressionModel()
    
    # Train model
    start_time = time.time()
    results = model.train(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time
    
    logger.info("Logistic Regression training completed in %.2f seconds", training_time)
    logger.info("Training ROC-AUC: %.4f", results['train_metrics']['roc_auc'])
    if 'val_metrics' in results:
        logger.info("Validation ROC-AUC: %.4f", results['val_metrics']['roc_auc'])
    
    return model, results

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")
    
    model = XGBoostModel()
    
    # Train model
    start_time = time.time()
    results = model.train(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time
    
    logger.info("XGBoost training completed in %.2f seconds", training_time)
    logger.info("Best iteration: %d", results['best_iteration'])
    logger.info("Training ROC-AUC: %.4f", results['train_metrics']['roc_auc'])
    logger.info("Validation ROC-AUC: %.4f", results['val_metrics']['roc_auc'])
    
    return model, results

def train_neural_network(X_train, y_train, X_val, y_val):
    """Train deep neural network model."""
    logger.info("Training Deep Neural Network model...")
    
    model = DeepNeuralNetworkModel()
    
    # Train model
    start_time = time.time()
    results = model.train(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time
    
    logger.info("Neural Network training completed in %.2f seconds", training_time)
    logger.info("Training ROC-AUC: %.4f", results['train_metrics']['roc_auc'])
    if 'val_metrics' in results:
        logger.info("Validation ROC-AUC: %.4f", results['val_metrics']['roc_auc'])
    
    return model, results

def evaluate_models(models_dict, X_val, y_val):
    """Evaluate all trained models."""
    logger.info("Evaluating all models on validation set...")
    
    comparison = ModelComparison()
    
    # Add models to comparison
    for name, model in models_dict.items():
        comparison.add_model(name, model)
    
    # Compare models
    comparison_results = comparison.compare_models(X_val, y_val)
    
    # Create performance summary
    summary_df = comparison.create_performance_summary()
    
    logger.info("\n" + "="*80)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info("\n%s", summary_df.to_string(index=False))
    
    # Get best model
    best_model_name, best_roc_auc = comparison.get_best_model('roc_auc')
    logger.info("\nBest model by ROC-AUC: %s (%.4f)", best_model_name, best_roc_auc)
    
    return comparison, comparison_results

def save_models(models_dict, output_dir="models/trained"):
    """Save all trained models."""
    logger.info("Saving trained models to %s", output_dir)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_models = {}
    
    for name, model in models_dict.items():
        try:
            model_path = output_path / f"{name}_model.pkl"
            model.save_model(str(model_path))
            saved_models[name] = str(model_path)
            logger.info("Saved %s model to %s", name, model_path)
            
        except Exception as e:
            logger.error("Failed to save %s model: %s", name, str(e))
    
    return saved_models

def main():
    """Main training pipeline."""
    logger.info("Starting fraud detection model training pipeline...")
    
    try:
        # Step 1: Generate synthetic data
        df = generate_training_data()
        
        # Step 2: Engineer features
        features_df = engineer_features(df)
        
        # Step 3: Prepare training data
        X_train, X_val, y_train, y_val = prepare_training_data(
            features_df, df['is_fraud']
        )
        
        # Step 4: Train models
        models_dict = {}
        training_results = {}
        
        # Train Logistic Regression
        lr_model, lr_results = train_logistic_regression(X_train, y_train, X_val, y_val)
        models_dict['logistic_regression'] = lr_model
        training_results['logistic_regression'] = lr_results
        
        # Train XGBoost
        xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
        models_dict['xgboost'] = xgb_model
        training_results['xgboost'] = xgb_results
        
        # Train Neural Network
        nn_model, nn_results = train_neural_network(X_train, y_train, X_val, y_val)
        models_dict['neural_network'] = nn_model
        training_results['neural_network'] = nn_results
        
        # Step 5: Evaluate and compare models
        comparison, comparison_results = evaluate_models(models_dict, X_val, y_val)
        
        # Step 6: Save models
        saved_models = save_models(models_dict)
        
        # Step 7: Generate training report
        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("Models trained: %s", list(models_dict.keys()))
        logger.info("Models saved: %s", list(saved_models.keys()))
        logger.info("Best model: %s", comparison.get_best_model('roc_auc')[0])
        
        # Save training summary
        summary_path = Path("models/training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        training_summary = {
            'training_date': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'total_transactions': len(df),
                'fraud_count': df['is_fraud'].sum(),
                'fraud_rate': df['is_fraud'].mean(),
                'feature_count': len(features_df.columns)
            },
            'model_performance': {
                name: {
                    'roc_auc': results['val_metrics']['roc_auc'] if 'val_metrics' in results else results['train_metrics']['roc_auc'],
                    'training_time': results['training_time_seconds'],
                    'model_size_mb': results['model_size_mb']
                }
                for name, results in training_results.items()
            },
            'saved_models': saved_models
        }
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info("Training summary saved to %s", summary_path)
        
    except Exception as e:
        logger.error("Training pipeline failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()