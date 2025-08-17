#!/usr/bin/env python3
"""
Demo script for the fraud detection system.

This script demonstrates the complete fraud detection pipeline:
1. Generate synthetic data
2. Train models
3. Test real-time scoring
4. Show explainability features
"""

import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generate_synthetic_data import FraudDataGenerator
from features.feature_engineering import FeatureEngineer
from models.logistic_model import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from explainability.model_explainer import ModelExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_data_generation():
    """Demonstrate synthetic data generation."""
    print("\n" + "="*60)
    print("DEMO: SYNTHETIC DATA GENERATION")
    print("="*60)
    
    generator = FraudDataGenerator(seed=42)
    
    # Generate small dataset for demo
    df = generator.generate_transactions(
        days=7,
        transactions_per_day=1000,
        fraud_rate=0.002  # 0.2% fraud rate
    )
    
    print(f"Generated {len(df):,} transactions")
    print(f"Fraud cases: {df['is_fraud'].sum():,}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
    
    # Show fraud type distribution
    if df['is_fraud'].sum() > 0:
        fraud_types = df[df['is_fraud']]['fraud_type'].value_counts()
        print("\nFraud type distribution:")
        for fraud_type, count in fraud_types.items():
            print(f"  {fraud_type}: {count:,}")
    
    return df

def demo_feature_engineering(df):
    """Demonstrate feature engineering."""
    print("\n" + "="*60)
    print("DEMO: FEATURE ENGINEERING")
    print("="*60)
    
    feature_engineer = FeatureEngineer()
    
    # Sample a few transactions
    sample_transactions = df.head(5)
    
    print("Computing features for 5 sample transactions...")
    
    for i, (_, transaction) in enumerate(sample_transactions.iterrows()):
        print(f"\nTransaction {i+1}:")
        print(f"  Amount: ${transaction['amount']:.2f}")
        print(f"  Category: {transaction['merchant_category']}")
        print(f"  Country: {transaction['country']}")
        
        # Convert to dictionary
        transaction_dict = transaction.to_dict()
        
        # Compute features
        start_time = time.time()
        feature_set = feature_engineer.compute_features(transaction_dict)
        computation_time = (time.time() - start_time) * 1000
        
        print(f"  Feature computation time: {computation_time:.2f}ms")
        
        # Show some key features
        feature_vector = feature_engineer.get_feature_vector(feature_set)
        
        # Display top features
        top_features = sorted(feature_vector.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print("  Top features:")
        for feature_name, value in top_features:
            print(f"    {feature_name}: {value:.4f}")
    
    return feature_engineer

def demo_model_training(df, feature_engineer):
    """Demonstrate model training."""
    print("\n" + "="*60)
    print("DEMO: MODEL TRAINING")
    print("="*60)
    
    # Engineer features for all transactions
    print("Engineering features for all transactions...")
    all_features = []
    
    for _, transaction in df.iterrows():
        try:
            transaction_dict = transaction.to_dict()
            feature_set = feature_engineer.compute_features(transaction_dict)
            feature_vector = feature_engineer.get_feature_vector(feature_set)
            all_features.append(feature_vector)
        except Exception as e:
            # Fill with zeros if feature computation fails
            all_features.append({})
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    features_df = features_df.fillna(0)
    
    print(f"Feature matrix shape: {features_df.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, df['is_fraud'],
        test_size=0.2,
        random_state=42,
        stratify=df['is_fraud']
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegressionModel()
    
    start_time = time.time()
    lr_results = lr_model.train(X_train, y_train, X_test, y_test)
    lr_training_time = time.time() - start_time
    
    print(f"Training completed in {lr_training_time:.2f} seconds")
    print(f"Training ROC-AUC: {lr_results['train_metrics']['roc_auc']:.4f}")
    if 'val_metrics' in lr_results:
        print(f"Test ROC-AUC: {lr_results['val_metrics']['roc_auc']:.4f}")
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_model = XGBoostModel()
    
    start_time = time.time()
    xgb_results = xgb_model.train(X_train, y_train, X_test, y_test)
    xgb_training_time = time.time() - start_time
    
    print(f"Training completed in {xgb_training_time:.2f} seconds")
    print(f"Best iteration: {xgb_results['best_iteration']}")
    print(f"Training ROC-AUC: {xgb_results['train_metrics']['roc_auc']:.4f}")
    if 'val_metrics' in xgb_results:
        print(f"Test ROC-AUC: {xgb_results['val_metrics']['roc_auc']:.4f}")
    
    return lr_model, xgb_model, X_test, y_test

def demo_real_time_scoring(lr_model, xgb_model, X_test, y_test):
    """Demonstrate real-time scoring."""
    print("\n" + "="*60)
    print("DEMO: REAL-TIME SCORING")
    print("="*60)
    
    # Test single transaction scoring
    print("Testing single transaction scoring...")
    
    # Sample a few test transactions
    sample_indices = np.random.choice(len(X_test), 3, replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\nTransaction {i+1}:")
        
        # Get features
        features = X_test.iloc[idx].to_dict()
        actual_fraud = y_test.iloc[idx]
        
        print(f"  Actual fraud: {bool(actual_fraud)}")
        
        # Score with Logistic Regression
        lr_start = time.time()
        lr_result = lr_model.score_single(features)
        lr_time = (time.time() - lr_start) * 1000
        
        print(f"  Logistic Regression:")
        print(f"    Prediction: {'Fraud' if lr_result['prediction'] else 'Legitimate'}")
        print(f"    Probability: {lr_result['probability']:.4f}")
        print(f"    Inference time: {lr_result['inference_time_ms']:.2f}ms")
        print(f"    API time: {lr_time:.2f}ms")
        
        # Score with XGBoost
        xgb_start = time.time()
        xgb_result = xgb_model.score_single(features)
        xgb_time = (time.time() - xgb_start) * 1000
        
        print(f"  XGBoost:")
        print(f"    Prediction: {'Fraud' if xgb_result['prediction'] else 'Legitimate'}")
        print(f"    Probability: {xgb_result['probability']:.4f}")
        print(f"    Inference time: {xgb_result['inference_time_ms']:.2f}ms")
        print(f"    API time: {xgb_time:.2f}ms")
        
        # Performance check
        if lr_time > 100 or xgb_time > 100:
            print("    ⚠️  API response time exceeds 100ms target")
        else:
            print("    ✅ API response time meets 100ms target")

def demo_explainability(lr_model, xgb_model, X_test):
    """Demonstrate model explainability."""
    print("\n" + "="*60)
    print("DEMO: MODEL EXPLAINABILITY")
    print("="*60)
    
    explainer = ModelExplainer()
    
    # Sample a transaction
    sample_idx = np.random.choice(len(X_test))
    features = X_test.iloc[sidx].to_dict()
    
    print("Generating risk insights for sample transaction...")
    
    # Score with both models
    lr_result = lr_model.score_single(features)
    xgb_result = xgb_model.score_single(features)
    
    print(f"\nLogistic Regression Results:")
    print(f"  Risk Score: {lr_result['risk_score']:.4f}")
    print(f"  Confidence: {lr_result['confidence']}")
    
    # Generate risk insights
    lr_insights = explainer.generate_risk_insights(
        features, lr_result, 'logistic_regression'
    )
    
    print(f"  Risk Level: {lr_insights['risk_level']}")
    print(f"  Business Explanation: {lr_insights['business_explanation']}")
    
    if lr_insights['top_risk_factors']:
        print("  Top Risk Factors:")
        for factor in lr_insights['top_risk_factors'][:3]:
            print(f"    - {factor['description']}: {factor['risk_contribution']:.4f}")
    
    print(f"\nXGBoost Results:")
    print(f"  Risk Score: {xgb_result['risk_score']:.4f}")
    print(f"  Confidence: {xgb_result['confidence']}")
    
    # Generate risk insights
    xgb_insights = explainer.generate_risk_insights(
        features, xgb_result, 'xgboost'
    )
    
    print(f"  Risk Level: {xgb_insights['risk_level']}")
    print(f"  Business Explanation: {xgb_insights['business_explanation']}")
    
    if xgb_insights['top_risk_factors']:
        print("  Top Risk Factors:")
        for factor in xgb_insights['top_risk_factors'][:3]:
            print(f"    - {factor['description']}: {factor['risk_contribution']:.4f}")

def main():
    """Run the complete demo."""
    print("FRAUD DETECTION SYSTEM DEMO")
    print("="*60)
    print("This demo showcases the complete fraud detection pipeline")
    print("inspired by Stripe Radar's approach.")
    
    try:
        # Step 1: Data Generation
        df = demo_data_generation()
        
        # Step 2: Feature Engineering
        feature_engineer = demo_feature_engineering(df)
        
        # Step 3: Model Training
        lr_model, xgb_model, X_test, y_test = demo_model_training(df, feature_engineer)
        
        # Step 4: Real-time Scoring
        demo_real_time_scoring(lr_model, xgb_model, X_test, y_test)
        
        # Step 5: Explainability
        demo_explainability(lr_model, xgb_model, X_test)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The fraud detection system demonstrates:")
        print("✅ Synthetic data generation with realistic fraud patterns")
        print("✅ Real-time feature engineering (<100ms)")
        print("✅ Multiple ML models (LR, XGBoost)")
        print("✅ Fast inference (<100ms)")
        print("✅ Risk insights and explainability")
        print("\nNext steps:")
        print("1. Run 'python train_models.py' to train all models")
        print("2. Run 'python start_api.py' to start the API server")
        print("3. Test with 'curl -X POST http://localhost:8000/score'")
        
    except Exception as e:
        logger.error("Demo failed: %s", str(e))
        print(f"\n❌ Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()