"""
Configuration file for the fraud detection system.

This module contains all configuration parameters for the system,
including model hyperparameters, API settings, and feature engineering
parameters.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Data generation configuration
DATA_CONFIG = {
    'default_seed': 42,
    'default_days': 30,
    'default_transactions_per_day': 10000,
    'default_fraud_rate': 0.001,  # 0.1%
    'user_count': 1000,
    'countries': ['US', 'UK', 'CA', 'DE', 'FR'],
    'merchant_categories': [
        'electronics', 'clothing', 'food_delivery', 'travel',
        'gaming', 'subscription', 'gift_cards', 'digital_goods',
        'jewelry', 'luxury_goods', 'pharmacy', 'automotive',
        'home_goods', 'sports', 'books', 'music', 'movies'
    ]
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'velocity_windows': [1, 24, 168],  # 1h, 24h, 7d
    'max_history_days': 30,
    'max_transaction_history': 100,
    'max_location_history': 20,
    'max_timing_history': 50,
    'distance_threshold_km': 100,
    'rapid_transaction_threshold_hours': 0.1
}

# Model configuration
MODEL_CONFIG = {
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l1',
        'solver': 'liblinear',
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced'
    },
    'xgboost': {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'early_stopping_rounds': 50
    },
    'neural_network': {
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
        'class_weight': {0: 1.0, 1: 10.0}
    }
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': True,
    'reload': True,
    'workers': 1,
    'log_level': 'info',
    'cors_origins': ['*'],
    'max_request_size': 10 * 1024 * 1024,  # 10MB
    'request_timeout': 30,  # seconds
    'max_concurrent_requests': 100
}

# Performance targets
PERFORMANCE_TARGETS = {
    'inference_time_ms': 100,
    'feature_computation_ms': 50,
    'api_response_ms': 100,
    'roc_auc_targets': {
        'logistic_regression': 0.85,
        'xgboost': 0.90,
        'neural_network': 0.92
    },
    'false_positive_rate': 0.01,  # 1%
    'fraud_detection_rate': 0.85  # 85% at 1% FPR
}

# Feature store configuration
FEATURE_STORE_CONFIG = {
    'primary_type': 'memory',  # 'redis' or 'memory'
    'fallback_type': 'memory',
    'redis_host': os.getenv('REDIS_HOST', 'localhost'),
    'redis_port': int(os.getenv('REDIS_PORT', 6379)),
    'redis_db': int(os.getenv('REDIS_DB', 0)),
    'redis_password': os.getenv('REDIS_PASSWORD'),
    'cache_ttl': 3600,  # 1 hour
    'max_cache_size': 10000
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'fraud_detection.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Monitoring configuration
MONITORING_CONFIG = {
    'metrics_enabled': True,
    'health_check_interval': 30,  # seconds
    'performance_metrics_interval': 60,  # seconds
    'model_performance_tracking': True,
    'cache_performance_tracking': True,
    'api_performance_tracking': True
}

# Security configuration
SECURITY_CONFIG = {
    'rate_limiting_enabled': True,
    'max_requests_per_minute': 1000,
    'api_key_required': False,
    'cors_enabled': True,
    'input_validation': True,
    'sql_injection_protection': True
}

# Development configuration
DEV_CONFIG = {
    'auto_reload': True,
    'debug_mode': True,
    'profiling_enabled': False,
    'test_data_generation': True,
    'mock_external_services': True
}

# Production configuration
PRODUCTION_CONFIG = {
    'auto_reload': False,
    'debug_mode': False,
    'profiling_enabled': False,
    'test_data_generation': False,
    'mock_external_services': False,
    'logging_level': 'WARNING',
    'cors_origins': ['https://yourdomain.com'],  # Restrict CORS
    'rate_limiting_enabled': True,
    'api_key_required': True
}

def get_config(environment: str = 'development') -> dict:
    """
    Get configuration for a specific environment.
    
    Args:
        environment: Environment name ('development' or 'production')
        
    Returns:
        Configuration dictionary
    """
    base_config = {
        'data': DATA_CONFIG,
        'features': FEATURE_CONFIG,
        'models': MODEL_CONFIG,
        'api': API_CONFIG,
        'performance': PERFORMANCE_TARGETS,
        'feature_store': FEATURE_STORE_CONFIG,
        'logging': LOGGING_CONFIG,
        'monitoring': MONITORING_CONFIG,
        'security': SECURITY_CONFIG
    }
    
    if environment == 'production':
        base_config.update(PRODUCTION_CONFIG)
    else:
        base_config.update(DEV_CONFIG)
    
    return base_config

def get_model_config(model_name: str) -> dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
    """
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")
    
    return MODEL_CONFIG[model_name]

def get_feature_config() -> dict:
    """Get feature engineering configuration."""
    return FEATURE_CONFIG

def get_api_config() -> dict:
    """Get API configuration."""
    return API_CONFIG

def get_performance_targets() -> dict:
    """Get performance targets."""
    return PERFORMANCE_TARGETS