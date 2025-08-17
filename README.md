# Fraud Detection System - Stripe Radar Case Study

A production-ready fraud detection system inspired by Stripe Radar, demonstrating real-time ML inference, advanced feature engineering, and model evolution from simple algorithms to deep neural networks.

## 🎯 Project Overview

This system replicates the core concepts of Stripe Radar's fraud prevention solution:
- **Real-time inference** (<100ms response time)
- **Advanced feature engineering** (1000+ characteristics)
- **Model evolution** (Logistic Regression → XGBoost → Neural Networks)
- **Explainability** (SHAP values, risk insights)
- **Production API** (FastAPI with monitoring)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Transaction   │───▶│ Feature Pipeline │───▶│ ML Models      │
│   Input        │    │ (1000+ features) │    │ (LR/XGB/DNN)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Feature Store    │    │ Risk Scoring    │
                       │ (Redis Cache)    │    │ & Explainability│
                       └──────────────────┘    └─────────────────┘
```

## 🚀 Key Features

### 1. **Synthetic Data Generation**
- Realistic transaction data with embedded fraud patterns
- Card testing attacks (high velocity, small amounts)
- Stolen card fraud (unusual locations, high amounts)
- Account takeover patterns
- Diverse user spending profiles

### 2. **Feature Engineering**
- **Velocity features**: Transaction frequency in time windows
- **Behavioral features**: Amount Z-scores, spending anomalies
- **Network features**: IP reputation, email domain analysis
- **Geographic features**: Distance calculations, country risk
- **Temporal features**: Time-based patterns, rapid transactions

### 3. **Model Implementation**
- **LogisticRegressionModel**: Simple baseline with L1 regularization
- **XGBoostModel**: Tree-based model with early stopping
- **DeepNeuralNetworkModel**: Multi-branch architecture (ResNeXt-inspired)

### 4. **Production API**
- FastAPI implementation with real-time scoring
- Feature extraction optimized for speed
- Model A/B testing capabilities
- Performance monitoring and metrics

### 5. **Explainability**
- SHAP integration for all model types
- Human-readable risk insights
- Feature contribution analysis
- Risk factor templates

## 📊 Performance Targets

- **ROC-AUC**: 0.85+ (LR), 0.90+ (XGB), 0.92+ (DNN)
- **Inference time**: <100ms for all models
- **False positive rate**: <1%
- **Fraud detection rate**: 85%+ at 1% FPR

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If you have M1/M2 Mac, use this instead:
pip install tensorflow-macos tensorflow-metal

# Install development dependencies
pip install -r requirements-dev.txt
```

## 🚀 Quick Start

### 1. Generate Synthetic Data
```bash
python -m data.generate_synthetic_data
```

### 2. Train Models
```bash
python -m models.train_all_models
```

### 3. Start API Server
```bash
uvicorn api.fraud_api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test API
```bash
curl -X POST "http://localhost:8000/score" \
     -H "Content-Type: application/json" \
     -d '{"amount": 100.00, "user_id": "user_123", "merchant_id": "merchant_456"}'
```

## 📁 Project Structure

```
fraud-detection-system/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                    # Data generation and management
│   ├── __init__.py
│   └── generate_synthetic_data.py
├── features/                # Feature engineering pipeline
│   ├── __init__.py
│   ├── feature_store.py
│   └── feature_engineering.py
├── models/                  # ML model implementations
│   ├── __init__.py
│   ├── logistic_model.py
│   ├── xgboost_model.py
│   ├── neural_network.py
│   └── model_comparison.py
├── api/                     # FastAPI application
│   ├── __init__.py
│   └── fraud_api.py
├── explainability/          # Model explainability tools
│   ├── __init__.py
│   └── model_explainer.py
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_models.py
└── notebooks/               # Jupyter notebooks
    ├── 01_data_exploration.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_model_training.ipynb
    └── 04_model_evaluation.ipynb
```

## 🔧 Configuration

The system uses environment variables for configuration:

```bash
# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Model configuration
MODEL_CACHE_DIR=./models/cache
FEATURE_CACHE_TTL=3600

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 📈 API Endpoints

### Core Endpoints
- `POST /score` - Single transaction scoring
- `POST /batch_score` - Multiple transactions
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /metrics` - Performance metrics
- `POST /feedback` - Model improvement feedback

### Model Management
- `POST /models/train` - Train new model
- `GET /models/{model_id}/performance` - Model performance
- `POST /models/{model_id}/deploy` - Deploy model

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_models.py

# Run performance tests
pytest tests/test_performance.py
```

## 📊 Monitoring

The system includes comprehensive monitoring:
- **Performance metrics**: Response time, throughput, error rates
- **Model metrics**: Accuracy, precision, recall, F1-score
- **Business metrics**: Fraud detection rate, false positive rate
- **System metrics**: CPU, memory, disk usage

## 🔍 Explainability

### Risk Insights
- Feature contribution analysis
- Human-readable risk factors
- Business context explanations
- Historical comparison

### SHAP Integration
- Feature importance ranking
- Individual prediction explanations
- Waterfall plots
- Force plots

## 🚀 Deployment

### Docker
```bash
docker build -t fraud-detection-system .
docker run -p 8000:8000 fraud-detection-system
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📚 Learning Resources

- [Stripe Radar Blog Post](https://stripe.com/blog/how-we-built-it-stripe-radar)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📄 License

This project is for educational purposes. Please refer to the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by Stripe Radar's fraud prevention system
- Built for educational and technical blog content
- Demonstrates production-ready ML system design