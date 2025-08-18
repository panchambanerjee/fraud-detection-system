# Stripe Fraud Detection System - Setup Instructions

This document provides step-by-step instructions for setting up and running the Stripe Fraud Detection System case study.

## System Requirements

- Python 3.8+
- 8GB+ RAM (for model training)
- macOS, Linux, or Windows

## Setup Instructions

### 1. Create a Virtual Environment

```bash
# Navigate to the project directory
cd ml_case_studies/case_study_stripe/fraud-detection-system

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install the required packages
pip install -r requirements.txt

# If you have an M1/M2 Mac and want to use TensorFlow:
# pip install tensorflow-macos tensorflow-metal
```

### 3. Fix TensorFlow/Keras Compatibility (Optional)

The project has a version mismatch between TensorFlow and Keras. You have two options:

#### Option A: Run without TensorFlow features (Recommended for quick start)
- The project has been modified to work without TensorFlow
- Deep Neural Network model and explainability features are disabled
- You can still run the demo with Logistic Regression and XGBoost models

#### Option B: Fix the dependency issues
```bash
# Ensure Keras version matches TensorFlow version
pip install keras==2.15.0

# Or downgrade TensorFlow to match Keras
# pip install tensorflow==2.15.0 tensorflow-io-gcs-filesystem==0.34.0
```

## Running the Project

### 1. Run the Demo

```bash
# Run the simplified demo (no TensorFlow required)
python demo.py
```

This will:
- Generate synthetic transaction data
- Engineer features
- Train Logistic Regression and XGBoost models
- Test real-time scoring

### 2. Train Models (Optional)

```bash
# If you fixed the TensorFlow dependencies:
python train_models.py
```

### 3. Start the API Server

```bash
# Start the FastAPI server
python start_api.py
```

### 4. Test the API

```bash
# In a new terminal window
curl -X POST "http://localhost:8000/score" \
     -H "Content-Type: application/json" \
     -d '{"amount": 100.00, "user_id": "user_123", "merchant_id": "merchant_456"}'
```

## Troubleshooting

### Common Issues

1. **TensorFlow/Keras Version Mismatch**
   - Error: `AttributeError: module 'tensorflow._api.v2.compat.v2.__internal__' has no attribute 'register_load_context_function'`
   - Solution: Follow the instructions in section 3 above to fix compatibility issues

2. **Feature Engineering Errors**
   - Error: `Error computing features: 'fraud_rate'` or similar
   - Solution: The demo automatically falls back to creating basic features when the advanced feature engineering fails. This is expected behavior.

3. **Empty Feature Matrix**
   - Error: `at least one array or dtype is required`
   - Solution: The demo automatically creates basic features when the advanced feature engineering fails. This is expected behavior.

## Project Structure

- `data/` - Data generation module
- `features/` - Feature engineering pipeline
- `models/` - ML model implementations
- `api/` - FastAPI application
- `explainability/` - Model explainability tools
- `tests/` - Test suite

## Next Steps

For a complete experience, you would need to:

1. Fix the feature engineering module
2. Update the dependencies to resolve version conflicts
3. Implement the missing components

This case study is primarily for educational purposes to demonstrate the architecture of a fraud detection system inspired by Stripe Radar.
