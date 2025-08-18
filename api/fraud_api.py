"""
FastAPI application for fraud detection system.

This module provides production-ready API endpoints for real-time fraud scoring,
model management, and system monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_engineering import FeatureEngineer
from features.feature_store import FeatureStoreManager, create_feature_store_manager
from models.logistic_model import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
# from models.neural_network import DeepNeuralNetworkModel  # Disabled for now
# from explainability.model_explainer import ModelExplainer  # Disabled for now

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models and feature engineering
feature_engineer = None
feature_store = None
models = {}
model_explainer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting fraud detection API...")
    
    # Initialize components
    await initialize_components()
    
    yield
    
    # Shutdown
    logger.info("Shutting down fraud detection API...")
    if feature_store:
        feature_store.close()

async def initialize_components():
    """Initialize all system components."""
    global feature_engineer, feature_store, models, model_explainer
    
    try:
        # Initialize feature engineering
        feature_engineer = FeatureEngineer()
        logger.info("Feature engineering pipeline initialized")
        
        # Initialize feature store
        feature_store = create_feature_store_manager(
            primary_type='memory',  # Use in-memory for demo
            fallback_type='memory'
        )
        logger.info("Feature store initialized")
        
        # Initialize models
        models = {
            'logistic_regression': LogisticRegressionModel(),
            'xgboost': XGBoostModel(),
            # 'neural_network': DeepNeuralNetworkModel()  # Disabled for now
        }
        logger.info("Models initialized: %s", list(models.keys()))
        
        # Initialize model explainer (disabled for now)
        # model_explainer = ModelExplainer()
        # logger.info("Model explainer initialized")
        
        # Load pre-trained models if available
        await load_pretrained_models()
        
    except Exception as e:
        logger.error("Failed to initialize components: %s", str(e))
        raise

async def load_pretrained_models():
    """Load pre-trained models from disk if available."""
    model_dir = Path("models/trained")
    
    if not model_dir.exists():
        logger.info("No pre-trained models found. Models will need training.")
        return
    
    for model_name, model_instance in models.items():
        model_path = model_dir / f"{model_name}_model.pkl"
        if model_path.exists():
            try:
                model_instance.load_model(str(model_path))
                logger.info("Loaded pre-trained model: %s", model_name)
            except Exception as e:
                logger.warning("Failed to load model %s: %s", model_name, str(e))

# Pydantic models for API requests/responses
class TransactionRequest(BaseModel):
    """Request model for single transaction scoring."""
    user_id: str = Field(..., description="Unique user identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category")
    ip_address: str = Field(..., description="IP address")
    email: str = Field(..., description="User email")
    card_last4: str = Field(..., description="Last 4 digits of card")
    card_brand: str = Field(..., description="Card brand (visa, mastercard, etc.)")
    country: str = Field(..., description="Transaction country")
    city: str = Field(..., description="Transaction city")
    latitude: float = Field(..., description="Transaction latitude")
    longitude: float = Field(..., description="Transaction longitude")
    device_type: str = Field(..., description="Device type (desktop, mobile, tablet)")
    browser: str = Field(..., description="Browser type")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp (ISO format)")

class BatchTransactionRequest(BaseModel):
    """Request model for batch transaction scoring."""
    transactions: List[TransactionRequest] = Field(..., description="List of transactions")
    model_name: Optional[str] = Field(None, description="Specific model to use")

class ScoringResponse(BaseModel):
    """Response model for fraud scoring."""
    transaction_id: str
    prediction: int = Field(..., description="0=legitimate, 1=fraud")
    probability: float = Field(..., description="Fraud probability (0-1)")
    risk_score: float = Field(..., description="Risk score (0-1)")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_name: str = Field(..., description="Model used for prediction")
    risk_insights: Optional[Dict[str, Any]] = Field(None, description="Risk insights and explanations")

class BatchScoringResponse(BaseModel):
    """Response model for batch fraud scoring."""
    results: List[ScoringResponse]
    total_transactions: int
    fraud_count: int
    fraud_rate: float
    processing_time_ms: float
    model_name: str

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    is_trained: bool
    feature_count: int
    last_training_date: Optional[str]
    model_size_mb: float
    performance_metrics: Optional[Dict[str, float]]

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    uptime_seconds: float
    models_status: Dict[str, str]
    feature_store_status: str
    system_metrics: Dict[str, Any]

class FeedbackRequest(BaseModel):
    """Model feedback request."""
    transaction_id: str
    actual_fraud: bool
    model_name: str
    feedback_notes: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Production-ready fraud detection system inspired by Stripe Radar",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# API endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "description": "Production-ready fraud detection system inspired by Stripe Radar",
        "endpoints": {
            "score": "/score - Single transaction scoring",
            "batch_score": "/batch_score - Batch transaction scoring",
            "health": "/health - System health check",
            "models": "/models - List available models",
            "metrics": "/metrics - Performance metrics"
        }
    }

@app.post("/score", response_model=ScoringResponse, tags=["Scoring"])
async def score_transaction(transaction: TransactionRequest):
    """
    Score a single transaction for fraud detection.
    
    This endpoint provides real-time fraud scoring with <100ms response time.
    """
    start_time = time.time()
    
    try:
        # Generate transaction ID
        transaction_id = f"txn_{int(time.time() * 1000)}"
        
        # Convert to dictionary for feature engineering
        transaction_dict = transaction.dict()
        transaction_dict['timestamp'] = transaction_dict.get('timestamp') or datetime.now().isoformat()
        
        # Check cache first
        cache_key = feature_store.generate_key(transaction_dict)
        cached_result = feature_store.get_features(cache_key)
        
        if cached_result:
            logger.info("Cache hit for transaction %s", transaction_id)
            return ScoringResponse(
                transaction_id=transaction_id,
                **cached_result
            )
        
        # Compute features
        feature_set = feature_engineer.compute_features(transaction_dict)
        feature_vector = feature_engineer.get_feature_vector(feature_set)
        
        # Score with best available model
        best_model_name = get_best_available_model()
        best_model = models[best_model_name]
        
        # Make prediction
        scoring_result = best_model.score_single(feature_vector)
        
        # Generate risk insights if explainer is available
        risk_insights = None
        if model_explainer:
            try:
                risk_insights = model_explainer.generate_risk_insights(
                    feature_vector, scoring_result, best_model_name
                )
            except Exception as e:
                logger.warning("Failed to generate risk insights: %s", str(e))
        
        # Create response
        response = ScoringResponse(
            transaction_id=transaction_id,
            prediction=scoring_result['prediction'],
            probability=scoring_result['probability'],
            risk_score=scoring_result['risk_score'],
            confidence=scoring_result['confidence'],
            inference_time_ms=scoring_result['inference_time_ms'],
            model_name=best_model_name,
            risk_insights=risk_insights
        )
        
        # Cache result
        feature_store.store_features(cache_key, response.dict(), ttl=3600)
        
        # Performance check
        total_time = (time.time() - start_time) * 1000
        if total_time > 100:
            logger.warning("Total API response time %.2fms exceeds 100ms target", total_time)
        
        logger.info("Scored transaction %s in %.2fms", transaction_id, total_time)
        return response
        
    except Exception as e:
        logger.error("Error scoring transaction: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/batch_score", response_model=BatchScoringResponse, tags=["Scoring"])
async def score_batch_transactions(batch_request: BatchTransactionRequest):
    """
    Score multiple transactions in batch.
    
    This endpoint provides efficient batch processing for multiple transactions.
    """
    start_time = time.time()
    
    try:
        transactions = batch_request.transactions
        model_name = batch_request.model_name or get_best_available_model()
        
        if model_name not in models:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
        model = models[model_name]
        
        if not model.is_trained:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")
        
        results = []
        fraud_count = 0
        
        # Process transactions
        for transaction in transactions:
            try:
                # Convert to dictionary
                transaction_dict = transaction.dict()
                transaction_dict['timestamp'] = transaction_dict.get('timestamp') or datetime.now().isoformat()
                
                # Generate transaction ID
                transaction_id = f"batch_{int(time.time() * 1000)}_{len(results)}"
                
                # Compute features
                feature_set = feature_engineer.compute_features(transaction_dict)
                feature_vector = feature_engineer.get_feature_vector(feature_set)
                
                # Score transaction
                scoring_result = model.score_single(feature_vector)
                
                # Create response
                response = ScoringResponse(
                    transaction_id=transaction_id,
                    prediction=scoring_result['prediction'],
                    probability=scoring_result['probability'],
                    risk_score=scoring_result['risk_score'],
                    confidence=scoring_result['confidence'],
                    inference_time_ms=scoring_result['inference_time_ms'],
                    model_name=model_name,
                    risk_insights=None  # Skip insights for batch processing
                )
                
                results.append(response)
                
                if scoring_result['prediction'] == 1:
                    fraud_count += 1
                    
            except Exception as e:
                logger.error("Error processing transaction in batch: %s", str(e))
                continue
        
        # Calculate batch statistics
        total_transactions = len(results)
        fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0
        processing_time = (time.time() - start_time) * 1000
        
        logger.info("Batch processed %d transactions in %.2fms", total_transactions, processing_time)
        
        return BatchScoringResponse(
            results=results,
            total_transactions=total_transactions,
            fraud_count=fraud_count,
            fraud_rate=fraud_rate,
            processing_time_ms=processing_time,
            model_name=model_name
        )
        
    except Exception as e:
        logger.error("Error in batch scoring: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Check system health and status."""
    try:
        # Check models status
        models_status = {}
        for name, model in models.items():
            models_status[name] = "trained" if model.is_trained else "not_trained"
        
        # Check feature store status
        feature_store_status = "healthy"
        try:
            # Simple test
            test_key = "health_check"
            feature_store.store_features(test_key, {"test": "data"}, ttl=60)
            test_data = feature_store.get_features(test_key)
            if test_data and test_data.get("test") == "data":
                feature_store_status = "healthy"
            else:
                feature_store_status = "degraded"
        except Exception:
            feature_store_status = "unhealthy"
        
        # System metrics
        system_metrics = {
            "memory_usage_mb": get_memory_usage(),
            "cpu_usage_percent": get_cpu_usage(),
            "cache_hit_rate": feature_store.get_cache_stats()['hit_rate_percent'] if feature_store else 0
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=time.time() - app.startup_time if hasattr(app, 'startup_time') else 0,
            models_status=models_status,
            feature_store_status=feature_store_status,
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error("Health check failed: %s", str(e))
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=0,
            models_status={},
            feature_store_status="unknown",
            system_metrics={}
        )

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """List all available models and their status."""
    try:
        model_info_list = []
        
        for name, model in models.items():
            info = model.get_model_info()
            
            # Add performance metrics if available
            performance_metrics = None
            if hasattr(model, 'last_evaluation_metrics'):
                performance_metrics = model.last_evaluation_metrics
            
            model_info = ModelInfo(
                model_name=info['model_name'],
                model_type=info['model_type'],
                is_trained=info['is_trained'],
                feature_count=info['feature_count'],
                last_training_date=info['last_training_date'],
                model_size_mb=info['model_size_mb'],
                performance_metrics=performance_metrics
            )
            
            model_info_list.append(model_info)
        
        return model_info_list
        
    except Exception as e:
        logger.error("Error listing models: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get system performance metrics."""
    try:
        # Feature store metrics
        cache_stats = feature_store.get_cache_stats() if feature_store else {}
        
        # Model metrics
        model_metrics = {}
        for name, model in models.items():
            if hasattr(model, 'last_evaluation_metrics'):
                model_metrics[name] = model.last_evaluation_metrics
        
        # API metrics
        api_metrics = {
            "total_requests": getattr(app.state, 'total_requests', 0),
            "successful_requests": getattr(app.state, 'successful_requests', 0),
            "failed_requests": getattr(app.state, 'failed_requests', 0),
            "average_response_time_ms": getattr(app.state, 'avg_response_time', 0)
        }
        
        return {
            "cache_stats": cache_stats,
            "model_metrics": model_metrics,
            "api_metrics": api_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Error getting metrics: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.post("/feedback", tags=["Models"])
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for model improvement."""
    try:
        # Store feedback for model retraining
        feedback_data = {
            'transaction_id': feedback.transaction_id,
            'actual_fraud': feedback.actual_fraud,
            'model_name': feedback.model_name,
            'feedback_notes': feedback.feedback_notes,
            'timestamp': datetime.now().isoformat()
        }
        
        # In a production system, this would be stored in a database
        logger.info("Feedback received: %s", feedback_data)
        
        return {
            "message": "Feedback received successfully",
            "feedback_id": f"fb_{int(time.time() * 1000)}",
            "timestamp": feedback_data['timestamp']
        }
        
    except Exception as e:
        logger.error("Error processing feedback: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process feedback: {str(e)}")

# Helper functions
def get_best_available_model() -> str:
    """Get the best available trained model."""
    trained_models = {name: model for name, model in models.items() if model.is_trained}
    
    if not trained_models:
        raise HTTPException(status_code=503, detail="No trained models available")
    
    # For now, return the first trained model
    # In production, this would use model performance metrics
    return list(trained_models.keys())[0]

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        return 0.0

def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception: %s", str(exc))
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    app.startup_time = time.time()
    logger.info("Fraud Detection API started")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Fraud Detection API shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)