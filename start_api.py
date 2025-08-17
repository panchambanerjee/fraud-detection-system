#!/usr/bin/env python3
"""
Startup script for the fraud detection API.

This script initializes the system and starts the FastAPI server.
"""

import uvicorn
import logging
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the fraud detection API server."""
    logger.info("Starting Fraud Detection API...")
    
    # Check if models are trained
    models_dir = Path("models/trained")
    if not models_dir.exists():
        logger.warning("No trained models found. Please run train_models.py first.")
        logger.info("You can still start the API, but it will need models to be trained.")
    
    # Start the server
    uvicorn.run(
        "api.fraud_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()