import os
import pandas as pd  # pyright: ignore [reportMissingImports]
from fastapi import FastAPI, HTTPException  # pyright: ignore [reportMissingImports]
from pydantic import BaseModel, Field  # pyright: ignore [reportMissingImports]
from typing import List
from loguru import logger  # pyright: ignore [reportMissingImports]
from src.utils import setup_logging, load_env_variables
from src.model_pipeline import load_pipeline

# Initialize configurations
setup_logging("INFO")
load_env_variables()

app = FastAPI(
    title="Real-Time Fraud Detection System API",
    description="REST API for predicting credit card transaction fraud probability and classification.",
    version="1.0.0"
)

# Global variables
model_pipeline = None

class TransactionInput(BaseModel):
    Time: float = Field(..., example=86400.0, description="Seconds elapsed since first transaction")
    V1: float = Field(..., example=-1.359807)
    V2: float = Field(..., example=-0.072781)
    V3: float = Field(..., example=2.536347)
    V4: float = Field(..., example=1.378155)
    V5: float = Field(..., example=-0.338321)
    V6: float = Field(..., example=0.462388)
    V7: float = Field(..., example=0.239599)
    V8: float = Field(..., example=0.098698)
    V9: float = Field(..., example=0.363787)
    V10: float = Field(..., example=0.090794)
    V11: float = Field(..., example=-0.551600)
    V12: float = Field(..., example=-0.617801)
    V13: float = Field(..., example=-0.991390)
    V14: float = Field(..., example=-0.311169)
    V15: float = Field(..., example=1.468177)
    V16: float = Field(..., example=-0.470401)
    V17: float = Field(..., example=0.207971)
    V18: float = Field(..., example=0.025791)
    V19: float = Field(..., example=0.403993)
    V20: float = Field(..., example=0.251412)
    V21: float = Field(..., example=-0.018307)
    V22: float = Field(..., example=0.277838)
    V23: float = Field(..., example=-0.110474)
    V24: float = Field(..., example=0.066928)
    V25: float = Field(..., example=0.128539)
    V26: float = Field(..., example=-0.189115)
    V27: float = Field(..., example=0.133558)
    V28: float = Field(..., example=-0.021053)
    Amount: float = Field(..., example=149.62, description="Transaction dollar amount")

class PredictResponse(BaseModel):
    is_fraud: bool
    probability: float
    model_version: str

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

@app.on_event("startup")
def startup_event():
    global model_pipeline
    model_path = os.getenv("MODEL_PATH", "models/fraud_model_pipeline.joblib")
    logger.info(f"Loading serialized model pipeline from {model_path} during startup...")
    try:
        model_pipeline = load_pipeline(model_path)
        logger.info("Model pipeline loaded successfully. FastAPI is ready to receive requests.")
    except Exception as e:
        logger.error(f"Failed to load model pipeline at startup: {e}")
        raise RuntimeError(f"Startup failed: could not load model pipeline.")

@app.get("/health")
def health_check():
    """Health check endpoint to ensure server and model pipeline are functioning."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": os.getenv("MODEL_VERSION", "1.0.0")
    }

@app.post("/predict", response_model=PredictResponse)
def predict_single(transaction: TransactionInput):
    """Predicts fraud classification and probability for a single transaction."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    try:
        # Convert request to dict, supporting pydantic v1 & v2
        data_dict = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()
        
        # Convert request to DataFrame with correct column ordering
        columns_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        tx_data = pd.DataFrame([data_dict])[columns_order]
        
        # Make predictions
        prob = float(model_pipeline.predict_proba(tx_data)[0, 1])
        pred = bool(model_pipeline.predict(tx_data)[0])
        
        logger.info(f"Prediction: is_fraud={pred}, prob={prob:.6f}")
        return PredictResponse(
            is_fraud=pred,
            probability=prob,
            model_version=os.getenv("MODEL_VERSION", "1.0.0")
        )
    except Exception as e:
        logger.error(f"Error occurred during single prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(batch: List[TransactionInput]):
    """Predicts fraud classification and probability for a batch of transactions."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    if not batch:
        raise HTTPException(status_code=400, detail="Transaction list cannot be empty.")
        
    try:
        # Convert list of inputs to DataFrame
        columns_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        data_list = [tx.model_dump() if hasattr(tx, "model_dump") else tx.dict() for tx in batch]
        tx_data = pd.DataFrame(data_list)[columns_order]
        
        # Batch inference
        probs = model_pipeline.predict_proba(tx_data)[:, 1]
        preds = model_pipeline.predict(tx_data)
        
        responses = []
        for i in range(len(batch)):
            responses.append(PredictResponse(
                is_fraud=bool(preds[i]),
                probability=float(probs[i]),
                model_version=os.getenv("MODEL_VERSION", "1.0.0")
            ))
            
        logger.info(f"Processed batch prediction of size {len(batch)}")
        return BatchPredictResponse(predictions=responses)
    except Exception as e:
        logger.error(f"Error occurred during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")
