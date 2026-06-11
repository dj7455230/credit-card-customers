"""
FastAPI endpoints for churn prediction service
"""
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime
import uvicorn
import os
import sys

# Add utils to path
sys.path.append('utils')
from model_utils import model_predictor
from db_utils import db_manager
from report_utils import report_generator

app = FastAPI(
    title="Churn Prediction API",
    description="Advanced ML-powered customer churn prediction service",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class CustomerData(BaseModel):
    CLIENTNUM: int
    Customer_Age: int
    Dependent_count: int
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: float
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float
    NB1: float
    NB2: float
    Gender: str
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str

class PredictionResponse(BaseModel):
    customer_id: str
    predictions: Dict[str, Any]
    timestamp: str
    status: str

class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: List[Dict[str, Any]]
    summary_stats: Dict[str, Any]
    processing_time: float

class ModelPerformanceResponse(BaseModel):
    models: Dict[str, Dict[str, float]]
    timestamp: str

# Helper functions
def authenticate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple token authentication (in production, use proper JWT)"""
    token = credentials.credentials
    # For demo purposes, accept any token that starts with 'demo_'
    if not token.startswith('demo_'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

def prepare_model_input(customer_data: CustomerData):
    """Convert customer data to model input format"""
    # Gender encoding
    gender_F = 1 if customer_data.Gender == "Female" else 0
    gender_M = 1 - gender_F
    
    # Education encoding
    edu_cols = ["College","Doctorate","Graduate","High School","Post-Graduate","Uneducated","Unknown"]
    edu_vec = [1 if e == customer_data.Education_Level else 0 for e in edu_cols]
    
    # Marital status encoding
    mar_cols = ["Divorced","Married","Single","Unknown"]
    mar_vec = [1 if m == customer_data.Marital_Status else 0 for m in mar_cols]
    
    # Income encoding
    inc_cols = ["$120K +","$40K - $60K","$60K - $80K","$80K - $120K","Less than $40K","Unknown"]
    inc_vec = [1 if i == customer_data.Income_Category else 0 for i in inc_cols]
    
    # Card encoding
    card_cols = ["Blue","Gold","Platinum","Silver"]
    card_vec = [1 if c == customer_data.Card_Category else 0 for c in card_cols]
    
    # Combine all features
    features = np.array([[
        customer_data.CLIENTNUM,
        customer_data.Customer_Age,
        customer_data.Dependent_count,
        customer_data.Months_on_book,
        customer_data.Total_Relationship_Count,
        customer_data.Months_Inactive_12_mon,
        customer_data.Contacts_Count_12_mon,
        customer_data.Credit_Limit,
        customer_data.Total_Revolving_Bal,
        customer_data.Avg_Open_To_Buy,
        customer_data.Total_Amt_Chng_Q4_Q1,
        customer_data.Total_Trans_Amt,
        customer_data.Total_Trans_Ct,
        customer_data.Total_Ct_Chng_Q4_Q1,
        customer_data.Avg_Utilization_Ratio,
        customer_data.NB1,
        customer_data.NB2,
        gender_F, gender_M,
        *edu_vec, *mar_vec, *inc_vec, *card_vec
    ]])
    
    return features

# API Endpoints
@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Churn Prediction API v2.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_models": list(model_predictor.models.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(
    customer_data: CustomerData,
    model_name: Optional[str] = "Naive Bayes",
    token: str = Depends(authenticate_token)
):
    """Predict churn for a single customer"""
    try:
        # Prepare input
        X = prepare_model_input(customer_data)
        
        # Get predictions from all models
        all_predictions = model_predictor.predict_all_models(X)
        
        # Save to database
        customer_id = str(customer_data.CLIENTNUM)
        if model_name in all_predictions:
            pred_data = all_predictions[model_name]
            db_manager.save_prediction(
                customer_id=customer_id,
                prediction=pred_data['prediction'],
                probability=pred_data['probability'],
                confidence=pred_data['confidence'],
                model_used=model_name,
                input_features=customer_data.dict()
            )
        
        return PredictionResponse(
            customer_id=customer_id,
            predictions=all_predictions,
            timestamp=datetime.now().isoformat(),
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    file: UploadFile = File(...),
    token: str = Depends(authenticate_token)
):
    """Batch prediction from CSV file"""
    try:
        import time
        start_time = time.time()
        
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Validate columns
        required_cols = model_predictor.feature_names
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing columns: {missing_cols[:5]}..."
            )
        
        # Make predictions
        X = df[required_cols].values
        predictions = []
        
        for i, row in enumerate(X):
            row_reshaped = row.reshape(1, -1)
            pred_results = model_predictor.predict_all_models(row_reshaped)
            
            # Use best performing model (Naive Bayes as default)
            best_pred = pred_results.get('Naive Bayes', pred_results[list(pred_results.keys())[0]])
            
            predictions.append({
                'row_index': i,
                'customer_id': df.iloc[i].get('CLIENTNUM', f'customer_{i}'),
                'prediction': best_pred['prediction'],
                'probability': best_pred['probability'],
                'confidence': best_pred['confidence']
            })
        
        # Calculate summary stats
        total_processed = len(predictions)
        churn_count = sum(1 for p in predictions if p['prediction'] == 1)
        avg_probability = sum(p['probability'] for p in predictions) / total_processed
        high_risk_count = sum(1 for p in predictions if p['probability'] > 0.7)
        
        summary_stats = {
            'total_customers': total_processed,
            'predicted_churn': churn_count,
            'churn_rate': churn_count / total_processed,
            'average_risk': avg_probability,
            'high_risk_customers': high_risk_count
        }
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            total_processed=total_processed,
            predictions=predictions,
            summary_stats=summary_stats,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(token: str = Depends(authenticate_token)):
    """Get performance metrics for all models"""
    try:
        performance = model_predictor.get_model_performance()
        
        return ModelPerformanceResponse(
            models=performance,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance retrieval failed: {str(e)}")

@app.get("/analytics/stats")
async def get_analytics_stats(token: str = Depends(authenticate_token)):
    """Get prediction analytics and statistics"""
    try:
        stats = db_manager.get_prediction_stats()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")

@app.post("/reports/generate")
async def generate_report(
    customer_ids: List[str],
    include_performance: bool = True,
    token: str = Depends(authenticate_token)
):
    """Generate PDF report for specified customers"""
    try:
        # Get predictions for specified customers
        predictions_df = db_manager.get_predictions_history(1000)
        filtered_predictions = predictions_df[predictions_df['customer_id'].isin(customer_ids)]
        
        # Get model performance if requested
        performance = model_predictor.get_model_performance() if include_performance else {}
        
        # Convert to format expected by report generator
        predictions_dict = {}
        for _, row in filtered_predictions.iterrows():
            predictions_dict[row['customer_id']] = {
                'prediction': row['prediction'],
                'probability': row['probability'],
                'confidence': row['confidence']
            }
        
        # Generate report
        report_path = report_generator.generate_prediction_report(
            customer_data=customer_ids,
            predictions=predictions_dict,
            model_performance=performance
        )
        
        return {
            "status": "success",
            "report_path": report_path,
            "customers_analyzed": len(customer_ids),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.post("/alerts/high-risk")
async def send_high_risk_alert(
    recipient_email: str,
    threshold: float = 0.7,
    token: str = Depends(authenticate_token)
):
    """Send email alert for high-risk customers"""
    try:
        # Get recent high-risk predictions
        predictions_df = db_manager.get_predictions_history(100)
        high_risk = predictions_df[predictions_df['probability'] > threshold]
        
        if high_risk.empty:
            return {
                "status": "success",
                "message": "No high-risk customers found",
                "threshold": threshold
            }
        
        # Convert to format for email
        high_risk_customers = {}
        for _, row in high_risk.iterrows():
            high_risk_customers[row['customer_id']] = {
                'probability': row['probability'],
                'prediction': row['prediction']
            }
        
        # Send alert
        success = report_generator.generate_high_risk_alert(
            high_risk_customers, 
            recipient_email
        )
        
        return {
            "status": "success" if success else "failed",
            "high_risk_count": len(high_risk_customers),
            "threshold": threshold,
            "recipient": recipient_email
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert sending failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)