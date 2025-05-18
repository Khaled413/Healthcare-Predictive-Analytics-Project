from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Define the FastAPI app
app = FastAPI(title="Healthcare Predictive Model API",
            description="API for healthcare prediction model")

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None


class PatientData(BaseModel):
    """Data model for patient information"""
    # Define the fields based on your dataset
    # Example fields (modify according to your dataset):
    age: int
    gender: str
    blood_pressure: float
    cholesterol_level: float
    exercise_habits: str
    smoking: str
    family_heart_disease: str
    diabetes: str
    bmi: float
    # Add other fields as needed


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    probability: float
    risk_factors: list


def load_model_and_preprocessor(model_path, preprocessor_path):
    """Load the model and preprocessor"""
    global model, preprocessor, feature_names
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Try to load feature names if available
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Healthcare Predictive Model API"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PatientData):
    """Make a prediction based on patient data"""
    global model, preprocessor
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")
    
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    try:
        # Preprocess the input data
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0, 1]
        prediction = "Yes" if prediction_proba >= 0.5 else "No"
        
        # Get risk factors (if feature importances are available)
        risk_factors = []
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            # Get feature importances for this specific prediction
            importances = model.feature_importances_
            
            # Get top 3 risk factors
            top_indices = np.argsort(importances)[-3:]
            risk_factors = [feature_names[i] for i in top_indices]
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(prediction_proba),
            risk_factors=risk_factors
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def start_api(model_path, preprocessor_path, host="0.0.0.0", port=8000):
    """Start the FastAPI server"""
    import uvicorn
    
    # Load model and preprocessor
    load_model_and_preprocessor(model_path, preprocessor_path)
    
    # Start the server
    uvicorn.run(app, host=host, port=port)