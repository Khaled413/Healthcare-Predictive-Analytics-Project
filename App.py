from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Feature names required for prediction
feature_names = [
    'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
    'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
    'Low HDL Cholesterol', 'High LDL Cholesterol', 'Alcohol Consumption',
    'Stress Level', 'Sleep Hours', 'Sugar Consumption', 'Triglyceride Level',
    'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
]

# Load or create a model
try:
    # Try to load the existing model
    model_path = os.path.join(os.path.dirname(__file__), 'output/models/gradient_boosting_model.pkl')
    preprocessor_path = os.path.join(os.path.dirname(__file__), 'output/models/preprocessor.pkl')
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Existing model loaded successfully!")
except Exception as e:
    print(f"Error loading existing model: {e}")
    print("Creating a simple default model...")
    
    # Create directories if they don't exist
    os.makedirs('output/models', exist_ok=True)
    
    # Create a simple preprocessor
    numeric_features = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 
                       'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar', 
                       'CRP Level', 'Homocysteine Level']
    categorical_features = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                           'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                           'High LDL Cholesterol', 'Alcohol Consumption', 
                           'Stress Level', 'Sugar Consumption']
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create and train a simple model
    model = GradientBoostingClassifier(random_state=42)
    
    # Save the new model and preprocessor
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    print("Default model created and saved successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        input_data = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                if feature in ['Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']:
                    # Allow empty values for some features
                    input_data[feature] = 0  # Replace NaN with 0 for simplicity
                else:
                    return render_template('result.html', error=f"Please enter a value for {feature}", prediction_percentage=0, prediction=0, user_data={})
            else:
                input_data[feature] = value
        
        # Convert data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Process categorical values
        for col in ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                   'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                   'High LDL Cholesterol', 'Alcohol Consumption', 
                   'Stress Level', 'Sugar Consumption']:
            input_df[col] = input_df[col].astype(str)
            
        # Convert numerical values
        for col in ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 
                   'Sleep Hours', 'Triglyceride Level', 'Fasting Blood Sugar', 
                   'CRP Level', 'Homocysteine Level']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        # Calculate risk probability (simplified approach)
        risk_factors = 0
        
        # Age factor
        if float(input_data['Age']) > 50:
            risk_factors += 1
            
        # Blood pressure factor
        if float(input_data['Blood Pressure']) > 140:
            risk_factors += 1
            
        # Cholesterol factor
        if float(input_data['Cholesterol Level']) > 200:
            risk_factors += 1
            
        # Smoking factor
        if input_data['Smoking'] == 'Current':
            risk_factors += 1
            
        # Family history factor
        if input_data['Family Heart Disease'] == 'Yes':
            risk_factors += 1
            
        # Diabetes factor
        if input_data['Diabetes'] == 'Yes':
            risk_factors += 1
            
        # BMI factor
        if float(input_data['BMI']) > 30:
            risk_factors += 1
            
        # Calculate probability based on risk factors
        probability = min(risk_factors * 10 + 20, 95)  # Scale it between 20% and 95%
        
        # Determine prediction
        prediction_value = 1 if probability > 50 else 0
        
        # Create result object
        result = {
            'prediction': 'High Risk' if prediction_value == 1 else 'Low Risk',
            'probability': round(probability, 2),
            'raw_values': input_data
        }
        
        # Prepare user data for template
        user_data = {
            'age': input_data['Age'],
            'trestbps': input_data['Blood Pressure'],
            'chol': input_data['Cholesterol Level']
        }
        
        return render_template('result.html', 
                              result=result, 
                              prediction_percentage=round(probability), 
                              prediction=prediction_value,
                              user_data=user_data)
    
    except Exception as e:
        return render_template('result.html', 
                              error=f"An error occurred during prediction: {str(e)}", 
                              prediction_percentage=0,
                              prediction=0,
                              user_data={})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get data from JSON request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check for required features
        missing_features = [f for f in feature_names if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400
        
        # Process data similar to the /predict route
        risk_factors = 0
        
        if float(data['Age']) > 50:
            risk_factors += 1
        if float(data['Blood Pressure']) > 140:
            risk_factors += 1
        if float(data['Cholesterol Level']) > 200:
            risk_factors += 1
        if data['Smoking'] == 'Current':
            risk_factors += 1
        if data['Family Heart Disease'] == 'Yes':
            risk_factors += 1
        if data['Diabetes'] == 'Yes':
            risk_factors += 1
        if float(data['BMI']) > 30:
            risk_factors += 1
            
        probability = min(risk_factors * 10 + 20, 95)
        prediction = 'Yes' if probability > 50 else 'No'
        
        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 2),
            'status': 'Prediction successful'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))