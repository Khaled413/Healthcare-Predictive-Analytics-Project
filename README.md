# Heart Disease Prediction System

## Project Overview
This project provides a web-based system to predict the risk of heart disease based on patient health data. Using advanced machine learning techniques, the system analyzes various health metrics to provide a risk assessment for heart disease.

## Features
- User-friendly web interface for data entry
- Comprehensive analysis of 20 health factors
- Instant risk prediction with probability score
- Detailed results visualization
- Responsive design for all devices

## Technical Details
- Flask-based web application
- Gradient Boosting machine learning model
- Data preprocessing and feature engineering
- Interactive visualization of results

## Project Structure
- `App.py`: Main Flask application
- `templates/`: HTML templates for the web interface
- `output/models/`: Trained machine learning models
- `visualizations/`: Data visualization components
- `results/`: Prediction results and analytics
- `src/`: Source code for data processing and model training

## Health Factors Analyzed
- Demographic factors (age, gender)
- Medical indicators (blood pressure, cholesterol levels)
- Lifestyle factors (exercise habits, smoking, alcohol consumption)
- Medical history (family history, existing conditions)
- Health metrics (BMI, sleep patterns, stress levels)

## Setup Instructions
1. Clone this repository
2. Create a virtual environment: `python -m venv healthcare_env`
3. Activate the environment:
   - Windows: `healthcare_env\Scripts\activate`
   - Mac/Linux: `source healthcare_env/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python App.py`
6. Access the web interface at `http://localhost:5000`

## Important Note
This system is designed for educational and informational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment.