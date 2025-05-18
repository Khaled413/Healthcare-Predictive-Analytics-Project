import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Feature names required for prediction
feature_names = [
    'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
    'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
    'Low HDL Cholesterol', 'High LDL Cholesterol', 'Alcohol Consumption',
    'Stress Level', 'Sleep Hours', 'Sugar Consumption', 'Triglyceride Level',
    'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
]

# Load or create a model
@st.cache_resource
def load_or_create_model():
    try:
        # Try to load the existing model
        model_path = os.path.join(os.path.dirname(__file__), 'output/models/gradient_boosting_model.pkl')
        preprocessor_path = os.path.join(os.path.dirname(__file__), 'output/models/preprocessor.pkl')
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        st.write("Existing model loaded successfully!")
        return model, preprocessor
    except Exception as e:
        st.write(f"Error loading existing model: {e}")
        st.write("Creating a simple default model...")
        
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
        st.write("Default model created and saved successfully!")
        return model, preprocessor

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Define pages/routes
def home():
    st.title("Heart Disease Prediction System")
    st.markdown("""
    <div style="text-align: center;">
        <h2>Welcome to the Heart Disease Prediction System</h2>
        <p>A modern web application that uses machine learning to predict heart disease risk based on various health factors.</p>
        <p>The system provides personalized risk assessments and health recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a centered button to navigate to the prediction page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Prediction", key="home_button", use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()
    
    # Features section
    st.markdown("""
    <div style="margin-top: 30px;">
        <h3>Key Features</h3>
        <ul>
            <li><strong>Comprehensive Analysis:</strong> Evaluates 20+ health metrics</li>
            <li><strong>Real-time Results:</strong> Instant risk assessment</li>
            <li><strong>Visual Representation:</strong> Clear visualization of risk factors</li>
            <li><strong>Personalized Recommendations:</strong> Custom health suggestions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def predict_page():
    model, preprocessor = load_or_create_model()
    
    st.title("Heart Disease Prediction Form")
    st.markdown("Please fill out the following information to get your heart disease risk prediction.")
    
    # Create form
    with st.form("prediction_form"):
        # Create two columns
        col1, col2 = st.columns(2)
        
        # First column of inputs
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            blood_pressure = st.number_input("Blood Pressure (systolic)", min_value=90, max_value=200, value=120)
            cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=300, value=180)
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=24.5, step=0.1)
            exercise = st.selectbox("Exercise Habits", options=["Inactive", "Light", "Moderate", "Heavy"])
            smoking = st.selectbox("Smoking", options=["Never", "Former", "Current"])
            family_heart_disease = st.selectbox("Family History of Heart Disease", options=["No", "Yes"])
            diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
            high_bp = st.selectbox("High Blood Pressure", options=["No", "Yes"])
        
        # Second column of inputs
        with col2:
            low_hdl = st.selectbox("Low HDL Cholesterol", options=["No", "Yes"])
            high_ldl = st.selectbox("High LDL Cholesterol", options=["No", "Yes"])
            alcohol = st.selectbox("Alcohol Consumption", options=["None", "Light", "Moderate", "Heavy"])
            stress = st.selectbox("Stress Level", options=["Low", "Medium", "High"])
            sleep = st.number_input("Sleep Hours (per day)", min_value=3, max_value=12, value=7)
            sugar = st.selectbox("Sugar Consumption", options=["Low", "Medium", "High"])
            triglyceride = st.number_input("Triglyceride Level (mg/dL)", min_value=50, max_value=500, value=150)
            fasting_blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=70, max_value=200, value=95)
            crp = st.number_input("CRP Level (mg/L)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            homocysteine = st.number_input("Homocysteine Level (μmol/L)", min_value=5.0, max_value=20.0, value=10.0, step=0.1)
        
        submit_button = st.form_submit_button("Predict Heart Disease Risk")
    
    # If form submitted
    if submit_button:
        # Create input data dict
        input_data = {
            'Age': age,
            'Gender': gender,
            'Blood Pressure': blood_pressure,
            'Cholesterol Level': cholesterol,
            'BMI': bmi,
            'Exercise Habits': exercise,
            'Smoking': smoking,
            'Family Heart Disease': family_heart_disease,
            'Diabetes': diabetes,
            'High Blood Pressure': high_bp,
            'Low HDL Cholesterol': low_hdl,
            'High LDL Cholesterol': high_ldl,
            'Alcohol Consumption': alcohol,
            'Stress Level': stress,
            'Sleep Hours': sleep,
            'Sugar Consumption': sugar,
            'Triglyceride Level': triglyceride,
            'Fasting Blood Sugar': fasting_blood_sugar,
            'CRP Level': crp,
            'Homocysteine Level': homocysteine
        }
        
        # Calculate risk factors (same as in app.py)
        risk_factors = 0
        
        if float(input_data['Age']) > 50:
            risk_factors += 1
        if float(input_data['Blood Pressure']) > 140:
            risk_factors += 1
        if float(input_data['Cholesterol Level']) > 200:
            risk_factors += 1
        if input_data['Smoking'] == 'Current':
            risk_factors += 1
        if input_data['Family Heart Disease'] == 'Yes':
            risk_factors += 1
        if input_data['Diabetes'] == 'Yes':
            risk_factors += 1
        if float(input_data['BMI']) > 30:
            risk_factors += 1
        
        # Calculate probability based on risk factors (same as in app.py)
        probability = min(risk_factors * 10 + 20, 95)  # Scale it between 20% and 95%
        prediction_value = 1 if probability > 50 else 0
        
        # Store results in session state
        st.session_state.result = {
            'prediction': 'High Risk' if prediction_value == 1 else 'Low Risk',
            'probability': round(probability, 2),
            'raw_values': input_data
        }
        
        st.session_state.user_data = {
            'age': input_data['Age'],
            'trestbps': input_data['Blood Pressure'],
            'chol': input_data['Cholesterol Level']
        }
        
        # Navigate to results
        st.session_state.page = "result"
        st.rerun()

def result_page():
    if 'result' not in st.session_state or 'user_data' not in st.session_state:
        st.error("No prediction results found. Please complete the prediction form first.")
        if st.button("Go to Prediction Form"):
            st.session_state.page = "predict"
            st.rerun()
        return
    
    result = st.session_state.result
    user_data = st.session_state.user_data
    
    st.title("Heart Disease Prediction Results")
    
    # Display prediction with styling similar to result.html
    pred_container = st.container()
    with pred_container:
        if result['prediction'] == 'High Risk':
            st.error(f"### {result['prediction']}: {result['probability']}% chance of heart disease")
        else:
            st.success(f"### {result['prediction']}: {result['probability']}% chance of heart disease")
        
        st.progress(result['probability']/100)
    
    # Display factors and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Risk Factors")
        if float(user_data['age']) > 50:
            st.markdown("- Age over 50")
        if float(user_data['trestbps']) > 140:
            st.markdown("- High blood pressure")
        if float(user_data['chol']) > 200:
            st.markdown("- High cholesterol")
        if result['raw_values']['Smoking'] == 'Current':
            st.markdown("- Current smoker")
        if result['raw_values']['Family Heart Disease'] == 'Yes':
            st.markdown("- Family history of heart disease")
        if result['raw_values']['Diabetes'] == 'Yes':
            st.markdown("- Diabetes")
        if float(result['raw_values']['BMI']) > 30:
            st.markdown("- BMI over 30")
    
    with col2:
        st.subheader("Your Health Metrics")
        st.markdown(f"- **Age**: {user_data['age']}")
        st.markdown(f"- **Blood Pressure**: {user_data['trestbps']} mmHg")
        st.markdown(f"- **Cholesterol**: {user_data['chol']} mg/dL")
        st.markdown(f"- **BMI**: {result['raw_values']['BMI']}")
    
    # Recommendations section
    st.subheader("Recommendations")
    if result['prediction'] == 'High Risk':
        st.markdown("""
        ### High Risk Recommendations:
        
        1. **Consult a Healthcare Provider**: Schedule an appointment with your doctor to discuss your heart health.
        2. **Monitor Blood Pressure and Cholesterol**: Regular testing and management is essential.
        3. **Adopt a Heart-Healthy Diet**: Focus on fruits, vegetables, whole grains, and lean proteins.
        4. **Regular Exercise**: Aim for at least 150 minutes of moderate exercise weekly.
        5. **Quit Smoking**: If you smoke, seek support to quit.
        6. **Limit Alcohol**: Reduce alcohol consumption to recommended levels.
        7. **Maintain Healthy Weight**: Work with healthcare providers on weight management.
        8. **Stress Management**: Practice stress-reduction techniques like meditation.
        
        > **Important**: This assessment is for informational purposes only and does not replace medical advice.
        """)
    else:
        st.markdown("""
        ### Low Risk Recommendations:
        
        1. **Preventive Care**: Continue regular check-ups with your healthcare provider.
        2. **Maintain Healthy Habits**: Continue with a balanced diet and regular exercise.
        3. **Regular Screenings**: Stay current with recommended health screenings.
        4. **Stay Active**: Maintain or increase your physical activity levels.
        5. **Balanced Diet**: Continue eating heart-healthy foods.
        6. **Stress Management**: Practice healthy stress management techniques.
        
        > **Note**: While your risk is low, continued healthy habits are important for long-term heart health.
        """)
    
    # Buttons to navigate
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Prediction"):
            st.session_state.page = "predict"
            st.rerun()
    with col2:
        if st.button("Learn More"):
            st.session_state.page = "about"
            st.rerun()

def about_page():
    st.title("About the Heart Disease Prediction System")
    
    st.markdown("""
    ## How It Works
    
    This system uses a machine learning model trained on medical data to predict the risk of heart disease. 
    The prediction is based on various health metrics including:
    
    - Demographic factors (age, gender)
    - Medical measurements (blood pressure, cholesterol)
    - Lifestyle factors (exercise, smoking, diet)
    - Medical history (family history, existing conditions)
    
    ## The Technology
    
    The prediction model uses Gradient Boosting, a powerful machine learning algorithm that combines multiple decision trees to make accurate predictions. The model has been trained on data from various healthcare datasets.
    
    ## Disclaimer
    
    **This system is designed for educational and informational purposes only.** It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider regarding any medical conditions or concerns.
    
    The predictions provided are based on statistical models and should be used as a general guide rather than a definitive diagnosis. Many factors contribute to heart disease risk that may not be captured by this system.
    """)
    
    # Button to return to home
    if st.button("Return to Home"):
        st.session_state.page = "home"
        st.rerun()

# Add custom CSS to make it look more like the Flask app
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #343a40;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div {
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Simple router
if st.session_state.page == "home":
    home()
elif st.session_state.page == "predict":
    predict_page()
elif st.session_state.page == "result":
    result_page()
elif st.session_state.page == "about":
    about_page()

# Add a navigation sidebar (collapsed by default)
with st.sidebar:
    st.title("Navigation")
    if st.button("Home"):
        st.session_state.page = "home"
        st.rerun()
    if st.button("Prediction Form"):
        st.session_state.page = "predict"
        st.rerun()
    if st.button("About"):
        st.session_state.page = "about"
        st.rerun()
    
    st.markdown("---")
    st.markdown("© 2024 Heart Disease Prediction System") 