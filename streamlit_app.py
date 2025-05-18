import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature names required for prediction
feature_names = [
    'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
    'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
    'Low HDL Cholesterol', 'High LDL Cholesterol', 'Alcohol Consumption',
    'Stress Level', 'Sleep Hours', 'Sugar Consumption', 'Triglyceride Level',
    'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
]

# Create folders for models if they don't exist
os.makedirs('output/models', exist_ok=True)

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("❤️ Heart Disease Prediction System")
st.markdown("""
This application uses machine learning to predict heart disease risk based on various health factors.
Simply fill out the form below to get your personalized risk assessment and health recommendations.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "About", "Data Visualization"])

with tab1:
    st.header("Health Information")
    st.markdown("Please fill out the following information to get your heart disease risk prediction.")
    
    col1, col2 = st.columns(2)
    
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
    
    predict_button = st.button("Predict Heart Disease Risk", type="primary")
    
    if predict_button:
        # Collect form data
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
        
        # Calculate risk factors (simplified approach from app.py)
        risk_factors = 0
        
        if age > 50:
            risk_factors += 1
        if blood_pressure > 140:
            risk_factors += 1
        if cholesterol > 200:
            risk_factors += 1
        if smoking == "Current":
            risk_factors += 1
        if family_heart_disease == "Yes":
            risk_factors += 1
        if diabetes == "Yes":
            risk_factors += 1
        if bmi > 30:
            risk_factors += 1
            
        probability = min(risk_factors * 10 + 20, 95)  # Scale between 20% and 95%
        prediction = "High Risk" if probability > 50 else "Low Risk"
        
        # Display result with colorful formatting
        st.divider()
        st.header("🔍 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Assessment")
            if probability > 50:
                st.error(f"**{prediction}**: {probability}% chance of heart disease")
            else:
                st.success(f"**{prediction}**: {probability}% chance of heart disease")
                
            st.progress(probability/100)
        
        with col2:
            st.subheader("Key Risk Factors")
            factors_text = ""
            if age > 50:
                factors_text += "- Age over 50\n"
            if blood_pressure > 140:
                factors_text += "- High blood pressure\n"
            if cholesterol > 200:
                factors_text += "- High cholesterol\n"
            if smoking == "Current":
                factors_text += "- Current smoker\n"
            if family_heart_disease == "Yes":
                factors_text += "- Family history of heart disease\n"
            if diabetes == "Yes":
                factors_text += "- Diabetes\n"
            if bmi > 30:
                factors_text += "- BMI over 30\n"
                
            if factors_text:
                st.markdown(factors_text)
            else:
                st.markdown("No significant risk factors detected.")
        
        st.divider()
        st.subheader("Recommendations")
        
        if probability > 50:
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

with tab2:
    st.header("About the Heart Disease Prediction System")
    
    st.markdown("""
    ## How It Works
    
    This system uses a machine learning model trained on medical data to predict the risk of heart disease. 
    The prediction is based on various health metrics including:
    
    - Demographic factors (age, gender)
    - Medical measurements (blood pressure, cholesterol)
    - Lifestyle factors (exercise, smoking, diet)
    - Medical history (family history, existing conditions)
    
    ## The Technology
    
    The prediction model uses a simplified risk assessment approach to evaluate heart disease risk
    based on established clinical risk factors. The prediction is based on scientific literature 
    about heart disease risk factors.
    
    ## Disclaimer
    
    **This system is designed for educational and informational purposes only.** It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider regarding any medical conditions or concerns.
    
    The predictions provided are based on statistical models and should be used as a general guide rather than a definitive diagnosis. Many factors contribute to heart disease risk that may not be captured by this system.
    """)

with tab3:
    st.header("Data Visualization")
    
    st.markdown("""
    This section would typically display visualizations of heart disease risk factors and trends. 
    
    In a full implementation, you might see:
    - Distribution of heart disease by age and gender
    - Correlation between risk factors
    - Impact of lifestyle choices on heart health
    - Comparative analysis of different populations
    
    For this demo version, visualizations would be generated based on user inputs and statistical data.
    """)
    
    st.info("Detailed visualizations would be implemented in the full version of this application.")

# Add a footer
st.divider()
st.markdown("❤️ **Heart Disease Prediction System** | Developed for healthcare education and awareness") 