# Heart Disease Prediction system

## Overvie
A modern web application that uses machine learning to predict heart disease risk based on various health factors. The system provides personalized risk assessments and health recommendations.

![image](https://github.com/user-attachments/assets/43033a27-3082-47f9-9c17-52949a1d71fd)

## Features
- **User-friendly Interface**: Intuitive design for easy data entry and navigation
- **Comprehensive Analysis**: Evaluates 20+ health metrics to generate accurate predictions
- **Real-time Results**: Instant risk assessment with percentage probability
- **Visual Representation**: Clear visualization of risk factors and impact levels
- **Personalized Recommendations**: Custom health suggestions based on individual results
- **Responsive Design**: Optimized experience across desktop and mobile devices

## Screenshots

### Home Page
![image](https://github.com/user-attachments/assets/bd1b42bf-d936-4701-99bc-89c055e67662)
*The landing page with system introduction and navigation*

### Prediction Form
![image](https://github.com/user-attachments/assets/d6021863-3354-46e3-806d-2e699091b969)
*The data entry form for health information*

### Results Page

**Hight Risk**

![image](https://github.com/user-attachments/assets/927996cd-f762-41a0-a3a4-d9d65ce40371)

**Low Risk**

![image](https://github.com/user-attachments/assets/78798ba2-fb78-483c-ba76-06f9c82feadf)

*Detailed prediction results with risk assessment and recommendations*

### About Page
![image](https://github.com/user-attachments/assets/c33319a6-3465-4f82-be9d-5320289437c8)
*Information about the system, how it works, and disclaimer*

## Health Factors Analyzed
- **Demographic Factors**: Age, gender, BMI
- **Vital Measurements**: Blood pressure, cholesterol, triglyceride levels, fasting blood sugar, CRP level
- **Lifestyle Factors**: Exercise habits, smoking status, alcohol consumption, stress level, sleep patterns
- **Medical History**: Family history of heart disease, diabetes, high blood pressure, cholesterol levels

## Technical Details
- **Backend**: Flask web framework with Python
- **Machine Learning**: Gradient Boosting algorithm from scikit-learn
- **Frontend**: HTML5, CSS3 with responsive design principles
- **Data Processing**: NumPy, Pandas for data manipulation
- **Visualization**: Custom visualization components for risk representation

## Project Structure
```
│
├── app.py                   # Main Flask application
├── templates/               # HTML templates
│   ├── index.html           # Home page
│   ├── predict.html         # Prediction form
│   ├── result.html          # Results display
│   └── about.html           # About page
├── static/                  # Static assets (CSS, JS, images)
├── models/                  # Trained machine learning models
├── screenshots/             # Application screenshots
└── requirements.txt         # Dependencies
```

## Setup Instructions
1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv healthcare_env
   ```
3. Activate the environment:
   - Windows: `healthcare_env\Scripts\activate`
   - Mac/Linux: `source healthcare_env/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the application:
   ```
   python app.py
   ```
6. Access the web interface at `http://localhost:5000`

## 👥 Contributors

Thanks to these wonderful Team:

<a href="https://github.com/mostafataha12">
  <img src="https://github.com/mostafataha12.png" width="60px" alt="Mostafa Taha"/>
</a>

<a href="https://github.com/AliGaMal1">
  <img src="https://github.com/AliGaMal1.png" width="60px" alt="Ali Gamal"/>
</a>


## Important Disclaimer
**This system is designed for educational and informational purposes only.** It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider regarding any medical conditions or concerns.
