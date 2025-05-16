# Heart Disease Prediction System

## Overview
A modern web application that uses machine learning to predict heart disease risk based on various health factors. The system provides personalized risk assessments and health recommendations.

![Homepage Screenshot](screenshots/homepage.png)

## Features
- **User-friendly Interface**: Intuitive design for easy data entry and navigation
- **Comprehensive Analysis**: Evaluates 20+ health metrics to generate accurate predictions
- **Real-time Results**: Instant risk assessment with percentage probability
- **Visual Representation**: Clear visualization of risk factors and impact levels
- **Personalized Recommendations**: Custom health suggestions based on individual results
- **Responsive Design**: Optimized experience across desktop and mobile devices

## Screenshots

### Home Page
![Home Page](screenshots/home.png)
*The landing page with system introduction and navigation*

### Prediction Form
![Prediction Form](screenshots/prediction-form.png)
*The data entry form for health information*

### Results Page
![Results Page](screenshots/results.png)
*Detailed prediction results with risk assessment and recommendations*

### About Page
![About Page](screenshots/about.png)
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

## Creating Screenshots
To add the screenshots referenced in this README:
1. Create a `screenshots` directory in the project root
2. Take screenshots of each page of the running application
3. Save them with the following names:
   - `home.png`
   - `prediction-form.png`
   - `results.png`
   - `about.png`

## Important Disclaimer
**This system is designed for educational and informational purposes only.** It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider regarding any medical conditions or concerns.