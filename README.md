# Heart Disease Prediction System

## Overview
A modern web application that uses machine learning to predict heart disease risk based on various health factors. The system provides personalized risk assessments and health recommendations.

![image](https://github.com/user-attachments/assets/ff9d2c86-85ec-491f-b484-19ef1efc44ea)

## Features
- **User-friendly Interface**: Intuitive design for easy data entry and navigation
- **Comprehensive Analysis**: Evaluates 20+ health metrics to generate accurate predictions
- **Real-time Results**: Instant risk assessment with percentage probability
- **Visual Representation**: Clear visualization of risk factors and impact levels
- **Personalized Recommendations**: Custom health suggestions based on individual results
- **Responsive Design**: Optimized experience across desktop and mobile devices

## Screenshots

### Home Page
![image](https://github.com/user-attachments/assets/f9c91fb0-3315-47fc-9171-801db50c72ea)
*The landing page with system introduction and navigation*

### Prediction Form
![image](https://github.com/user-attachments/assets/c8d6cb7f-6d76-415a-951c-7f823a2dd382)
*The data entry form for health information*

### Results Page

**Hight Risk**

![image](https://github.com/user-attachments/assets/817e58ed-ed09-4d94-abe1-82770b59bfcc)

**Low Risk**

![image](https://github.com/user-attachments/assets/ca96e452-cc99-4e09-8cb8-c21d15bdebff)

*Detailed prediction results with risk assessment and recommendations*

### About Page
![image](https://github.com/user-attachments/assets/70f3ef26-de93-4dd9-bd86-1813d028527c)
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
