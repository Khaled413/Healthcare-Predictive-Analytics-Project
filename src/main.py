import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data.data_loader import load_data, get_feature_target_split
from src.data.data_preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.visualization.data_visualizer import DataVisualizer
from src.deployment.model_api import start_api


def main():
    """Main function to run the healthcare predictive model pipeline"""
    print("Starting Healthcare Predictive Model Pipeline...")
    
    # Create output directories
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)
    os.makedirs("output/evaluation", exist_ok=True)
    
    # Initialize visualizer and evaluator
    visualizer = DataVisualizer(output_dir="output/visualizations")
    evaluator = ModelEvaluator(output_dir="output/evaluation")
    
    # 1. Load data - use the specified file path directly
    data_path = "e:\\رواد مصر الرقمية\\Healthcare-PredictiveProject\\heart_disease.csv"
    df = load_data(data_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # 2. Visualize raw data
    print("Generating visualizations of raw data...")
    visualizer.plot_missing_values(df)
    visualizer.plot_feature_distributions(df)
    visualizer.plot_correlation_matrix(df)
    
    # 3. Split into features and target - use the specified target column
    target_column = 'Heart Disease Status'
    X, y = get_feature_target_split(df, target_column)
    
    # Visualize target distribution
    visualizer.plot_target_distribution(y, target_name=target_column, filename='target_distribution_before.png')
    print(f"Target distribution before balancing: {pd.Series(y).value_counts()}")
    print(f"Target distribution percentage: {pd.Series(y).value_counts(normalize=True) * 100}")
    
    # 4. Split into train and test sets
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 5. Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. Apply SMOTE to balance the dataset
    X_train_resampled, y_train_resampled = preprocessor.apply_smote(X_train_processed, y_train)
    
    # Visualize target distribution after SMOTE
    visualizer.plot_before_after_smote(y_train, y_train_resampled, target_name=target_column)
    print(f"Target distribution after balancing with SMOTE: {pd.Series(y_train_resampled).value_counts()}")
    print(f"Target distribution percentage after SMOTE: {pd.Series(y_train_resampled).value_counts(normalize=True) * 100}")
    
    # 7. Train model - call the specific training method directly
    print("Training gradient boosting model...")
    trainer = ModelTrainer()
    model = trainer.train_gradient_boosting(X_train_resampled, y_train_resampled)
    
    # 8. Evaluate model
    print("Evaluating model...")
    feature_names = preprocessor.get_feature_names()
    metrics = evaluator.evaluate_model(model, X_test_processed, y_test, feature_names)
    
    # 9. Visualize results
    print("Generating result visualizations...")
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    
    visualizer.plot_confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])
    visualizer.plot_roc_curve(y_test, y_prob)
    visualizer.plot_precision_recall_curve(y_test, y_prob)
    visualizer.plot_feature_importance(model, feature_names)
    
    # 10. Save model and preprocessor
    print("Saving model and preprocessor...")
    model_path = "output/models/heart_disease_model.joblib"
    preprocessor_path = "output/models/preprocessor.joblib"
    
    trainer.save_model(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    # 11. Ask if user wants to start API
    start_api_choice = input("Do you want to start the prediction API? (y/n): ")
    if start_api_choice.lower() == 'y':
        print("Starting API...")
        start_api(model_path, preprocessor_path)


if __name__ == "__main__":
    main()