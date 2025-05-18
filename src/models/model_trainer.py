import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os
from sklearn.model_selection import RandomizedSearchCV  # أضف هذا الاستيراد في بداية الملف

class ModelTrainer:
    """Class for training and evaluating machine learning models with SMOTE balancing"""
    
    def __init__(self, output_dir='./results/models'):
        """
        Initialize model trainer
        
        Args:
            output_dir: Directory to save model results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of data to use for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(self, X_train, y_train, random_state=42):
        """
        Apply SMOTE to balance the training data
        
        Args:
            X_train: Training features
            y_train: Training target
            random_state: Random seed
            
        Returns:
            Tuple of (X_train_balanced, y_train_balanced)
        """
        # Check class distribution before SMOTE
        print("Class distribution before SMOTE:")
        print(pd.Series(y_train).value_counts(normalize=True))
        
        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        # Check class distribution after SMOTE
        print("Class distribution after SMOTE:")
        print(pd.Series(y_train_balanced).value_counts(normalize=True))
        
        print(f"Original training set: {X_train.shape[0]} samples")
        print(f"Balanced training set: {X_train_balanced.shape[0]} samples")
        
        return X_train_balanced, y_train_balanced
    
    def train_logistic_regression(self, X_train, y_train, cv=5):
        """
        Train logistic regression model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Trained model
        """
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 200, 300]
        }
        
        # Create base model
        base_model = LogisticRegression(random_state=42)
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Store model
        self.models['logistic_regression'] = best_model
        
        print(f"Logistic Regression - Best parameters: {grid_search.best_params_}")
        print(f"Logistic Regression - Best ROC AUC: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def train_random_forest(self, X_train, y_train, cv=5):
        """
        Train random forest model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Trained model
        """
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        base_model = RandomForestClassifier(random_state=42)
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Store model
        self.models['random_forest'] = best_model
        
        print(f"Random Forest - Best parameters: {grid_search.best_params_}")
        print(f"Random Forest - Best ROC AUC: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def train_gradient_boosting(self, X_train, y_train, cv=5):
        """
        Train gradient boosting model with predefined parameters
        
        Args:
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Trained model
        """
        # Create model with predefined parameters
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Store model
        self.models['gradient_boosting'] = model
        
        # Evaluate on training data
        train_score = model.score(X_train, y_train)
        print(f"Gradient Boosting - Training accuracy: {train_score:.4f}")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, cv=5):
        """Train gradient boosting model with hyperparameter tuning"""
        # تعريف نطاق أوسع من المعلمات
        param_dist = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Create base model
        base_model = GradientBoostingClassifier(random_state=42)
        
        # استخدام RandomizedSearchCV بدلاً من GridSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=30,  # عدد التركيبات التي سيتم تجربتها
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        
        # Fit random search
        random_search.fit(X_train, y_train)
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Store model
        self.models['gradient_boosting'] = best_model
        
        print(f"Gradient Boosting - Best parameters: {random_search.best_params_}")
        print(f"Gradient Boosting - Best ROC AUC: {random_search.best_score_:.4f}")
        
        return best_model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='Yes'),  # Set positive label
            'recall': recall_score(y_test, y_pred, pos_label='Yes'),  # Set positive label
            'f1': f1_score(y_test, y_pred, pos_label='Yes'),  # Set positive label
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        # Print metrics
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        
        return joblib.load(filepath)