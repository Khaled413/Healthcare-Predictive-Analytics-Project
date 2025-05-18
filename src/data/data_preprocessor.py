import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib


class DataPreprocessor:
    def __init__(self):
        """Initialize data preprocessor"""
        self.preprocessor = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
    
    def fit_transform(self, X):
        """
        Fit preprocessor and transform data
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed features
        """
        # Identify numerical and categorical features
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"Numerical features: {self.numerical_features}")
        print(f"Categorical features: {self.categorical_features}")
        
        # Create preprocessing pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create column transformer
        self.preprocessor = ColumnTransformer([
            ('numerical', numerical_pipeline, self.numerical_features),
            ('categorical', categorical_pipeline, self.categorical_features)
        ])
        
        # Fit and transform
        X_transformed = self.preprocessor.fit_transform(X)
        
        return X_transformed
    
    def transform(self, X):
        """
        Transform data using fitted preprocessor
        
        Args:
            X: Features DataFrame
            
        Returns:
            Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        return self.preprocessor.transform(X)
    
    def apply_smote(self, X, y, random_state=42):
        """
        Apply SMOTE to balance the dataset
        
        Args:
            X: Features
            y: Target
            random_state: Random seed for reproducibility
            
        Returns:
            X_resampled, y_resampled: Balanced features and target
        """
        print("Applying SMOTE to balance the dataset...")
        print(f"Before SMOTE - Class distribution: {pd.Series(y).value_counts()}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"After SMOTE - Class distribution: {pd.Series(y_resampled).value_counts()}")
        
        return X_resampled, y_resampled
    
    def get_feature_names(self):
        """
        Get feature names after transformation
        
        Returns:
            List of feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        try:
            # For sklearn >= 1.0
            feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            # For sklearn < 1.0
            feature_names = []
            for name, trans, cols in self.preprocessor.transformers_:
                if name == 'numerical':
                    feature_names.extend(cols)
                elif name == 'categorical':
                    for col in cols:
                        feature_names.extend([f"{col}_{cat}" for cat in trans.named_steps['onehot'].categories_[cols.index(col)]])
        
        return feature_names