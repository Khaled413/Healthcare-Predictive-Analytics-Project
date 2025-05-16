import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import os

class DataCleaner:
    """Class for cleaning and preprocessing healthcare data"""
    
    def __init__(self, output_dir='./results/preprocessing'):
        """
        Initialize the data cleaner
        
        Args:
            output_dir: Directory to save preprocessing results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize preprocessing components
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.numerical_scaler = None
        self.categorical_encoder = None
        self.label_encoder = None
        
        # Store column information
        self.numerical_columns = []
        self.categorical_columns = []
        self.target_column = None
    
    def identify_column_types(self, df, target_column):
        """
        Identify numerical and categorical columns
        
        Args:
            df: DataFrame containing the data
            target_column: Name of the target column
            
        Returns:
            Tuple of (numerical_columns, categorical_columns)
        """
        self.target_column = target_column
        
        # Exclude target column from features
        feature_df = df.drop(columns=[target_column])
        
        # Identify numerical and categorical columns
        numerical_columns = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = feature_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        print(f"Identified {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns")
        
        return numerical_columns, categorical_columns
    
    def check_missing_values(self, df):
        """
        Check for missing values in the dataset
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            DataFrame with missing value statistics
        """
        # Calculate missing values
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        # Create summary DataFrame
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentage
        })
        
        # Filter to show only columns with missing values
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
        
        # Save missing values report
        if not missing_df.empty:
            missing_df.to_csv(f"{self.output_dir}/missing_values_report.csv")
            print(f"Missing values report saved to {self.output_dir}/missing_values_report.csv")
        
        return missing_df
    
    def handle_missing_values(self, df, numerical_strategy='median', categorical_strategy='most_frequent', knn_impute=False, k=5):
        """
        Handle missing values in the dataset
        
        Args:
            df: DataFrame containing the data
            numerical_strategy: Strategy for numerical imputation ('mean', 'median', 'most_frequent')
            categorical_strategy: Strategy for categorical imputation ('most_frequent', 'constant')
            knn_impute: Whether to use KNN imputation for numerical features
            k: Number of neighbors for KNN imputation
            
        Returns:
            DataFrame with imputed values
        """
        # Create a copy of the DataFrame
        df_imputed = df.copy()
        
        # Handle numerical missing values
        if self.numerical_columns:
            if knn_impute:
                self.numerical_imputer = KNNImputer(n_neighbors=k)
                numerical_data = df_imputed[self.numerical_columns].values
                imputed_numerical = self.numerical_imputer.fit_transform(numerical_data)
                df_imputed[self.numerical_columns] = imputed_numerical
            else:
                self.numerical_imputer = SimpleImputer(strategy=numerical_strategy)
                df_imputed[self.numerical_columns] = self.numerical_imputer.fit_transform(df_imputed[self.numerical_columns])
        
        # Handle categorical missing values
        if self.categorical_columns:
            self.categorical_imputer = SimpleImputer(strategy=categorical_strategy)
            df_imputed[self.categorical_columns] = self.categorical_imputer.fit_transform(df_imputed[self.categorical_columns])
        
        return df_imputed
    
    def handle_outliers(self, df, method='iqr', threshold=1.5):
        """
        Handle outliers in numerical features
        
        Args:
            df: DataFrame containing the data
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        # Create a copy of the DataFrame
        df_no_outliers = df.copy()
        
        # Handle outliers for each numerical column
        for col in self.numerical_columns:
            if method == 'iqr':
                # IQR method
                Q1 = df_no_outliers[col].quantile(0.25)
                Q3 = df_no_outliers[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers
                df_no_outliers[col] = df_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                mean = df_no_outliers[col].mean()
                std = df_no_outliers[col].std()
                
                # Cap outliers
                df_no_outliers[col] = df_no_outliers[col].clip(
                    lower=mean - threshold * std,
                    upper=mean + threshold * std
                )
        
        return df_no_outliers
    
    def scale_numerical_features(self, df, method='standard'):
        """
        Scale numerical features
        
        Args:
            df: DataFrame containing the data
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            DataFrame with scaled numerical features
        """
        # Create a copy of the DataFrame
        df_scaled = df.copy()
        
        if method == 'standard':
            self.numerical_scaler = StandardScaler()
        elif method == 'minmax':
            self.numerical_scaler = MinMaxScaler()
        
        # Scale numerical features
        if self.numerical_columns:
            df_scaled[self.numerical_columns] = self.numerical_scaler.fit_transform(df_scaled[self.numerical_columns])
        
        return df_scaled
    
    def encode_categorical_features(self, df, method='onehot'):
        """
        Encode categorical features
        
        Args:
            df: DataFrame containing the data
            method: Encoding method ('onehot', 'label')
            
        Returns:
            DataFrame with encoded categorical features
        """
        # Create a copy of the DataFrame
        df_encoded = df.copy()
        
        if method == 'onehot':
            # One-hot encoding
            if self.categorical_columns:
                self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded_data = self.categorical_encoder.fit_transform(df_encoded[self.categorical_columns])
                
                # Create DataFrame with encoded data
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=self.categorical_encoder.get_feature_names_out(self.categorical_columns)
                )
                
                # Drop original categorical columns and add encoded columns
                df_encoded = df_encoded.drop(columns=self.categorical_columns)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
        
        elif method == 'label':
            # Label encoding
            if self.categorical_columns:
                self.categorical_encoder = {}
                
                for col in self.categorical_columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.categorical_encoder[col] = le
        
        return df_encoded
    
    def encode_target(self, df):
        """
        Encode target variable
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            DataFrame with encoded target
        """
        # Create a copy of the DataFrame
        df_encoded = df.copy()
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        df_encoded[self.target_column] = self.label_encoder.fit_transform(df_encoded[self.target_column])
        
        return df_encoded
    
    def save_preprocessor(self, filename='preprocessor.pkl'):
        """
        Save preprocessing components
        
        Args:
            filename: Name of the file to save
        """
        import joblib
        
        # Create preprocessor dictionary
        preprocessor = {
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'target_column': self.target_column,
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'numerical_scaler': self.numerical_scaler,
            'categorical_encoder': self.categorical_encoder,
            'label_encoder': self.label_encoder
        }
        
        # Save preprocessor
        joblib.dump(preprocessor, f"{self.output_dir}/{filename}")
        print(f"Preprocessor saved to {self.output_dir}/{filename}")
        
        return f"{self.output_dir}/{filename}"