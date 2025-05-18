import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataCleaner:
    """
    Class for cleaning and preprocessing healthcare data
    """
    def __init__(self, output_dir='./cleaned_data'):
        """
        Initialize data cleaner
        
        Args:
            output_dir: Directory to save cleaned data and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_data(self, data):
        """
        Analyze data for missing values, duplicates, and basic statistics
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Check for missing values
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        
        # Check for duplicates
        duplicates = data.duplicated().sum()
        
        # Get basic statistics
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        analysis = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': missing_values,
            'missing_percentage': missing_percentage,
            'duplicates': duplicates,
            'numeric_columns': list(numeric_columns),
            'categorical_columns': list(categorical_columns)
        }
        
        # Print analysis summary
        print(f"Data Analysis Summary:")
        print(f"Total rows: {analysis['total_rows']}")
        print(f"Total columns: {analysis['total_columns']}")
        print(f"Duplicates: {analysis['duplicates']}")
        print("\nMissing Values:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  {col}: {count} ({missing_percentage[col]:.2f}%)")
        
        return analysis
    
    def clean_data(self, data):
        """
        Clean data by handling missing values and duplicates
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_data = data.copy()
        
        # Remove duplicates
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        if len(cleaned_data) < initial_rows:
            print(f"Removed {initial_rows - len(cleaned_data)} duplicate rows")
        
        # Handle missing values based on column type
        for column in cleaned_data.columns:
            missing_count = cleaned_data[column].isnull().sum()
            if missing_count > 0:
                if cleaned_data[column].dtype in ['int64', 'float64']:
                    # For numeric columns, fill with median
                    median_value = cleaned_data[column].median()
                    cleaned_data[column].fillna(median_value, inplace=True)
                    print(f"Filled {missing_count} missing values in '{column}' with median ({median_value})")
                else:
                    # For categorical columns, fill with mode
                    mode_value = cleaned_data[column].mode()[0]
                    cleaned_data[column].fillna(mode_value, inplace=True)
                    print(f"Filled {missing_count} missing values in '{column}' with mode ({mode_value})")
        
        return cleaned_data
    
    def create_preprocessor(self, X):
        """
        Create a preprocessing pipeline for features
        
        Args:
            X: Features DataFrame
            
        Returns:
            Preprocessor pipeline
        """
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def visualize_data(self, data, target_column, output_dir=None):
        """
        Create visualizations for data exploration
        
        Args:
            data: DataFrame to visualize
            target_column: Name of the target column
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Distribution of target variable
        plt.figure(figsize=(10, 6))
        target_counts = data[target_column].value_counts()
        ax = sns.countplot(x=target_column, data=data)
        plt.title(f'Distribution of {target_column}')
        plt.xlabel(target_column)
        plt.ylabel('Count')
        
        # Add count labels
        for i, count in enumerate(target_counts):
            ax.text(i, count + 5, str(count), ha='center')
            
        plt.savefig(os.path.join(output_dir, f'{target_column}_distribution.png'))
        plt.close()
        
        # Correlation heatmap for numeric features
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        if len(numeric_data.columns) > 1:  # Need at least 2 columns for correlation
            plt.figure(figsize=(12, 10))
            correlation = numeric_data.corr()
            mask = np.triu(correlation)
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
            plt.close()
        
        # Distribution of numeric features by target
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        for feature in numeric_features:
            if feature != target_column:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=data, x=feature, hue=target_column, kde=True, element="step")
                plt.title(f'Distribution of {feature} by {target_column}')
                plt.savefig(os.path.join(output_dir, f'{feature}_by_{target_column}.png'))
                plt.close()
        
        # Bar plots for categorical features
        categorical_features = data.select_dtypes(include=['object', 'category']).columns
        for feature in categorical_features:
            if feature != target_column:
                plt.figure(figsize=(12, 6))
                crosstab = pd.crosstab(data[feature], data[target_column], normalize='index')
                crosstab.plot(kind='bar', stacked=True)
                plt.title(f'{feature} vs {target_column}')
                plt.ylabel('Proportion')
                plt.legend(title=target_column)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{feature}_vs_{target_column}.png'))
                plt.close()
        
        print(f"Visualizations saved to {output_dir}")