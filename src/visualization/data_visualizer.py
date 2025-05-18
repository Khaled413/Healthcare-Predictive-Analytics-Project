import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os

class DataVisualizer:
    def __init__(self, output_dir='./output/visualizations'):
        """
        Initialize data visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_missing_values(self, df, filename='missing_values.png'):
        """
        Plot missing values in the dataset
        
        Args:
            df: DataFrame with data
            filename: Name of the output file
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate missing values percentage
        missing_percentage = df.isnull().mean() * 100
        
        # Plot missing values
        missing_percentage.sort_values(ascending=False).plot(kind='bar')
        plt.title('Missing Values Percentage')
        plt.xlabel('Features')
        plt.ylabel('Missing Values (%)')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def plot_feature_distributions(self, df, filename_prefix='feature_distribution'):
        """
        Plot distributions of features
        
        Args:
            df: DataFrame with data
            filename_prefix: Prefix for output files
        """
        # Separate numerical and categorical features
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        
        # Plot numerical features
        if len(numerical_features) > 0:
            n_cols = 3
            n_rows = (len(numerical_features) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, n_rows * 4))
            
            for i, feature in enumerate(numerical_features):
                plt.subplot(n_rows, n_cols, i + 1)
                sns.histplot(df[feature].dropna(), kde=True)
                plt.title(f'Distribution of {feature}')
                plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, f'{filename_prefix}_numerical.png'))
            plt.close()
        
        # Plot categorical features
        if len(categorical_features) > 0:
            n_cols = 2
            n_rows = (len(categorical_features) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, n_rows * 5))
            
            for i, feature in enumerate(categorical_features):
                plt.subplot(n_rows, n_cols, i + 1)
                value_counts = df[feature].value_counts().sort_values(ascending=False)
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'Distribution of {feature}')
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, f'{filename_prefix}_categorical.png'))
            plt.close()
    
    def plot_correlation_matrix(self, df, filename='correlation_matrix.png'):
        """
        Plot correlation matrix of numerical features
        
        Args:
            df: DataFrame with data
            filename: Name of the output file
        """
        # Select only numerical features
        numerical_df = df.select_dtypes(include=['int64', 'float64'])
        
        if numerical_df.shape[1] > 1:
            plt.figure(figsize=(12, 10))
            
            # Calculate correlation matrix
            corr_matrix = numerical_df.corr()
            
            # Plot heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
    
    def plot_target_distribution(self, y, target_name='Target', filename='target_distribution.png'):
        """
        Plot distribution of target variable
        
        Args:
            y: Target variable
            target_name: Name of the target variable
            filename: Name of the output file
        """
        plt.figure(figsize=(10, 6))
        
        # Plot target distribution
        value_counts = pd.Series(y).value_counts().sort_index()
        sns.barplot(x=value_counts.index, y=value_counts.values)
        
        # Add percentage labels
        total = len(y)
        for i, count in enumerate(value_counts.values):
            percentage = 100 * count / total
            plt.text(i, count + 5, f'{percentage:.1f}%', ha='center')
        
        plt.title(f'Distribution of {target_name}')
        plt.xlabel(target_name)
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def plot_before_after_smote(self, y_before, y_after, target_name='Target', filename='smote_comparison.png'):
        """
        Plot target distribution before and after SMOTE
        
        Args:
            y_before: Target variable before SMOTE
            y_after: Target variable after SMOTE
            target_name: Name of the target variable
            filename: Name of the output file
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate value counts
        before_counts = pd.Series(y_before).value_counts().sort_index()
        after_counts = pd.Series(y_after).value_counts().sort_index()
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'Before SMOTE': before_counts,
            'After SMOTE': after_counts
        })
        
        # Plot
        df_plot.plot(kind='bar')
        plt.title(f'Distribution of {target_name} Before and After SMOTE')
        plt.xlabel(target_name)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, filename='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            filename: Name of the output file
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def plot_roc_curve(self, y_true, y_prob, filename='roc_curve.png'):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label='Yes')  # تحديد 'Yes' كقيمة إيجابية
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_prob, filename='precision_recall_curve.png'):
        """
        Plot precision-recall curve
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            filename: Name of the output file
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # Plot precision-recall curve
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, top_n=20, filename='feature_importance.png'):
        """
        Plot feature importance
        
        Args:
            model: Trained model
            feature_names: Names of features
            top_n: Number of top features to show
            filename: Name of the output file
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute. Skipping feature importance plot.")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Select top N features
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_feature_names = [feature_names[i] for i in top_indices]
        
        plt.figure(figsize=(12, 8))
        
        # Plot feature importances
        sns.barplot(x=top_importances, y=top_feature_names)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()