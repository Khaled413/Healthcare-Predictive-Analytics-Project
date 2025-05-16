import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import os


class ModelEvaluator:
    def __init__(self, output_dir='./output/evaluation'):
        """
        Initialize model evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model(self, model, X_test, y_test, feature_names=None, threshold=0.3):
        """Evaluate model performance"""
        # Get predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Apply custom threshold instead of default 0.5
        y_pred = (y_prob >= threshold).astype(str)
        y_pred = np.where(y_pred == "True", "Yes", "No")
        
        # Calculate metrics with pos_label='Yes'
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='Yes'),
            'recall': recall_score(y_test, y_pred, pos_label='Yes'),
            'f1': f1_score(y_test, y_pred, pos_label='Yes'),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'avg_precision': average_precision_score(y_test, y_prob, pos_label='Yes')
        }
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save metrics to file
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.output_dir, 'metrics.csv'), index=False)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred)
        
        # Plot ROC curve
        self._plot_roc_curve(y_test, y_prob)
        
        # Plot precision-recall curve
        self._plot_precision_recall_curve(y_test, y_prob)
        
        # Plot feature importance if available
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            self._plot_feature_importance(model, feature_names)
        
        return metrics
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_prob):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label='Yes')  # تحديد 'Yes' كقيمة إيجابية
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'))
        plt.close()
    
    def _plot_precision_recall_curve(self, y_true, y_prob):
        """Plot precision-recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label='Yes')  # تحديد 'Yes' كقيمة إيجابية
        avg_precision = average_precision_score(y_true, y_prob, pos_label='Yes')  # تحديد 'Yes' كقيمة إيجابية
        
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'))
        plt.close()
    
    def _plot_feature_importance(self, model, feature_names, top_n=20):
        """Plot feature importance"""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Select top N features
        top_n = min(top_n, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(top_n), importances[top_indices])
        plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        plt.close()