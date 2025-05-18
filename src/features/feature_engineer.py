import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FeatureEngineer:
    """
    Class for feature engineering and selection
    """
    def __init__(self, output_dir='./feature_engineering'):
        """
        Initialize feature engineer
        
        Args:
            output_dir: Directory to save feature engineering results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def apply_smote(self, X, y, random_state=42):
        """
        Apply SMOTE to balance the target distribution
        
        Args:
            X: Features
            y: Target
            random_state: Random seed for reproducibility
            
        Returns:
            X_resampled, y_resampled: Balanced datasets
        """
        print("Applying SMOTE to balance class distribution...")
        
        # Print original class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Original class distribution:")
        for cls, count in zip(unique, counts):
            print(f"  {cls}: {count} ({count/len(y)*100:.2f}%)")
        
        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Print new class distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        print("Balanced class distribution after SMOTE:")
        for cls, count in zip(unique, counts):
            print(f"  {cls}: {count} ({count/len(y_resampled)*100:.2f}%)")
        
        # Visualize the effect of SMOTE
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        original_counts = pd.Series(y).value_counts()
        sns.barplot(x=original_counts.index, y=original_counts.values)
        plt.title('Original Class Distribution')
        plt.ylabel('Count')
        plt.xlabel('Class')
        
        plt.subplot(1, 2, 2)
        resampled_counts = pd.Series(y_resampled).value_counts()
        sns.barplot(x=resampled_counts.index, y=resampled_counts.values)
        plt.title('Class Distribution After SMOTE')
        plt.ylabel('Count')
        plt.xlabel('Class')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'smote_effect.png'))
        plt.close()
        
        return X_resampled, y_resampled
    
    def select_features(self, X, y, method='all', k=10):
        """
        Select important features using various methods
        
        Args:
            X: Features
            y: Target
            method: Feature selection method ('univariate', 'rfe', 'importance', or 'all')
            k: Number of features to select
            
        Returns:
            selected_features: List of selected feature names
            feature_scores: Dictionary of feature scores
        """
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        feature_scores = {}
        
        if method in ['univariate', 'all']:
            # Univariate feature selection
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X, y)
            univariate_scores = selector.scores_
            feature_scores['univariate'] = dict(zip(feature_names, univariate_scores))
            
            # Visualize univariate scores
            plt.figure(figsize=(12, 6))
            scores_df = pd.DataFrame({'Feature': feature_names, 'Score': univariate_scores})
            scores_df = scores_df.sort_values('Score', ascending=False)
            sns.barplot(x='Score', y='Feature', data=scores_df.head(k))
            plt.title(f'Top {k} Features (Univariate F-test)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'univariate_feature_scores.png'))
            plt.close()
        
        if method in ['rfe', 'all']:
            # Recursive feature elimination
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator, n_features_to_select=k)
            rfe.fit(X, y)
            rfe_scores = rfe.ranking_
            # Convert rankings to scores (lower rank = higher score)
            rfe_scores = max(rfe_scores) - rfe_scores + 1
            feature_scores['rfe'] = dict(zip(feature_names, rfe_scores))
            
            # Visualize RFE results
            plt.figure(figsize=(12, 6))
            selected_features_mask = rfe.support_
            plt.bar(range(len(selected_features_mask)), 
                    [1 if x else 0 for x in selected_features_mask], 
                    tick_label=feature_names)
            plt.xticks(rotation=90)
            plt.title('Features Selected by RFE')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'rfe_selected_features.png'))
            plt.close()
        
        if method in ['importance', 'all']:
            # Feature importance from Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importance_scores = rf.feature_importances_
            feature_scores['importance'] = dict(zip(feature_names, importance_scores))
            
            # Visualize feature importances
            plt.figure(figsize=(12, 6))
            importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
            importances_df = importances_df.sort_values('Importance', ascending=False)
            sns.barplot(x='Importance', y='Feature', data=importances_df.head(k))
            plt.title(f'Top {k} Features (Random Forest Importance)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importances.png'))
            plt.close()
        
        # Combine scores from different methods if 'all' is selected
        if method == 'all':
            # Normalize scores from each method
            normalized_scores = {}
            for method_name, scores in feature_scores.items():
                max_score = max(scores.values())
                normalized_scores[method_name] = {feat: score/max_score for feat, score in scores.items()}
            
            # Combine normalized scores
            combined_scores = {feat: sum(normalized_scores[method_name][feat] for method_name in normalized_scores) 
                              for feat in feature_names}
            feature_scores['combined'] = combined_scores
            
            # Select top k features based on combined scores
            selected_features = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]
            
            # Visualize combined scores
            plt.figure(figsize=(12, 6))
            combined_df = pd.DataFrame({'Feature': feature_names, 'Score': [combined_scores[feat] for feat in feature_names]})
            combined_df = combined_df.sort_values('Score', ascending=False)
            sns.barplot(x='Score', y='Feature', data=combined_df.head(k))
            plt.title(f'Top {k} Features (Combined Methods)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'combined_feature_scores.png'))
            plt.close()
        else:
            # Select top k features based on the specified method
            method_scores = feature_scores[method]
            selected_features = sorted(method_scores, key=method_scores.get, reverse=True)[:k]
        
        print(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
        return selected_features, feature_scores