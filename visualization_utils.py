"""
Professional Visualization Utils for Supervised Learning Algorithms
Creates comprehensive graphs and charts for model analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
from sklearn.model_selection import learning_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class SupervisedAlgorithmVisualizer:
    """Professional visualizations for supervised learning models"""
    
    def __init__(self):
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#4facfe',
            'danger': '#f5576c',
            'warning': '#ffa726',
            'info': '#00f2fe'
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, normalize=False):
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = f'{model_name} - Normalized Confusion Matrix'
            fmt = '.2%'
        else:
            title = f'{model_name} - Confusion Matrix'
            fmt = 'd'
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Safe Pattern', 'Dark Pattern'],
            y=['Safe Pattern', 'Dark Pattern'],
            colorscale=[[0, '#4facfe'], [1, '#f5576c']],
            text=cm if not normalize else cm.round(2),
            texttemplate='%{text}',
            textfont={"size": 20},
            colorbar=dict(title="Count" if not normalize else "Proportion")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=500,
            template="plotly_white",
            font=dict(size=14)
        )
        
        return fig
    
    def plot_roc_curve(self, y_true, y_proba, model_name):
        """Plot ROC curve with AUC score"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            line=dict(width=3, color=self.colors['primary']),
            fill='tonexty'
        ))
        
        # Diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray', width=2)
        ))
        
        fig.update_layout(
            title=f'{model_name} - ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            template="plotly_white",
            hovermode='x unified',
            font=dict(size=14),
            showlegend=True
        )
        
        return fig, roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_proba, model_name):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{model_name} (AUC = {pr_auc:.3f})',
            line=dict(width=3, color=self.colors['danger']),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f'{model_name} - Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            template="plotly_white",
            font=dict(size=14)
        )
        
        return fig, pr_auc
    
    def plot_feature_importance(self, feature_names, importances, model_name):
        """Plot feature importance (for tree-based models)"""
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_importances,
                y=sorted_features,
                orientation='h',
                marker=dict(
                    color=sorted_importances,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f'{imp:.4f}' for imp in sorted_importances],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f'{model_name} - Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=600,
            template="plotly_white",
            font=dict(size=12),
            yaxis=dict(autorange="reversed")
        )
        
        return fig
    
    def plot_learning_curve(self, model, X, y, model_name, cv=5):
        """Plot learning curve showing train/test performance vs sample size"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            error_y=dict(type='data', array=train_std),
            mode='lines+markers',
            name='Training Score',
            line=dict(width=3, color=self.colors['success']),
            marker=dict(size=8)
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            error_y=dict(type='data', array=val_std),
            mode='lines+markers',
            name='Validation Score',
            line=dict(width=3, color=self.colors['danger']),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'{model_name} - Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='Accuracy Score',
            height=500,
            template="plotly_white",
            hovermode='x unified',
            font=dict(size=14),
            showlegend=True
        )
        
        return fig
    
    def plot_model_comparison(self, model_results):
        """Compare all supervised models side by side"""
        model_names = list(model_results.keys())
        accuracies = [results['accuracy'] for results in model_results.values()]
        
        # Get classification reports
        precisions = []
        recalls = []
        f1_scores = []
        
        for results in model_results.values():
            report = results.get('classification_report', {})
            if '1' in report:
                precisions.append(report['1']['precision'])
                recalls.append(report['1']['recall'])
                f1_scores.append(report['1']['f1-score'])
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'Precision Comparison', 
                          'Recall Comparison', 'F1-Score Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Accuracy
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='Accuracy',
                  marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # Precision
        fig.add_trace(
            go.Bar(x=model_names, y=precisions, name='Precision',
                  marker_color=self.colors['success']),
            row=1, col=2
        )
        
        # Recall
        fig.add_trace(
            go.Bar(x=model_names, y=recalls, name='Recall',
                  marker_color=self.colors['warning']),
            row=2, col=1
        )
        
        # F1-Score
        fig.add_trace(
            go.Bar(x=model_names, y=f1_scores, name='F1-Score',
                  marker_color=self.colors['danger']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Supervised Algorithm Performance Comparison',
            height=800,
            template="plotly_white",
            showlegend=False,
            font=dict(size=12)
        )
        
        return fig
    
    def plot_feature_correlation_heatmap(self, df):
        """Plot correlation heatmap of features"""
        corr = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmid=0,
            text=corr.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            height=700,
            template="plotly_white",
            xaxis_title="Features",
            yaxis_title="Features",
            font=dict(size=11)
        )
        
        return fig
    
    def plot_classification_distribution(self, y_true, y_pred, model_name):
        """Plot distribution of predictions vs actual"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Actual Distribution', 'Predicted Distribution'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}]]
        )
        
        # Actual
        fig.add_trace(
            go.Histogram(x=y_true, name='Actual', 
                        marker_color=self.colors['primary'],
                        nbinsx=2),
            row=1, col=1
        )
        
        # Predicted
        fig.add_trace(
            go.Histogram(x=y_pred, name='Predicted',
                        marker_color=self.colors['danger'],
                        nbinsx=2),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'{model_name} - Classification Distribution',
            height=400,
            template="plotly_white",
            showlegend=False,
            font=dict(size=14)
        )
        
        return fig
    
    def create_model_performance_dashboard(self, model_results, X_test, y_test):
        """Create comprehensive dashboard for all models"""
        # This will be called from the main app
        return model_results
