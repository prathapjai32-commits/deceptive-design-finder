"""
Dark Pattern Detector - Model Training Script
Implements multiple supervised learning algorithms for detecting dark patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class DarkPatternTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        
    def load_data(self, file_path='data/dark_patterns_dataset.csv'):
        """Load the dark patterns dataset"""
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df
        else:
            # Generate synthetic data if file doesn't exist
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic dark pattern data for demonstration"""
        np.random.seed(42)
        
        # Features that indicate dark patterns
        # Higher values indicate more deceptive patterns
        data = {
            'urgency_language': np.random.rand(n_samples) * 10,  # Count of urgency words
            'hidden_costs': np.random.rand(n_samples) * 5,  # Hidden cost indicators
            'misleading_labels': np.random.rand(n_samples) * 8,  # Misleading UI elements
            'forced_continuity': np.random.rand(n_samples) * 6,  # Subscription traps
            'roach_motel': np.random.rand(n_samples) * 7,  # Easy in, hard out
            'trick_questions': np.random.rand(n_samples) * 5,  # Confusing options
            'sneak_into_basket': np.random.rand(n_samples) * 4,  # Auto-added items
            'confirm_shaming': np.random.rand(n_samples) * 6,  # Guilt-inducing language
            'disguised_ads': np.random.rand(n_samples) * 5,  # Ads as content
            'price_comparison_prevention': np.random.rand(n_samples) * 4,
            'popup_frequency': np.random.rand(n_samples) * 10,
            'opt_out_difficulty': np.random.rand(n_samples) * 8,
            'fake_reviews': np.random.rand(n_samples) * 7,
            'social_proof_manipulation': np.random.rand(n_samples) * 6,
        }
        
        df = pd.DataFrame(data)
        
        # Create target: Dark pattern if weighted sum > threshold
        weights = np.array([0.15, 0.12, 0.10, 0.08, 0.10, 0.08, 0.06, 0.08, 
                           0.07, 0.05, 0.05, 0.03, 0.02, 0.01])
        weighted_sum = (df.values * weights).sum(axis=1)
        threshold = np.percentile(weighted_sum, 60)  # 40% dark patterns
        
        df['is_dark_pattern'] = (weighted_sum > threshold).astype(int)
        
        # Add noise
        noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        df.loc[noise_indices, 'is_dark_pattern'] = 1 - df.loc[noise_indices, 'is_dark_pattern']
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        X = df.drop('is_dark_pattern', axis=1)
        y = df['is_dark_pattern']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
        """Train multiple supervised learning models"""
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for SVM and Logistic Regression
            if name in ['Support Vector Machine', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Track best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = name
        
        self.models = results
        return results
    
    def save_models(self, directory='models'):
        """Save trained models"""
        os.makedirs(directory, exist_ok=True)
        
        for name, result in self.models.items():
            model = result['model']
            filename = os.path.join(directory, f"{name.lower().replace(' ', '_')}.pkl")
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(directory, 'scaler.pkl'))
        print(f"Saved scaler to {directory}/scaler.pkl")
        
        # Save best model info
        with open(os.path.join(directory, 'best_model.txt'), 'w') as f:
            f.write(f"Best Model: {self.best_model}\n")
            f.write(f"Best Accuracy: {self.best_score:.4f}\n")
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get feature importance from tree-based models"""
        if model_name in self.models:
            model = self.models[model_name]['model']
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
        return None

def main():
    print("="*60)
    print("Dark Pattern Detector - Model Training")
    print("="*60)
    
    trainer = DarkPatternTrainer()
    
    # Load or generate data
    print("\nLoading dataset...")
    df = trainer.load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Dark patterns: {df['is_dark_pattern'].sum()} ({df['is_dark_pattern'].mean()*100:.1f}%)")
    
    # Save dataset if it was generated
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/dark_patterns_dataset.csv'):
        df.to_csv('data/dark_patterns_dataset.csv', index=False)
        print("Generated synthetic dataset saved to data/dark_patterns_dataset.csv")
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = trainer.prepare_data(df)
    
    # Train models
    print("\nTraining models...")
    results = trainer.train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # Display results
    print("\n" + "="*60)
    print("Model Performance Summary")
    print("="*60)
    for name, result in results.items():
        print(f"{name:25s}: {result['accuracy']:.4f}")
    
    print(f"\nBest Model: {trainer.best_model} (Accuracy: {trainer.best_score:.4f})")
    
    # Save models
    print("\nSaving models...")
    trainer.save_models()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
