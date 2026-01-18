"""
Dark Pattern Detector - Streamlit Web Application
Attractive UI for detecting dark patterns using trained ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from website_analyzer import WebsiteAnalyzer
from visualization_utils import SupervisedAlgorithmVisualizer
from screenshot_capture import PatternScreenshotCapture
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Dark Pattern Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive design
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dark-pattern-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .safe-pattern-alert {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model(model_name='Random Forest'):
    """Load trained model"""
    model_path = f"models/{model_name.lower().replace(' ', '_')}.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_scaler():
    """Load scaler"""
    scaler_path = "models/scaler.pkl"
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

def predict_dark_pattern(features, model, scaler, model_name):
    """Predict if pattern is dark pattern"""
    feature_names = [
        'urgency_language', 'hidden_costs', 'misleading_labels', 'forced_continuity',
        'roach_motel', 'trick_questions', 'sneak_into_basket', 'confirm_shaming',
        'disguised_ads', 'price_comparison_prevention', 'popup_frequency',
        'opt_out_difficulty', 'fake_reviews', 'social_proof_manipulation'
    ]
    
    feature_array = np.array([features[name] for name in feature_names]).reshape(1, -1)
    
    # Scale features if needed
    if model_name in ['Support Vector Machine', 'Logistic Regression']:
        feature_array = scaler.transform(feature_array)
    
    prediction = model.predict(feature_array)[0]
    probability = model.predict_proba(feature_array)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
    
    return prediction, probability[1]

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Dark Pattern Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning System for Detecting Deceptive UI Patterns</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_option = st.selectbox(
            "Select ML Model",
            ["Random Forest", "Support Vector Machine", "Logistic Regression", 
             "Naive Bayes", "Decision Tree"],
            help="Choose the supervised learning algorithm"
        )
        
        st.markdown("---")
        st.header("📊 About")
        st.markdown("""
        This tool uses **supervised machine learning** to detect dark patterns in user interfaces.
        
        **Supported Dark Patterns:**
        - Urgency Language
        - Hidden Costs
        - Misleading Labels
        - Forced Continuity
        - Roach Motel
        - Trick Questions
        - And more...
        """)
        
        st.markdown("---")
        st.header("📚 Model Info")
        if os.path.exists("models/best_model.txt"):
            with open("models/best_model.txt", "r") as f:
                st.text(f.read())
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Manual Detection", "🌐 Website Analysis", "📊 Model Analytics", "📈 Algorithm Graphs", "ℹ️ Information"])
    
    with tab1:
        st.header("Manual Pattern Detection")
        st.markdown("Adjust the sliders below to manually input pattern features for analysis.")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pattern Features")
            
            features = {
                'urgency_language': st.slider("Urgency Language", 0.0, 10.0, 5.0, 0.1,
                    help="Count of urgent/scarcity words (e.g., 'Limited time', 'Only 3 left')"),
                'hidden_costs': st.slider("Hidden Costs", 0.0, 5.0, 2.0, 0.1,
                    help="Indicators of hidden or surprise costs"),
                'misleading_labels': st.slider("Misleading Labels", 0.0, 8.0, 3.0, 0.1,
                    help="Confusing or deceptive UI labels"),
                'forced_continuity': st.slider("Forced Continuity", 0.0, 6.0, 2.0, 0.1,
                    help="Subscription traps or forced recurring charges"),
                'roach_motel': st.slider("Roach Motel", 0.0, 7.0, 2.0, 0.1,
                    help="Easy to sign up, hard to cancel"),
                'trick_questions': st.slider("Trick Questions", 0.0, 5.0, 1.0, 0.1,
                    help="Confusing options designed to mislead"),
                'sneak_into_basket': st.slider("Sneak into Basket", 0.0, 4.0, 1.0, 0.1,
                    help="Items automatically added to cart"),
                'confirm_shaming': st.slider("Confirm Shaming", 0.0, 6.0, 2.0, 0.1,
                    help="Guilt-inducing language on opt-out buttons"),
            }
        
        with col2:
            st.subheader("Additional Features")
            
            features.update({
                'disguised_ads': st.slider("Disguised Ads", 0.0, 5.0, 1.0, 0.1,
                    help="Ads presented as regular content"),
                'price_comparison_prevention': st.slider("Price Comparison Prevention", 0.0, 4.0, 1.0, 0.1,
                    help="Techniques to prevent price comparison"),
                'popup_frequency': st.slider("Popup Frequency", 0.0, 10.0, 3.0, 0.1,
                    help="Frequency of intrusive popups"),
                'opt_out_difficulty': st.slider("Opt-Out Difficulty", 0.0, 8.0, 2.0, 0.1,
                    help="Difficulty in opting out of services"),
                'fake_reviews': st.slider("Fake Reviews", 0.0, 7.0, 1.0, 0.1,
                    help="Indicators of fake or manipulated reviews"),
                'social_proof_manipulation': st.slider("Social Proof Manipulation", 0.0, 6.0, 1.0, 0.1,
                    help="Fake social proof indicators"),
            })
        
        # Predict button
        if st.button("🔍 Detect Pattern", type="primary"):
            model = load_model(model_option)
            scaler = load_scaler()
            
            if model is None or scaler is None:
                st.error("⚠️ Models not found! Please run model_trainer.py first.")
            else:
                prediction, probability = predict_dark_pattern(features, model, scaler, model_option)
                
                # Display result
                st.markdown("---")
                
                if prediction == 1:
                    st.markdown(
                        f'<div class="dark-pattern-alert">'
                        f'⚠️ DARK PATTERN DETECTED<br>'
                        f'Confidence: {probability*100:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="safe-pattern-alert">'
                        f'✅ SAFE PATTERN<br>'
                        f'Confidence: {(1-probability)*100:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Probability visualization
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Safe Pattern', 'Dark Pattern'],
                        y=[(1-probability)*100, probability*100],
                        marker_color=['#4facfe', '#f5576c'],
                        text=[f'{(1-probability)*100:.1f}%', f'{probability*100:.1f}%'],
                        textposition='auto',
                    )
                ])
                fig.update_layout(
                    title="Detection Confidence",
                    xaxis_title="Pattern Type",
                    yaxis_title="Probability (%)",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🌐 Website URL Analysis")
        st.markdown("Enter a website URL to automatically detect dark patterns using web scraping and analysis.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            website_url = st.text_input(
                "Enter Website URL",
                placeholder="https://example.com or example.com",
                help="Enter the full URL or domain name of the website to analyze"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            analyze_button = st.button("🔍 Analyze Website", type="primary", use_container_width=True)
        
        if analyze_button and website_url:
            if not website_url.strip():
                st.warning("⚠️ Please enter a valid website URL")
            else:
                with st.spinner("🔍 Analyzing website... This may take a few seconds"):
                    analyzer = WebsiteAnalyzer()
                    result = analyzer.analyze_url(website_url)
                    
                    if result['success']:
                        st.success(f"✅ Successfully analyzed: {result['url']}")
                        
                        # Display website info
                        with st.expander("📄 Website Information", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Title:** {result['title']}")
                                st.markdown(f"**URL:** {result['url']}")
                            with col2:
                                if result['description']:
                                    st.markdown(f"**Description:** {result['description']}")
                        
                        # Load model and predict
                        model = load_model(model_option)
                        scaler = load_scaler()
                        
                        if model is None or scaler is None:
                            st.error("⚠️ Models not found! Please run model_trainer.py first.")
                        else:
                            # Get features
                            features = result['features']
                            
                            # Display extracted features
                            with st.expander("📊 Extracted Features", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                
                                feature_items = list(features.items())
                                items_per_col = len(feature_items) // 3 + 1
                                
                                for i, (feature_name, value) in enumerate(feature_items):
                                    col_idx = i // items_per_col
                                    if col_idx == 0:
                                        col1.metric(feature_name.replace('_', ' ').title(), f"{value:.2f}")
                                    elif col_idx == 1:
                                        col2.metric(feature_name.replace('_', ' ').title(), f"{value:.2f}")
                                    else:
                                        col3.metric(feature_name.replace('_', ' ').title(), f"{value:.2f}")
                            
                            # Predict
                            prediction, probability = predict_dark_pattern(features, model, scaler, model_option)
                            
                            # Display result
                            st.markdown("---")
                            
                            if prediction == 1:
                                st.markdown(
                                    f'<div class="dark-pattern-alert">'
                                    f'⚠️ DARK PATTERN DETECTED<br>'
                                    f'Confidence: {probability*100:.1f}%<br>'
                                    f'Website: {result["url"]}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div class="safe-pattern-alert">'
                                    f'✅ SAFE PATTERN<br>'
                                    f'Confidence: {(1-probability)*100:.1f}%<br>'
                                    f'Website: {result["url"]}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Probability visualization
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=['Safe Pattern', 'Dark Pattern'],
                                    y=[(1-probability)*100, probability*100],
                                    marker_color=['#4facfe', '#f5576c'],
                                    text=[f'{(1-probability)*100:.1f}%', f'{probability*100:.1f}%'],
                                    textposition='auto',
                                )
                            ])
                            fig.update_layout(
                                title=f"Detection Confidence for {result['url']}",
                                xaxis_title="Pattern Type",
                                yaxis_title="Probability (%)",
                                height=400,
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature breakdown chart
                            st.subheader("🔍 Feature Analysis Breakdown")
                            feature_df = pd.DataFrame({
                                'Feature': [f.replace('_', ' ').title() for f in features.keys()],
                                'Value': list(features.values()),
                                'Status': ['High Risk' if v > 5 else 'Medium Risk' if v > 2 else 'Low Risk' for v in features.values()]
                            })
                            
                            fig = px.bar(
                                feature_df,
                                x='Feature',
                                y='Value',
                                color='Status',
                                color_discrete_map={
                                    'High Risk': '#f5576c',
                                    'Medium Risk': '#ffa726',
                                    'Low Risk': '#4facfe'
                                },
                                title="Dark Pattern Features Detected",
                                labels={'Value': 'Risk Score', 'Feature': 'Dark Pattern Feature'}
                            )
                            fig.update_layout(
                                xaxis_tickangle=-45,
                                height=500,
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display captured evidence (screenshots of fake reviews, items, etc.)
                            st.markdown("---")
                            st.subheader("📸 Visual Evidence of Detected Patterns")
                            
                            evidence = result.get('evidence', {})
                            
                            # Fake Reviews Evidence
                            if evidence.get('fake_reviews', {}).get('screenshots'):
                                with st.expander("⚠️ Fake Reviews Detected - Visual Evidence", expanded=True):
                                    for evidence_item in evidence['fake_reviews']['screenshots']:
                                        st.image(f"data:image/png;base64,{evidence_item['base64']}", 
                                                caption="Fake Review Detection Evidence", use_container_width=True)
                                        if evidence_item.get('elements'):
                                            st.markdown("**Detected Review Elements:**")
                                            for i, elem in enumerate(evidence_item['elements'][:5], 1):
                                                st.markdown(f"{i}. {elem.get('text', 'N/A')[:100]}...")
                            
                            # Sneak into Basket Evidence
                            if evidence.get('sneak_into_basket', {}).get('screenshots'):
                                with st.expander("⚠️ Auto-Added Items Detected - Visual Evidence", expanded=True):
                                    for evidence_item in evidence['sneak_into_basket']['screenshots']:
                                        st.image(f"data:image/png;base64,{evidence_item['base64']}", 
                                                caption="Auto-Added Items Detection Evidence", use_container_width=True)
                                        if evidence_item.get('elements'):
                                            st.markdown("**Detected Auto-Added Items:**")
                                            for i, elem in enumerate(evidence_item['elements'][:5], 1):
                                                st.markdown(f"{i}. ✓ {elem.get('name', 'N/A')}")
                            
                            # Confirm Shaming Evidence
                            if evidence.get('confirm_shaming', {}).get('screenshots'):
                                with st.expander("⚠️ Confirm Shaming Detected - Visual Evidence", expanded=True):
                                    for evidence_item in evidence['confirm_shaming']['screenshots']:
                                        st.image(f"data:image/png;base64,{evidence_item['base64']}", 
                                                caption="Confirm Shaming Detection Evidence", use_container_width=True)
                                        if evidence_item.get('elements'):
                                            st.markdown("**Detected Shaming Text:**")
                                            for i, text in enumerate(evidence_item['elements'][:5], 1):
                                                st.markdown(f'{i}. Button: "{text}"')
                            
                            # Export Report
                            st.markdown("---")
                            st.subheader("📄 Export Analysis Report")
                            
                            report_data = {
                                'website_url': result['url'],
                                'website_title': result['title'],
                                'prediction': 'Dark Pattern' if prediction == 1 else 'Safe Pattern',
                                'confidence': probability * 100 if prediction == 1 else (1-probability) * 100,
                                'model_used': model_option,
                                'features': features,
                                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Create report text
                            report_text = f"""
# Dark Pattern Detection Report

## Website Information
- **URL**: {report_data['website_url']}
- **Title**: {report_data['website_title']}
- **Analysis Date**: {report_data['analysis_date']}

## Detection Results
- **Classification**: {report_data['prediction']}
- **Confidence**: {report_data['confidence']:.2f}%
- **Model Used**: {report_data['model_used']}

## Feature Analysis
"""
                            for feature, value in features.items():
                                report_text += f"- **{feature.replace('_', ' ').title()}**: {value:.2f}\n"
                            
                            st.download_button(
                                label="📥 Download Report (TXT)",
                                data=report_text,
                                file_name=f"dark_pattern_report_{result['url'].replace('https://', '').replace('http://', '').replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error(f"❌ Error analyzing website: {result.get('error', 'Unknown error')}")
                        st.info("💡 Tips:\n- Make sure the URL is correct and accessible\n- Some websites may block automated access\n- Try adding https:// or http:// prefix")
        
        elif analyze_button:
            st.warning("⚠️ Please enter a website URL to analyze")
        
        # Instructions
        with st.expander("ℹ️ How Website Analysis Works"):
            st.markdown("""
            **Website Analysis Process:**
            1. **Web Scraping**: The system fetches the website's HTML content
            2. **Feature Extraction**: Analyzes the content for 14 different dark pattern indicators:
               - Urgency language and scarcity tactics
               - Hidden costs and fees
               - Misleading UI elements
               - Subscription traps
               - Popup frequency
               - And more...
            3. **ML Prediction**: Uses trained supervised learning models to classify the website
            4. **Results**: Displays confidence scores and detailed feature breakdown
            
            **Supported Features:**
            - Urgency/scarcity language detection
            - Hidden cost identification
            - Subscription trap detection
            - Popup and overlay analysis
            - Confirm shaming detection
            - Fake review indicators
            - Social proof manipulation
            
            **Note**: This tool analyzes publicly accessible HTML content. Some JavaScript-heavy sites
            may require additional processing. The analysis is based on content patterns and cannot
            guarantee 100% accuracy.
            """)
    
    with tab3:
        st.header("Model Analytics")
        
        if os.path.exists("data/dark_patterns_dataset.csv"):
            df = pd.read_csv("data/dark_patterns_dataset.csv")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Dark Patterns", df['is_dark_pattern'].sum())
            with col3:
                st.metric("Safe Patterns", (df['is_dark_pattern'] == 0).sum())
            
            # Feature distribution
            st.subheader("Feature Distribution")
            selected_feature = st.selectbox("Select Feature", df.columns[:-1])
            
            fig = px.histogram(
                df, x=selected_feature, color='is_dark_pattern',
                color_discrete_map={0: '#4facfe', 1: '#f5576c'},
                labels={'is_dark_pattern': 'Pattern Type'},
                title=f"Distribution of {selected_feature}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlation")
            corr_matrix = df.corr()
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Run model_trainer.py to generate dataset and view analytics.")
    
    with tab4:
        st.header("📈 Supervised Algorithm Performance Graphs")
        st.markdown("Professional visualizations of supervised learning algorithm performance")
        
        # Load model results if available
        if os.path.exists("data/dark_patterns_dataset.csv"):
            # Load dataset and models
            df = pd.read_csv("data/dark_patterns_dataset.csv")
            X = df.drop('is_dark_pattern', axis=1)
            y = df['is_dark_pattern']
            
            # Prepare test data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Load scaler
            scaler = load_scaler()
            if scaler:
                X_test_scaled = scaler.transform(X_test)
                X_train_scaled = scaler.transform(X_train)
            else:
                X_test_scaled = X_test
                X_train_scaled = X_train
            
            visualizer = SupervisedAlgorithmVisualizer()
            
            # Model selection for detailed analysis
            selected_model_name = st.selectbox(
                "Select Model for Detailed Analysis",
                ["Random Forest", "Support Vector Machine", "Logistic Regression", 
                 "Naive Bayes", "Decision Tree"],
                help="Choose a model to view detailed performance graphs"
            )
            
            # Load selected model
            model = load_model(selected_model_name)
            
            if model:
                # Determine if scaled data is needed
                if selected_model_name in ['Support Vector Machine', 'Logistic Regression']:
                    X_test_model = X_test_scaled
                    X_train_model = X_train_scaled
                else:
                    X_test_model = X_test
                    X_train_model = X_train
                
                # Get predictions
                y_pred = model.predict(X_test_model)
                y_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Tabs for different visualizations
                graph_tab1, graph_tab2, graph_tab3, graph_tab4, graph_tab5 = st.tabs([
                    "📊 Confusion Matrix", "📈 ROC Curve", "🎯 Precision-Recall", 
                    "🌳 Feature Importance", "📉 Learning Curve"
                ])
                
                with graph_tab1:
                    st.subheader(f"{selected_model_name} - Confusion Matrix")
                    normalize = st.checkbox("Normalize", value=False)
                    fig_cm = visualizer.plot_confusion_matrix(y_test, y_pred, selected_model_name, normalize)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Calculate metrics
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2:
                        st.metric("Precision", f"{precision:.4f}")
                    with col3:
                        st.metric("Recall", f"{recall:.4f}")
                    with col4:
                        st.metric("F1-Score", f"{f1:.4f}")
                
                with graph_tab2:
                    if y_proba is not None:
                        st.subheader(f"{selected_model_name} - ROC Curve")
                        fig_roc, roc_auc = visualizer.plot_roc_curve(y_test, y_proba, selected_model_name)
                        st.plotly_chart(fig_roc, use_container_width=True)
                        st.metric("AUC Score", f"{roc_auc:.4f}")
                    else:
                        st.info("ROC curve not available for this model")
                
                with graph_tab3:
                    if y_proba is not None:
                        st.subheader(f"{selected_model_name} - Precision-Recall Curve")
                        fig_pr, pr_auc = visualizer.plot_precision_recall_curve(y_test, y_proba, selected_model_name)
                        st.plotly_chart(fig_pr, use_container_width=True)
                        st.metric("PR AUC Score", f"{pr_auc:.4f}")
                    else:
                        st.info("Precision-Recall curve not available for this model")
                
                with graph_tab4:
                    if hasattr(model, 'feature_importances_'):
                        st.subheader(f"{selected_model_name} - Feature Importance")
                        importances = model.feature_importances_
                        feature_names = X.columns.tolist()
                        fig_fi = visualizer.plot_feature_importance(feature_names, importances, selected_model_name)
                        st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model type")
                
                with graph_tab5:
                    st.subheader(f"{selected_model_name} - Learning Curve")
                    try:
                        fig_lc = visualizer.plot_learning_curve(model, X_train_model, y_train, selected_model_name)
                        st.plotly_chart(fig_lc, use_container_width=True)
                    except Exception as e:
                        st.info(f"Learning curve generation may take time. Error: {str(e)}")
                
                # Model Comparison
                st.markdown("---")
                st.subheader("🔍 Model Comparison Dashboard")
                
                if st.button("Generate All Model Comparisons", type="primary"):
                    with st.spinner("Loading all models and generating comparisons..."):
                        all_models_results = {}
                        models_to_compare = ["Random Forest", "Support Vector Machine", "Logistic Regression", 
                                            "Naive Bayes", "Decision Tree"]
                        
                        for model_name in models_to_compare:
                            m = load_model(model_name)
                            if m:
                                if model_name in ['Support Vector Machine', 'Logistic Regression']:
                                    X_test_m = X_test_scaled
                                else:
                                    X_test_m = X_test
                                
                                y_pred_m = m.predict(X_test_m)
                                y_proba_m = m.predict_proba(X_test_m)[:, 1] if hasattr(m, 'predict_proba') else None
                                
                                from sklearn.metrics import classification_report, accuracy_score
                                report = classification_report(y_test, y_pred_m, output_dict=True)
                                
                                all_models_results[model_name] = {
                                    'model': m,
                                    'accuracy': accuracy_score(y_test, y_pred_m),
                                    'y_pred': y_pred_m,
                                    'y_proba': y_proba_m,
                                    'classification_report': report
                                }
                        
                        if all_models_results:
                            fig_comparison = visualizer.plot_model_comparison(all_models_results)
                            st.plotly_chart(fig_comparison, use_container_width=True)
                            
                            # Comparison table
                            comparison_df = pd.DataFrame({
                                'Model': list(all_models_results.keys()),
                                'Accuracy': [r['accuracy'] for r in all_models_results.values()],
                                'Precision': [r['classification_report'].get('1', {}).get('precision', 0) 
                                            for r in all_models_results.values()],
                                'Recall': [r['classification_report'].get('1', {}).get('recall', 0) 
                                         for r in all_models_results.values()],
                                'F1-Score': [r['classification_report'].get('1', {}).get('f1-score', 0) 
                                           for r in all_models_results.values()]
                            })
                            st.dataframe(comparison_df.style.highlight_max(axis=0), use_container_width=True)
            else:
                st.error("⚠️ Models not found! Please run model_trainer.py first.")
        else:
            st.info("📊 Run model_trainer.py to generate dataset and view algorithm performance graphs.")
    
    with tab5:
        st.header("About Dark Pattern Detection")
        
        st.markdown("""
        ### What are Dark Patterns?
        Dark patterns are deceptive UI/UX designs that trick users into doing things they didn't intend to do.
        These patterns exploit human psychology and cognitive biases.
        
        ### Types of Dark Patterns Detected:
        
        1. **Urgency Language** - Creating false scarcity or time pressure
        2. **Hidden Costs** - Concealing additional fees until the last moment
        3. **Misleading Labels** - Confusing button labels or options
        4. **Forced Continuity** - Subscription traps that are hard to cancel
        5. **Roach Motel** - Easy to sign up, nearly impossible to cancel
        6. **Trick Questions** - Confusing options designed to mislead
        7. **Sneak into Basket** - Items automatically added to cart
        8. **Confirm Shaming** - Guilt-inducing language on opt-out buttons
        9. **Disguised Ads** - Ads presented as regular content
        10. **Price Comparison Prevention** - Techniques to prevent price comparison
        
        ### Machine Learning Approach:
        This system uses **supervised learning** algorithms including:
        - Random Forest Classifier
        - Support Vector Machine (SVM)
        - Logistic Regression
        - Naive Bayes
        - Decision Tree
        
        The models are trained on a comprehensive dataset of dark pattern features and can classify
        UI patterns with high accuracy.
        """)
        
        st.markdown("---")
        st.header("How to Use")
        st.markdown("""
        ### Option 1: Manual Detection
        1. Go to the **Manual Detection** tab
        2. Adjust the feature sliders based on the pattern you want to analyze
        3. Click **Detect Pattern** to get results
        4. View confidence scores and probability distribution
        
        ### Option 2: Website URL Analysis
        1. Go to the **Website Analysis** tab
        2. Enter a website URL (e.g., https://example.com)
        3. Click **Analyze Website** to automatically detect dark patterns
        4. View extracted features and ML prediction results
        5. Explore detailed feature breakdown and risk analysis
        
        ### Option 3: Model Analytics Dashboard
        - Explore dataset statistics in the **Model Analytics** tab
        - View feature distributions and correlations
        - Analyze model performance metrics
        
        ### Option 4: Supervised Algorithm Graphs
        - View detailed performance graphs for each supervised algorithm
        - Explore confusion matrices, ROC curves, precision-recall curves
        - Compare all supervised models side by side
        - Analyze feature importance and learning curves
        """)

if __name__ == "__main__":
    main()
