# 🔍 Dark Pattern Detector

A sophisticated machine learning system for detecting deceptive UI patterns (dark patterns) using supervised learning algorithms. This project features an attractive web interface built with Streamlit and supports multiple ML models.

## 📋 Project Overview

Dark patterns are deceptive user interface designs that trick users into performing actions they didn't intend. This project uses **supervised machine learning** to automatically detect these patterns based on various features extracted from UI elements.

## ✨ Features

- **Multiple Supervised Algorithms**: Random Forest, SVM, Logistic Regression, Naive Bayes, Decision Tree
- **Attractive Web UI**: Modern, gradient-based design with interactive visualizations
- **Website URL Analysis**: Automatically analyze any website by entering its URL
- **Web Scraping**: Advanced HTML parsing to extract dark pattern features
- **Manual Detection**: Slider-based interface for manual pattern analysis
- **Real-time Detection**: Instant pattern classification with confidence scores
- **Analytics Dashboard**: Feature distributions, correlations, and model performance metrics
- **Feature Breakdown**: Detailed visualization of detected dark pattern indicators

## 🚀 Quick Start

### Installation

1. Clone or download this project
2. Install required packages:

```bash
pip install -r requirements.txt
```

### Training Models

Train the machine learning models using the training script:

```bash
python model_trainer.py
```

This will:
- Generate a synthetic dataset (or load existing one)
- Train multiple supervised learning models
- Evaluate and compare model performance
- Save trained models to the `models/` directory

### Running the Web Application

Start the Streamlit web app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Usage

### Option 1: Website URL Analysis (Recommended)

1. Navigate to the **Website Analysis** tab
2. Enter a website URL (e.g., `https://example.com` or `example.com`)
3. Click **Analyze Website** to automatically detect dark patterns
4. View extracted features, confidence scores, and detailed breakdown
5. Explore feature-specific risk analysis

### Option 2: Manual Pattern Detection

1. Navigate to the **Manual Detection** tab
2. Adjust the feature sliders to match the UI pattern you want to analyze
3. Click **Detect Pattern** to get instant results
4. View the confidence score and probability distribution

### Model Selection

Choose from different supervised learning models in the sidebar:
- **Random Forest**: Ensemble method with high accuracy
- **Support Vector Machine**: Good for complex pattern separation
- **Logistic Regression**: Interpretable linear classifier
- **Naive Bayes**: Fast probabilistic classifier
- **Decision Tree**: Interpretable tree-based model

### Website Analysis Features

The website analyzer extracts 14 dark pattern features automatically:
- Urgency/scarcity language detection
- Hidden cost identification  
- Subscription trap detection
- Popup and overlay analysis
- Confirm shaming detection
- Fake review indicators
- Social proof manipulation
- And more...

### Analytics

Explore the dataset in the **Analytics** tab:
- Feature distributions
- Correlation heatmaps
- Dataset statistics

## 🧠 Machine Learning Details

### Features Detected

The system analyzes 14 key features:

1. Urgency Language
2. Hidden Costs
3. Misleading Labels
4. Forced Continuity
5. Roach Motel
6. Trick Questions
7. Sneak into Basket
8. Confirm Shaming
9. Disguised Ads
10. Price Comparison Prevention
11. Popup Frequency
12. Opt-Out Difficulty
13. Fake Reviews
14. Social Proof Manipulation

### Model Performance

Models are evaluated using:
- Accuracy score
- Classification report
- Cross-validation
- Confusion matrix

The best performing model is automatically selected and saved.

## 📁 Project Structure

```
dark-pattern-detector/
├── app.py                      # Streamlit web application
├── model_trainer.py            # ML model training script
├── website_analyzer.py         # Website scraping and analysis module
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── models/                     # Saved trained models (generated)
│   ├── random_forest.pkl
│   ├── support_vector_machine.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── decision_tree.pkl
│   ├── scaler.pkl
│   └── best_model.txt
└── data/                       # Dataset (generated)
    └── dark_patterns_dataset.csv
```

## 🎨 UI Features

- **Modern Design**: Gradient-based color scheme with smooth animations
- **Interactive Visualizations**: Plotly charts for data exploration
- **Responsive Layout**: Works on different screen sizes
- **Intuitive Interface**: Easy-to-use sliders and buttons

## 🔬 Technical Details

### Supervised Learning Algorithms

1. **Random Forest**: Ensemble of decision trees, handles non-linear relationships well
2. **SVM**: Uses RBF kernel for complex pattern classification
3. **Logistic Regression**: Linear model with regularization
4. **Naive Bayes**: Based on Bayes' theorem with feature independence assumption
5. **Decision Tree**: Rule-based classifier with interpretable decisions

### Data Preprocessing

- Feature scaling (StandardScaler) for SVM and Logistic Regression
- Train-test split (80-20) with stratification
- Synthetic data generation for demonstration purposes

## 📈 Results

After training, you'll see:
- Accuracy scores for all models
- Best model identification
- Detailed classification reports
- Saved models for prediction

## 🎓 Academic Use

This project is suitable for:
- Machine Learning courses
- Human-Computer Interaction research
- UI/UX analysis projects
- Ethical design studies

## 🛠️ Customization

### Adding Real Data

Replace the synthetic data generation in `model_trainer.py` with your own dataset:

```python
def load_data(self, file_path='data/your_dataset.csv'):
    df = pd.read_csv(file_path)
    return df
```

### Adding New Models

Extend the `models_config` dictionary in `model_trainer.py`:

```python
models_config['Your Model'] = YourModelClass()
```

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created for academic demonstration and research purposes.

---

**Note**: This project uses synthetic data for demonstration. For production use, train on real-world dark pattern datasets.
