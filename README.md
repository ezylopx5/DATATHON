# 🏎️ GDGC Datathon 2025 - Formula Racing Lap Time Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/Haxxsh/gdgc-datathon-data)
[![Models](https://img.shields.io/badge/Models-HuggingFace-orange)](https://huggingface.co/Haxxsh/gdgc-datathon-models)

## 🏆 Competition Overview

Machine learning solution for predicting Formula racing lap times using historical data from F1, F2, and F3 races (1949-2021). This solution uses an ensemble of Random Forest and XGBoost models with comprehensive feature engineering.

## 📊 Dataset

- **Source**: [HuggingFace Dataset](https://huggingface.co/datasets/Haxxsh/gdgc-datathon-data)
- **Size**: 734,002 racing records
- **Features**: 36 original features → 200+ engineered features
- **Target**: Lap Time in seconds

### Quick Data Access:
```python
from datasets import load_dataset

# Load dataset from HuggingFace
dataset = load_dataset("Haxxsh/gdgc-datathon-data")
train_data = dataset['train']
test_data = dataset['test']
```

## 🤖 Pre-trained Models

Access our trained models on [HuggingFace](https://huggingface.co/Haxxsh/gdgc-datathon-models)

```python
from huggingface_hub import hf_hub_download
import joblib

# Download pre-trained models
rf_model = joblib.load(
    hf_hub_download(repo_id="Haxxsh/gdgc-datathon-models", 
                    filename="rf_final.pkl")
)
xgb_model = joblib.load(
    hf_hub_download(repo_id="Haxxsh/gdgc-datathon-models", 
                    filename="xgb_final.pkl")
)
```

## 🚀 Features

### Advanced Feature Engineering (200+ features)
- **Driver Psychology & Pressure** - Performance under championship pressure
- **Team Dynamics** - Driver vs teammate comparisons
- **Circuit Mastery** - Track-specific performance patterns
- **Weather Adaptability** - Performance in changing conditions
- **Technical Indicators** - Tire strategy, fuel management
- **Momentum Detection** - Form trends and breakthrough indicators

### Model Architecture
- Random Forest (500 trees, optimized depth)
- XGBoost (2000 rounds, GPU accelerated)
- 5-Fold Cross-Validation
- Ensemble Learning (weighted by CV performance)

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/ezylopx5/DATATHON.git
cd DATATHON

# Install requirements
pip install -r requirements.txt
```

## 🎯 Usage

### Training Models
```bash
python train.py
```

### Making Predictions
```bash
python predict.py
```

## 🏗️ Project Structure

```
DATATHON/
├── features.py          # Advanced feature engineering
├── train.py            # Model training pipeline
├── predict.py          # Prediction pipeline
├── requirements.txt    # Project dependencies
├── submissions/        # Prediction outputs
│   ├── submission.csv
│   └── submission_cv_a100.csv
└── README.md
```

## 📈 Key Feature Categories

1. **Team Dynamics** - Driver vs teammate features are among the most important
2. **Pressure Performance** - Championship pressure significantly impacts lap times
3. **Circuit Familiarity** - Track-specific experience is a top predictor
4. **Weather Adaptability** - Drivers who adapt to conditions perform consistently
5. **Momentum Indicators** - Recent form and breakthrough patterns

## 🙏 Acknowledgments

- GDGC Datathon 2025 organizers
- HuggingFace for hosting datasets and models
