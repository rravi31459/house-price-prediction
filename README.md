# Valuate: Premium House Price Prediction Engine

[![Python Version](https://img.shields.kr/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.kr/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Flask](https://img.shields.kr/badge/framework-Flask-lightgrey)](https://flask.palletsprojects.com/)
[![ML Library: Scikit--Learn](https://img.shields.kr/badge/library-scikit--learn-orange)](https://scikit-learn.org/)

Valuate is a production-quality, modular Machine Learning application designed to predict house prices using key numerical specifications: Square Footage, Number of Bedrooms, and Number of Bathrooms. It features a complete end-to-end data cleaning, preprocessing, OLS estimation, and model registry pipeline, wrapped inside a premium, minimalist SaaS-inspired Flask web dashboard.

---

## Architecture Diagram

The system separates raw ingestion schemas from internal modeling components to support dataset changes with zero code modifications:

```mermaid
flowchart TD
    subgraph Storage [Persistent Storage]
        CSV[data/train.csv]
        ModelRegistry[models/model_latest.joblib]
        MetadataRegistry[models/metadata_latest.json]
        HistoryStore[data/prediction_history.json]
    end

    subgraph DataPipeline [Data & Modeling Pipeline]
        DataLoader[src/data_loader.py]
        FeatureEng[src/feature_engineering.py]
        Preprocessor[src/preprocessing.py]
        TrainPipeline[src/train.py]
        Evaluator[src/evaluate.py]
    end

    subgraph Server [Flask Web Application]
        AppCtrl[app.py]
        Predictor[src/predict.py]
        PDFGen[src/utils.py]
    end

    subgraph Frontend [User Interface]
        HTML[HTML5 Templates]
        CSS[Vanilla CSS Grid/Flex]
        JS[Main Javascript]
    end

    CSV --> DataLoader
    DataLoader --> |Align Columns via config.py| FeatureEng
    FeatureEng --> |Clean & Remove Outliers| Preprocessor
    Preprocessor --> |Impute & Scale| TrainPipeline
    TrainPipeline --> |Save Joblib Pipeline| ModelRegistry
    TrainPipeline --> |Write stats & params| MetadataRegistry
    TrainPipeline --> |Calculate metrics & plots| Evaluator

    ModelRegistry --> Predictor
    MetadataRegistry --> AppCtrl
    Predictor --> AppCtrl
    AppCtrl --> |Render stats, coefficients, plots| HTML
    AppCtrl --> |Async predictions API| JS
    AppCtrl --> |PDF Generation Call| PDFGen
    PDFGen --> |Download Valuation PDF| HTML
    AppCtrl --> |Append logs| HistoryStore
```

---

## Folder Structure

```
house-price-prediction/
│
├── app.py                      # Flask Application entrypoint
├── config.py                   # Configuration and column mappings
├── requirements.txt            # Python package dependencies
├── Procfile                    # Deployment command file for Render/Railway
├── runtime.txt                 # Python runtime version definition
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── README.md                   # Premium documentation
│
├── data/                       # Raw CSVs & runtime logs
│   ├── train.csv
│   └── test.csv
│
├── models/                     # Registry of serialized pipeline runs
│   ├── model_latest.joblib     # Active fitted pipeline copy
│   └── metadata_latest.json    # Performance metrics & OLS coefficients
│
├── notebooks/                  # Professional Jupyter notebooks
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis Report
│   ├── 02_Preprocessing.ipynb  # Transformation checks
│   └── 03_Model_Training.ipynb # Linear Regression parameter fitting
│
├── src/                        # Core ML Engine package
│   ├── __init__.py
│   ├── data_loader.py          # Column standardization mapping
│   ├── preprocessing.py        # Imputation & Standard Scaling pipelines
│   ├── feature_engineering.py  # Outlier removal via IQR method
│   ├── train.py                # Fitting and exporting registry models
│   ├── evaluate.py             # Plotting utilities and validation metrics
│   ├── predict.py              # Wrapper for single & batch predictions
│   ├── generate_mock_data.py   # Utility to create test CSV files
│   └── utils.py                # PDF valuation certificate generator
│
├── templates/                  # Flask Jinja2 HTML templates
│   ├── base.html               # Grid structure, navigation & theme switcher
│   ├── index.html              # Core Landing & single prediction form
│   ├── model_info.html         # Live metrics, feature weights, OLS equation & plots
│   ├── batch.html              # CSV drag-and-drop batch valuation uploads
│   ├── history.html            # Chronological prediction history auditing
│   └── 404.html                # Premium minimalist page-not-found layout
│
└── static/                     # Web assets
    ├── css/
    │   └── style.css           # Premium minimalist stylesheet
    ├── js/
    │   └── main.js             # Async requests, animations, dropzone file handlers
    └── images/                 # Exported validation diagnostic plots
```

---

## Installation & Setup

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```

2. **Initialize a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Mock Dataset**:
   *The system is ready to self-initialize. Run this script to generate realistic training and test data immediately:*
   ```bash
   python3 src/generate_mock_data.py
   ```

---

## Running the Application Locally

1. **Train the ML model**:
   ```bash
   python3 src/train.py
   ```
   This fits the OLS Linear Regression, exports metrics, and generates diagnostic charts inside `static/images/`.

2. **Start the Flask Web App**:
   ```bash
   python3 app.py
   ```
   The application will start on `http://localhost:5000`. Open this URL in your web browser.

---




