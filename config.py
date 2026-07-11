import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

# Dataset Files
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Make sure standard directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "images").mkdir(parents=True, exist_ok=True)

# Dataset Column Mapping
# This mapping translates external dataset column names to internal standard features.
# - 'sqft': Above grade living area or square footage feature
# - 'bedrooms': Number of bedrooms
# - 'bathrooms': Number of bathrooms (FullBath + HalfBath logic is handled in the data loader)
# - 'target': The target price column
COLUMN_MAPPING = {
    "sqft": "GrLivArea",          # Kaggle House Prices: Above grade living area
    "bedrooms": "BedroomAbvGr",    # Kaggle House Prices: Bedrooms above grade
    "bathrooms": "FullBath",       # Kaggle House Prices: Full bathrooms
    "target": "SalePrice"          # Kaggle House Prices: Sale price in USD
}

# Configured columns to run in the Preprocessing Pipeline
# If the user replaces train.csv with another dataset, they can specify
# numerical and categorical features here.
NUMERICAL_FEATURES = ["sqft", "bedrooms", "bathrooms"]
CATEGORICAL_FEATURES = []  # Default is empty, can add string categorical columns here

# Fallback/Additional mappings for Bathrooms (e.g., if HalfBath is present, we can add 0.5 * HalfBath)
# Set to None if not applicable for the dataset.
BATHROOMS_HALF_BATH_COL = "HalfBath"

# Target variable formatting settings
CURRENCY_SYMBOL = "$"
CURRENCY_CODE = "USD"
PRICE_DECIMALS = 0

# Active Model Settings
# Path to the latest saved pipelines
LATEST_PREPROCESSOR_NAME = "preprocessor_latest.joblib"
LATEST_MODEL_NAME = "model_latest.joblib"

# Model hyperparameters (Linear Regression has few, but we can set fit_intercept here)
MODEL_PARAMS = {
    "fit_intercept": True
}

# Logging configuration
LOG_FILE = BASE_DIR / "app.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
