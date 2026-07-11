import logging
import json
import joblib
from datetime import datetime
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import DataLoader
from src.feature_engineering import clean_data
from src.preprocessing import build_preprocessing_pipeline
from src.evaluate import run_evaluation_suite

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def train_model(train_csv_path: Path = config.TRAIN_PATH) -> dict:
    """
    Runs the full model training and versioning pipeline:
      1. Loads train dataset.
      2. Cleans data and filters outliers.
      3. Splits data into train and validation sets.
      4. Fits the scikit-learn preprocessing and linear regression pipeline.
      5. Runs evaluation and generates diagnostic plots.
      6. Exports the fitted pipeline and metadata (versioned and latest).
    """
    logger.info("Initializing model training process...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load Data
    loader = DataLoader()
    raw_df = loader.load_data(train_csv_path, is_training=True)
    
    # 2. Clean Data (Handle outliers/missing values)
    cleaned_df = clean_data(raw_df, is_training=True)
    if len(cleaned_df) < 10:
        error_msg = f"Cleaned dataset has only {len(cleaned_df)} samples. Too small to train."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    X, y = loader.split_features_target(cleaned_df)
    
    # 3. Train-Test Split (80% Train, 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Split data: Train={X_train.shape[0]} rows, Validation={X_val.shape[0]} rows")
    
    # 4. Build Pipeline
    preprocessor = build_preprocessing_pipeline()
    regressor = LinearRegression(**config.MODEL_PARAMS)
    
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    # 5. Fit model
    logger.info("Fitting model pipeline on training data...")
    full_pipeline.fit(X_train, y_train)
    logger.info("Model pipeline fitted successfully!")
    
    # 6. Evaluate
    y_val_pred = full_pipeline.predict(X_val)
    # Save evaluation plots directly inside static/images for Flask display
    metrics = run_evaluation_suite(y_val, y_val_pred, config.STATIC_DIR / "images")
    
    # 7. Extract coefficients and intercept for feature weights analysis
    fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
    fitted_regressor = full_pipeline.named_steps["regressor"]
    
    try:
        # Get feature names from transformer output
        feature_names = list(fitted_preprocessor.get_feature_names_out())
        coefficients = list(fitted_regressor.coef_)
        
        # Clean prefix names (e.g., 'num__sqft' -> 'sqft')
        clean_feature_names = [name.split("__")[-1] for name in feature_names]
        feature_weights = dict(zip(clean_feature_names, coefficients))
        intercept = float(fitted_regressor.intercept_)
    except Exception as e:
        logger.warning(f"Could not extract feature weights: {str(e)}")
        feature_weights = {}
        intercept = 0.0
        
    # 8. Create metadata dict
    metadata = {
        "timestamp": timestamp,
        "features": {
            "numerical": config.NUMERICAL_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES
        },
        "hyperparameters": config.MODEL_PARAMS,
        "metrics": metrics,
        "parameters": {
            "intercept": intercept,
            "coefficients": feature_weights
        },
        "train_samples": len(X_train),
        "validation_samples": len(X_val)
    }
    
    # 9. Save artifacts with versions and copy to latest
    # Prepare filenames
    model_filename = f"model_{timestamp}.joblib"
    metadata_filename = f"metadata_{timestamp}.json"
    
    model_path = config.MODEL_DIR / model_filename
    metadata_path = config.MODEL_DIR / metadata_filename
    
    latest_model_path = config.MODEL_DIR / config.LATEST_MODEL_NAME
    latest_metadata_path = config.MODEL_DIR / "metadata_latest.json"
    
    # Save files
    try:
        # Save versioned models
        joblib.dump(full_pipeline, model_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved versioned model to {model_path}")
        logger.info(f"Saved versioned metadata to {metadata_path}")
        
        # Save latest copies (for app consumption)
        joblib.dump(full_pipeline, latest_model_path)
        with open(latest_metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved latest model to {latest_model_path}")
        logger.info(f"Saved latest metadata to {latest_metadata_path}")
        
    except Exception as e:
        logger.error(f"Failed to export training artifacts: {str(e)}")
        raise e
        
    logger.info("Model training pipeline finished successfully.")
    return metadata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train house price prediction model.")
    parser.add_argument("--csv", type=str, default=str(config.TRAIN_PATH), help="Path to training CSV file")
    args = parser.parse_ok = parser.parse_args()
    
    try:
        train_model(Path(args.csv))
    except Exception as exc:
        logger.critical(f"Training pipeline terminated due to error: {str(exc)}")
        sys.exit(1)
