import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, List
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import DataLoader
from src.feature_engineering import add_features

# Set up logger
logger = logging.getLogger(__name__)

class Predictor:
    """
    Predictor wraps the loaded preprocessor/model pipeline to run inference
    on single record inputs (from the Flask web form) or batch datasets (from uploaded CSVs).
    """
    
    def __init__(self, model_path: Path = config.MODEL_DIR / config.LATEST_MODEL_NAME):
        self.model_path = model_path
        self.pipeline = None
        self._load_pipeline()
        
    def _load_pipeline(self):
        """
        Loads the saved pipeline from disk.
        """
        if not self.model_path.exists():
            error_msg = f"Trained model not found at: {self.model_path}. Please run train.py first to create a model."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"Successfully loaded trained model pipeline from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading trained model pipeline: {str(e)}")
            raise e
            
    def predict_single(self, sqft: float, bedrooms: int, bathrooms: float) -> Dict[str, Union[float, str]]:
        """
        Runs prediction for a single house.
        Returns a dictionary containing the estimated price and prediction details.
        """
        logger.info(f"Predicting single house price - Sqft: {sqft}, Bedrooms: {bedrooms}, Bathrooms: {bathrooms}")
        
        if self.pipeline is None:
            self._load_pipeline()
            
        # 1. Create input DataFrame with standardized column names
        input_df = pd.DataFrame([{
            "sqft": float(sqft),
            "bedrooms": int(bedrooms),
            "bathrooms": float(bathrooms)
        }])
        
        # 2. Predict using full pipeline (handles both preprocessing and regression)
        try:
            prediction = self.pipeline.predict(input_df)[0]
            # Clip predictions below $10,000 to keep them realistic
            prediction = max(10000.0, float(prediction))
            logger.info(f"Single prediction successful. Result: ${prediction:,.2f}")
            
            # Confidence/Accuracy indicators based on simple heuristics
            if sqft < 400 or sqft > 10000 or bedrooms < 1 or bedrooms > 10:
                confidence_msg = "Low confidence: Input values are far outside the normal distribution of our training data."
            else:
                confidence_msg = "High confidence: Inputs are well within the standard profile of houses in the dataset."
                
            return {
                "estimated_price": prediction,
                "confidence_message": confidence_msg,
                "input_summary": f"{sqft:,.0f} sqft, {bedrooms} bed, {bathrooms:g} bath",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"Single prediction failed: {str(e)}")
            raise e
            
    def predict_batch(self, input_csv_path: Path, output_csv_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Runs batch predictions on an input CSV file.
        Uses DataLoader to apply the configured column mapping so that arbitrary CSVs
        can be passed and processed dynamically.
        
        Returns the original DataFrame with a new 'predicted_price' column added.
        """
        input_csv_path = Path(input_csv_path)
        if output_csv_path:
            output_csv_path = Path(output_csv_path)
        logger.info(f"Running batch prediction on: {input_csv_path}")
        
        if self.pipeline is None:
            self._load_pipeline()
            
        # 1. Load data via loader (handles column mapping and standardized names)
        # We set is_training=False since the input might not contain a target column
        loader = DataLoader()
        try:
            standardized_df = loader.load_data(input_csv_path, is_training=False)
        except Exception as e:
            logger.error(f"Failed to standardize batch file during loader mapping: {str(e)}")
            raise e
            
        # 2. Extract features
        X, _ = loader.split_features_target(standardized_df)
        
        # 3. Predict
        try:
            predictions = self.pipeline.predict(X)
            # Clip to a sensible minimum house price
            predictions = np.clip(predictions, 10000, None)
            logger.info(f"Batch prediction completed. Mean predicted price: ${predictions.mean():,.2f}")
        except Exception as e:
            logger.error(f"Prediction step in batch failed: {str(e)}")
            raise e
            
        # 4. Read original file to append predictions without losing other original columns
        original_df = pd.read_csv(input_csv_path)
        original_df["predicted_price"] = predictions.round(2)
        
        # 5. Save if path provided
        if output_csv_path:
            original_df.to_csv(output_csv_path, index=False)
            logger.info(f"Batch predictions saved to {output_csv_path}")
            
        return original_df
