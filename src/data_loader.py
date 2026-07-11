import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Set up logger
logger = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader is responsible for reading the raw CSV datasets, checking for required columns,
    and translating/mapping external column names to standardized internal names.
    This architecture enables swapping datasets by modifying config.COLUMN_MAPPING.
    """
    
    def __init__(self, mapping: dict = config.COLUMN_MAPPING, half_bath_col: Optional[str] = config.BATHROOMS_HALF_BATH_COL):
        self.mapping = mapping
        self.half_bath_col = half_bath_col
        
    def load_data(self, file_path: Path, is_training: bool = True) -> pd.DataFrame:
        """
        Loads dataset from a CSV file path, applies mappings, and returns a DataFrame
        with standardized column names.
        
        Standardized Column names:
          - 'sqft': Numerical, representing square footage
          - 'bedrooms': Numerical, representing number of bedrooms
          - 'bathrooms': Numerical, representing number of bathrooms
          - 'price' (Optional, only for training/validation): Numerical target variable
        """
        file_path = Path(file_path)
        logger.info(f"Attempting to load data from {file_path}")
        
        if not file_path.exists():
            error_msg = f"Data file not found at path: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read CSV. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {str(e)}")
            raise e
            
        # Standardized DataFrame to return
        processed_df = pd.DataFrame()
        
        # 1. Map features
        features_to_map = ["sqft", "bedrooms", "bathrooms"]
        for feat in features_to_map:
            col_in_csv = self.mapping.get(feat)
            if not col_in_csv:
                error_msg = f"Feature '{feat}' is not defined in COLUMN_MAPPING."
                logger.error(error_msg)
                raise KeyError(error_msg)
                
            if col_in_csv not in df.columns:
                error_msg = f"Required column '{col_in_csv}' (mapped to '{feat}') not found in CSV columns: {list(df.columns)}"
                logger.error(error_msg)
                raise KeyError(error_msg)
                
            processed_df[feat] = df[col_in_csv].copy()
            
        # 2. Add Half Bathrooms if configured and present (specifically for Kaggle)
        if self.half_bath_col and self.half_bath_col in df.columns:
            logger.info(f"Found half bath column '{self.half_bath_col}'. Adjusting bathrooms count: bathrooms = bathrooms + 0.5 * half_bath")
            processed_df["bathrooms"] = processed_df["bathrooms"] + 0.5 * df[self.half_bath_col].fillna(0)
            
        # 3. Handle target column if training mode
        if is_training:
            target_col = self.mapping.get("target")
            if not target_col:
                error_msg = "Target column is not defined in COLUMN_MAPPING but training mode was requested."
                logger.error(error_msg)
                raise KeyError(error_msg)
                
            if target_col not in df.columns:
                error_msg = f"Target column '{target_col}' not found in CSV columns during training: {list(df.columns)}"
                logger.error(error_msg)
                raise KeyError(error_msg)
                
            processed_df["price"] = df[target_col].copy()
            
        logger.info(f"Data loading and mapping completed. Output columns: {list(processed_df.columns)}")
        return processed_df

    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Splits standard DataFrame into features (X) and target (y).
        """
        X = df[["sqft", "bedrooms", "bathrooms"]].copy()
        y = df["price"].copy() if "price" in df.columns else None
        return X, y
