import logging
import pandas as pd
from typing import Tuple

# Set up logger
logger = logging.getLogger(__name__)

def remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame based on the Interquartile Range (IQR) method.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for outlier removal. Skipping.")
        return df
        
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    
    before_count = len(df)
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    after_count = len(filtered_df)
    
    removed = before_count - after_count
    logger.info(f"IQR outlier removal on '{column}': bounds [{lower_bound:.2f}, {upper_bound:.2f}]. Removed {removed} rows ({removed/before_count*100:.2f}%).")
    
    return filtered_df

def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Cleans raw standardized data.
    - Removes rows with missing values in key features.
    - If training, removes extreme outliers to stabilize the Linear Regression model.
    """
    logger.info("Starting data cleaning pipeline...")
    
    # 1. Handle missing values: drop records with missing features
    essential_cols = ["sqft", "bedrooms", "bathrooms"]
    if is_training:
        essential_cols.append("price")
        
    initial_len = len(df)
    df_cleaned = df.dropna(subset=essential_cols).copy()
    if len(df_cleaned) < initial_len:
        logger.info(f"Dropped {initial_len - len(df_cleaned)} rows due to missing essential columns.")
        
    # 2. Basic physical logic checks (e.g., area > 0, bedrooms >= 0)
    df_cleaned = df_cleaned[df_cleaned["sqft"] > 0]
    df_cleaned = df_cleaned[df_cleaned["bedrooms"] >= 0]
    df_cleaned = df_cleaned[df_cleaned["bathrooms"] >= 0]
    
    # 3. Outlier removal (only apply to training data to avoid leaking info or dropping test cases)
    if is_training and len(df_cleaned) > 0:
        # Remove outliers in sqft and price to avoid fitting high leverage points
        df_cleaned = remove_outliers_iqr(df_cleaned, "sqft", factor=2.0)
        df_cleaned = remove_outliers_iqr(df_cleaned, "price", factor=2.0)
        
    logger.info(f"Data cleaning finished. Input size: {initial_len}, output size: {len(df_cleaned)}")
    return df_cleaned

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional feature engineering step. Adds interaction and ratio features:
    - sqft_per_bedroom: average square footage per bedroom
    - bathrooms_per_bedroom: ratio of bathrooms to bedrooms
    """
    logger.info("Adding engineered features...")
    df_feat = df.copy()
    
    # Avoid division by zero
    df_feat["sqft_per_bedroom"] = df_feat["sqft"] / df_feat["bedrooms"].replace(0, 1)
    df_feat["bathrooms_per_bedroom"] = df_feat["bathrooms"] / df_feat["bedrooms"].replace(0, 1)
    
    # Fill any NaNs created by invalid inputs
    df_feat["sqft_per_bedroom"] = df_feat["sqft_per_bedroom"].fillna(0)
    df_feat["bathrooms_per_bedroom"] = df_feat["bathrooms_per_bedroom"].fillna(0)
    
    return df_feat
