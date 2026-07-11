import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Set up logger
logger = logging.getLogger(__name__)

def build_preprocessing_pipeline(
    numerical_features: List[str] = config.NUMERICAL_FEATURES,
    categorical_features: List[str] = config.CATEGORICAL_FEATURES
) -> ColumnTransformer:
    """
    Creates and returns a scikit-learn ColumnTransformer preprocessing pipeline.
    
    - Numerical features: Imputed with the median, then standardized.
    - Categorical features: Imputed with a constant 'missing' tag, then One-Hot Encoded.
    
    This modular design handles custom datasets by adapting to the lists defined in config.py.
    """
    logger.info(f"Building preprocessing pipeline with: Numerical={numerical_features}, Categorical={categorical_features}")
    
    transformers = []
    
    # 1. Numerical Pipeline
    if numerical_features:
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numerical_features))
        logger.info("Added numerical transformer step (SimpleImputer + StandardScaler).")
        
    # 2. Categorical Pipeline
    if categorical_features:
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))
        logger.info("Added categorical transformer step (SimpleImputer + OneHotEncoder).")
        
    if not transformers:
        error_msg = "No features specified in Preprocessing Pipeline! Check config.py features."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop" # Drop any other columns not explicitly listed
    )
    
    logger.info("ColumnTransformer preprocessor successfully constructed.")
    return preprocessor
