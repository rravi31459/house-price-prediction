import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def generate_house_prices_data(num_samples: int, seed: int = 42, is_test: bool = False) -> pd.DataFrame:
    """
    Generates mock housing dataset resembling the Kaggle House Prices dataset.
    Features:
      - Id: incremental identifier
      - GrLivArea: square footage (above grade living area)
      - BedroomAbvGr: number of bedrooms
      - FullBath: number of full bathrooms
      - HalfBath: number of half bathrooms
      - SalePrice (only for train): price of the house in USD
    """
    np.random.seed(seed)
    
    # 1. Generate core features
    # Square footage: log-normal or normal distribution between 800 and 4200
    gr_liv_area = np.random.normal(loc=1800, scale=600, size=num_samples).astype(int)
    gr_liv_area = np.clip(gr_liv_area, 600, 5000) # Clip extreme values
    
    # Bedrooms: correlated with area
    # Base bedrooms on area: 1 bedroom per ~500 sqft with some random variation
    bedroom_abv_gr = (gr_liv_area / 600).astype(int) + np.random.choice([-1, 0, 1], size=num_samples, p=[0.2, 0.6, 0.2])
    bedroom_abv_gr = np.clip(bedroom_abv_gr, 1, 6)
    
    # Bathrooms: correlated with bedrooms and area
    full_bath = (bedroom_abv_gr / 2).astype(int) + np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    full_bath = np.clip(full_bath, 1, 4)
    
    # HalfBath
    half_bath = np.random.choice([0, 1, 2], size=num_samples, p=[0.6, 0.3, 0.1])
    
    # Create DataFrame
    df = pd.DataFrame({
        "Id": np.arange(1, num_samples + 1) if not is_test else np.arange(10001, 10001 + num_samples),
        "GrLivArea": gr_liv_area,
        "BedroomAbvGr": bedroom_abv_gr,
        "FullBath": full_bath,
        "HalfBath": half_bath
    })
    
    # Add target column SalePrice if train set
    if not is_test:
        # Base price calculation
        # Price = 50000 + 130 * GrLivArea + 12000 * BedroomAbvGr + 28000 * (FullBath + 0.5 * HalfBath) + noise
        base_price = 50000 + (130 * gr_liv_area) + (12000 * bedroom_abv_gr) + (28000 * (full_bath + 0.5 * half_bath))
        # Add random normal noise (around 10% of base price)
        noise = np.random.normal(loc=0, scale=20000, size=num_samples)
        sale_price = base_price + noise
        
        # Ensure prices are positive and realistic (minimum $50k)
        sale_price = np.clip(sale_price, 50000, 950000).astype(int)
        df["SalePrice"] = sale_price
        
    return df

def main():
    print("Generating mock data files...")
    
    # Ensure data directory exists
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate 500 training samples
    train_df = generate_house_prices_data(num_samples=500, seed=42, is_test=False)
    train_df.to_csv(config.TRAIN_PATH, index=False)
    print(f"Created train.csv with {len(train_df)} rows at: {config.TRAIN_PATH}")
    
    # Generate 100 test samples (without SalePrice)
    test_df = generate_house_prices_data(num_samples=100, seed=101, is_test=True)
    test_df.to_csv(config.TEST_PATH, index=False)
    print(f"Created test.csv with {len(test_df)} rows at: {config.TEST_PATH}")
    
    print("Mock dataset generation completed successfully!")

if __name__ == "__main__":
    main()
