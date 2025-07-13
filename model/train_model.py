# walmart_ranker_project/model/train_model.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_initial_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic training data:
      - distance: uniform [0, 500] km
      - rating: uniform [1, 5]
      - relevance_score: (1/(distance+1)) * (rating/5), normalized to [0,1]
    """
    np.random.seed(42)
    distances = np.random.uniform(0, 500, n_samples)
    ratings = np.random.uniform(1, 5, n_samples)
    raw = (1 / (distances + 1)) * (ratings / 5)
    relevance = raw / raw.max()
    df = pd.DataFrame({
        'distance': distances,
        'rating': ratings,
        'relevance_score': relevance
    })
    logger.info(f"Created synthetic training data with {n_samples} samples.")
    return df

def train_model():
    """
    1) Creates synthetic `training_data.csv` on first run.
    2) Otherwise loads existing training_data.csv.
    3) Trains (or re‑trains) XGBRegressor on ['distance','rating'] -> relevance_score.
    4) Saves model to model/xgb_model.pkl.
    """
    root = Path(__file__).parent.parent  # project root
    data_dir = root / 'data'
    model_dir = root / 'model'
    data_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    train_csv = data_dir / 'training_data.csv'
    model_path = model_dir / 'xgb_model.pkl'

    # 1) Create synthetic data if missing
    if not train_csv.exists():
        df = create_initial_training_data(n_samples=1000)
        df.to_csv(train_csv, index=False)
        logger.info(f"Wrote initial training data to {train_csv}")
    else:
        df = pd.read_csv(train_csv)
        logger.info(f"Loaded existing training data ({len(df)} rows) from {train_csv}")

    # 2) Train the model
    X = df[['distance', 'rating']]
    y = df['relevance_score']
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)
    logger.info("Model training complete.")

    # 3) Save the trained model
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
