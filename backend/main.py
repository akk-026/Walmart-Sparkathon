from typing import List, Optional, Dict, Any
import pandas as pd
import joblib
import logging
from utils.geo_utils import compute_distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_model(purchase_data: Dict[str, Any], min_dist: float, rating: float) -> None:
    """Updates the ML model with new purchase data."""
    try:
        train_csv = 'data/training_data.csv'
        try:
            train_data = pd.read_csv(train_csv)
        except FileNotFoundError:
            train_data = pd.DataFrame(columns=['distance', 'rating', 'relevance_score'])
        
        new_data = pd.DataFrame({
            'distance': [min_dist],
            'rating': [rating],
            'relevance_score': [purchase_data.get('satisfaction_score', 0.0)]
        })
        
        train_data = pd.concat([train_data, new_data], ignore_index=True)
        train_data.to_csv(train_csv, index=False)
        
        X = train_data[['distance', 'rating']]
        y = train_data['relevance_score']
        model = joblib.load('model/xgb_model.pkl')
        model.fit(X, y)
        joblib.dump(model, 'model/xgb_model.pkl')
        logger.info("Model updated successfully")
    except Exception as e:
        logger.error(f"Failed to update model: {e}")

def rank_products(
    product_ids: List[str],  # Changed to List[str]
    customer_zip: str,
    new_purchase_record: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Ranks products based on distance and ratings using ML model.
    Args:
        product_ids: List of product ID strings to rank
        customer_zip: Customer's 6‑digit zipcode
        new_purchase_record: Optional dict with purchase data for online training
    Returns:
        List of product IDs sorted by predicted relevance (highest first)
    """
    try:
        products = pd.read_csv('data/Product.csv')
        warehouses = pd.read_csv('data/Warehouse.csv')
        model = joblib.load('model/xgb_model.pkl')
    except Exception as e:
        logger.error(f"Failed to load required data/model: {e}")
        raise RuntimeError(f"Data/model load error: {e}")

    scored_products = []
    
    for pid in product_ids:
        try:
            rating = float(products.loc[products['product_id'] == pid, 'product_rating'].iloc[0])
        except Exception:
            logger.warning(f"No rating found for product {pid}, defaulting to 3.0")
            rating = 3.0

        matching_wh = warehouses[warehouses['product_id'] == pid]
        if matching_wh.empty:
            logger.warning(f"No warehouse stocks product {pid}")
            continue

        min_dist = float('inf')
        for _, row in matching_wh.iterrows():
            loc_str = row['location_of_warehouse']
            wh_zip = loc_str.strip().split()[-2]
            dist = compute_distance(customer_zip, wh_zip)
            if dist < min_dist:
                min_dist = dist

        X = pd.DataFrame([[min_dist, rating]], columns=['distance', 'rating'])
        relevance = float(model.predict(X)[0])
        scored_products.append((pid, relevance))

        if new_purchase_record and new_purchase_record.get('product_id') == pid:
            update_model(new_purchase_record, min_dist, rating)

    scored_products.sort(key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in scored_products]
