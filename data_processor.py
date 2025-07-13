import pandas as pd
import numpy as np
from geopy.distance import geodesic
import re
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static mapping for city/state to coordinates (add more as needed)
CITY_STATE_COORDS = {
    ("GANDHI NAGAR", "Gujarat"): (23.2230, 72.6500),
    ("RAMANATHAPURAM", "Tamil Nadu"): (9.3716, 78.8308),
    ("KOZHIKODE", "Kerala"): (11.2588, 75.7804),
    ("Mandya", "Karnataka"): (12.5216, 76.8951),
    ("JAUNPUR", "Uttar Pradesh"): (25.7463, 82.6835),
    ("MALAPPURAM", "Kerala"): (11.0486, 76.0690),
    ("PUDUKKOTTAI", "Tamil Nadu"): (10.3813, 78.8214),
    ("NORTH GOA", "Goa"): (15.4887, 73.8278),
    ("BARMER", "Rajasthan"): (25.7531, 71.4031),
    ("AHMEDABAD", "Gujarat"): (23.0225, 72.5714),
    ("ALIGARH", "Uttar Pradesh"): (27.8974, 78.0880),
    ("ALIRAJPUR", "Madhya Pradesh"): (22.3131, 74.3640),
    ("ALLAHABAD", "Uttar Pradesh"): (25.4358, 81.8463),
    ("ASHOK NAGAR", "Madhya Pradesh"): (24.5804, 77.7310),
    ("AZAMGARH", "Uttar Pradesh"): (26.0671, 83.1836),
    ("BAGALKOT", "Karnataka"): (16.1860, 75.6961),
    ("BAGPAT", "Uttar Pradesh"): (28.9448, 77.2186),
    ("BAHRAICH", "Uttar Pradesh"): (27.5742, 81.5948),
    ("BALLIA", "Uttar Pradesh"): (25.7607, 84.1471),
    ("BANASKANTHA", "Gujarat"): (24.1124, 72.4388),
    ("BANKA", "Bihar"): (24.8853, 86.9226),
    ("BARABANKI", "Uttar Pradesh"): (26.6731, 81.1951),
    ("BARAN", "Rajasthan"): (25.1000, 76.5167),
    ("BARDHAMAN", "West Bengal"): (23.2324, 87.1905),
    ("BARWANI", "Madhya Pradesh"): (22.0363, 74.8981),
    ("BEED", "Maharashtra"): (18.9894, 75.7560),
    ("BEGUSARAI", "Bihar"): (25.4180, 86.1300),
    ("BELLARY", "Karnataka"): (15.1398, 76.9214),
    ("BENGALURU", "Karnataka"): (12.9716, 77.5946),
    ("BHAGALPUR", "Bihar"): (25.2445, 87.0088),
    ("BHANDARA", "Maharashtra"): (21.1702, 79.6499),
    ("BIJNOR", "Uttar Pradesh"): (29.3721, 78.1363),
    ("BULDHANA", "Maharashtra"): (20.5314, 76.1848),
    ("CHAMRAJNAGAR", "Karnataka"): (11.9218, 76.9395),
    ("CHANDRAPUR", "Maharashtra"): (19.9615, 79.2961),
    ("CHENNAI", "Tamil Nadu"): (13.0827, 80.2707),
    ("CHURACHANDPUR", "Manipur"): (24.3333, 93.6833),
    ("CHURU", "Rajasthan"): (28.3021, 74.9514),
    ("COOCH BEHAR", "West Bengal"): (26.3231, 89.4522),
    ("CUDDALORE", "Tamil Nadu"): (11.7463, 79.7644),
    ("DAHOD", "Gujarat"): (22.8312, 74.2634),
    ("DAKSHINA KANNADA", "Karnataka"): (12.8438, 75.2479),
    ("DANTEWADA", "Chattisgarh"): (18.9014, 81.3483),
    ("DAVANGERE", "Karnataka"): (14.4644, 75.9218),
    ("DEWAS", "Madhya Pradesh"): (22.9666, 76.0553),
    ("DHARWAD", "Karnataka"): (15.4589, 75.0078),
    ("DUNGARPUR", "Rajasthan"): (23.8431, 73.7147),
    ("EAST GODAVARI", "Andhra Pradesh"): (17.3213, 82.0409),
    ("ERNAKULAM", "Kerala"): (10.0168, 76.3418),
    ("ERODE", "Tamil Nadu"): (11.3410, 77.7172),
    ("GADCHIROLI", "Maharashtra"): (19.8000, 80.2000),
    ("GANJAM", "Odisha"): (19.3143, 84.7941),
    ("GHAZIPUR", "Uttar Pradesh"): (25.5836, 83.5603),
    ("GORAKHPUR", "Uttar Pradesh"): (26.7606, 83.3732),
    ("GUNA", "Madhya Pradesh"): (24.6465, 77.3113),
    ("GWALIOR", "Madhya Pradesh"): (26.2183, 78.1828),
    ("HANUMANGARH", "Rajasthan"): (29.5818, 74.3294),
    ("IMPHAL WEST", "Manipur"): (24.8170, 93.9368),
    ("JALOR", "Rajasthan"): (25.3451, 72.6155),
    ("JALPAIGURI", "West Bengal"): (26.5164, 88.7302),
    ("JAMNAGAR", "Gujarat"): (22.4707, 70.0577),
    ("JAMUI", "Bihar"): (24.9167, 86.2167),
    ("JHUJHUNU", "Rajasthan"): (28.1259, 75.3981),
    ("KALABURAGI", "Karnataka"): (17.3297, 76.8343),
    ("KANCHIPURAM", "Tamil Nadu"): (12.8397, 79.7006),
    ("KANDHAMAL", "Odisha"): (20.4674, 84.2301),
    ("KANNUR", "Kerala"): (11.8745, 75.3704),
    ("KAPURTHALA", "Punjab"): (31.3801, 75.3810),
    ("KARBI ANGLONG", "Assam"): (26.2006, 93.1381),
    ("KASARGOD", "Kerala"): (12.5021, 75.0159),
    ("KATIHAR", "Bihar"): (25.5335, 87.5834),
    ("KENDUJHAR", "Odisha"): (21.7751, 85.2312),
    ("KHEDA", "Gujarat"): (22.3039, 72.7688),
    ("KORAPUT", "Odisha"): (18.8097, 82.7108),
    ("KRISHNA", "Andhra Pradesh"): (16.1833, 80.6333),
    ("LALITPUR", "Uttar Pradesh"): (24.6887, 78.4108),
    ("LATUR", "Maharashtra"): (18.4088, 76.5604),
    ("MADURAI", "Tamil Nadu"): (9.9252, 78.1198),
    ("MAHABUB NAGAR", "Telangana"): (16.7432, 77.9856),
    ("MALEGAON", "Maharashtra"): (20.5576, 74.5089),
    ("MALKANGIRI", "Odisha"): (18.3643, 81.8990),
    ("MANDSAUR", "Madhya Pradesh"): (24.0719, 75.0699),
    ("MIRZAPUR", "Uttar Pradesh"): (25.1019, 82.5658),
    ("MOKOKCHUNG", "Nagaland"): (26.3274, 94.5152),
    ("MORENA", "Madhya Pradesh"): (26.4969, 78.0010),
    ("MUKTSAR", "Punjab"): (30.4743, 74.5170),
    ("MUNGER", "Bihar"): (25.3749, 86.4720),
    ("NADIA", "West Bengal"): (23.4723, 88.5563),
    ("NAGAON", "Assam"): (26.3509, 92.6923),
    ("NAGAUR", "Rajasthan"): (27.2020, 73.7339),
    ("NALANDA", "Bihar"): (25.1972, 85.5239),
    ("NANDED", "Maharashtra"): (19.1383, 77.3210),
    ("NAVSARI", "Gujarat"): (20.9517, 72.9324),
    ("NAWADA", "Bihar"): (24.8864, 85.5296),
    ("NIZAMABAD", "Telangana"): (18.6725, 78.0941),
    ("PALAKKAD", "Kerala"): (10.7867, 76.6548),
    ("PALGHAR", "Maharashtra"): (19.6915, 72.7645),
    ("PANCH MAHALS", "Gujarat"): (22.7477, 73.6069),
    ("PARBHANI", "Maharashtra"): (19.3833, 76.5667),
    ("PEDDAPALLI", "Telangana"): (18.6167, 79.2833),
    ("PORBANDAR", "Gujarat"): (21.6425, 69.6093),
    ("PUNE", "Maharashtra"): (18.5204, 73.8567),
    ("PURNIA", "Bihar"): (25.7770, 87.4750),
    ("RAIGARH", "Maharashtra"): (18.5783, 73.1193),
    ("RAIGARH(MH)", "Maharashtra"): (18.5783, 73.1193),
    ("RAMANAGAR", "Karnataka"): (12.7157, 77.2880),
    ("RATNAGIRI", "Maharashtra"): (16.9902, 73.3120),
    ("SABARKANTHA", "Gujarat"): (23.7333, 72.9833),
    ("SAMBALPUR", "Odisha"): (21.4704, 83.9701),
    ("SANT KABIR NAGAR", "Uttar Pradesh"): (26.7767, 83.0714),
    ("SANT RAVIDAS NAGAR", "Uttar Pradesh"): (25.3904, 82.5653),
    ("SARAN", "Bihar"): (25.9167, 84.7500),
    ("SATARA", "Maharashtra"): (17.6868, 74.0069),
    ("SHAHJAHANPUR", "Uttar Pradesh"): (27.8830, 79.9053),
    ("SHIVAMOGGA", "Karnataka"): (13.9299, 75.5681),
    ("SIDDIPET", "Telangana"): (18.1048, 78.8487),
    ("SIKAR", "Rajasthan"): (27.8919, 75.1384),
    ("SIRCILLA", "Telangana"): (18.3886, 78.8101),
    ("SITAMARHI", "Bihar"): (26.6000, 85.4833),
    ("SOLAPUR", "Maharashtra"): (17.6599, 75.9064),
    ("SOUTH ANDAMAN", "Andaman and Nico.In."): (11.6234, 92.7265),
    ("SOUTH TRIPURA", "Tripura"): (23.9408, 91.9882),
    ("SRIKAKULAM", "Andhra Pradesh"): (18.2989, 83.8975),
    ("SULTANPUR", "Uttar Pradesh"): (26.2649, 82.0724),
    ("SUNDERGARH", "Odisha"): (22.1166, 84.0333),
    ("SURAT", "Gujarat"): (21.1702, 72.8311),
    ("SURENDRA NAGAR", "Gujarat"): (22.7083, 71.6726),
    ("SURYAPET", "Telangana"): (17.1406, 79.6201),
    ("TAPI", "Gujarat"): (21.1800, 73.5625),
    ("TARN TARAN", "Punjab"): (31.4504, 74.9290),
    ("THANJAVUR", "Tamil Nadu"): (10.7905, 79.1398),
    ("TINSUKIA", "Assam"): (27.4863, 95.3533),
    ("TIRUNELVELI", "Tamil Nadu"): (8.7139, 77.7568),
    ("TIRUPPUR", "Tamil Nadu"): (11.1085, 77.3411),
    ("TIRUVARUR", "Tamil Nadu"): (10.7726, 79.6368),
    ("TONK", "Rajasthan"): (26.1654, 75.7902),
    ("TUMAKURU", "Karnataka"): (13.3409, 77.1020),
    ("UDAIPUR", "Rajasthan"): (24.5854, 73.7125),
    ("UDUPI", "Karnataka"): (13.3409, 74.7421),
    ("VADODARA", "Gujarat"): (22.3072, 73.1812),
    ("VARANASI", "Uttar Pradesh"): (25.3176, 82.9739),
    ("VELLORE", "Tamil Nadu"): (12.9716, 79.1590),
    ("VISAKHAPATNAM", "Andhra Pradesh"): (17.6868, 83.2185),
    ("WARANGAL", "Telangana"): (17.9689, 79.5941),
    ("WASHIM", "Maharashtra"): (20.1050, 77.1426),
    ("WAYANAD", "Kerala"): (11.6854, 76.1320),
    ("WEST GODAVARI", "Andhra Pradesh"): (16.5062, 81.0770),
    ("WEST MIDNAPORE", "West Bengal"): (22.4258, 87.3191),
    ("WEST TRIPURA", "Tripura"): (23.9408, 91.9882),
    ("YAVATMAL", "Maharashtra"): (20.3888, 78.1204),
    ("NOIDA", "Uttar Pradesh"): (28.5355, 77.3910),
    ("NOIDA", "uttar pradesh"): (28.5355, 77.3910),
    ("MANDYA", "Karnataka"): (12.5216, 76.8951),
}

DEFAULT_COORDS = (20.5937, 78.9629)  # Center of India

class DataProcessor:
    def __init__(self):
        self.products_df = None
        self.warehouses_df = None
        self.shipping_df = None
        self.warehouse_locations = {}
        self.zipcode_location_map = None

    def load_data(self):
        try:
            self.products_df = pd.read_csv('data/Product.csv')
            self.warehouses_df = pd.read_csv('data/Warehouse.csv')
            self.shipping_df = pd.read_csv('data/Shipping_matrix.csv')
            # Build zipcode-location map
            self.zipcode_location_map = self.shipping_df.drop_duplicates('customer_zipcode')[['customer_zipcode', 'customer_location']].set_index('customer_zipcode')['customer_location'].to_dict()
            logger.info("Data loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def extract_city_state(self, location_str: str) -> Tuple[str, str]:
        # Split by comma, strip whitespace
        parts = [p.strip() for p in location_str.split(',')]
        # Remove empty, numeric, and 'India' parts
        filtered = [p for p in parts if p and not p.isdigit() and p.upper() != 'INDIA']
        # City and state are usually the last two filtered parts
        if len(filtered) >= 2:
            city = filtered[-2].upper()
            state = filtered[-1].strip()
            return (city, state)
        return ("", "")

    def extract_location_coordinates(self, location_str: str) -> Tuple[float, float]:
        city, state = self.extract_city_state(location_str)
        coords = CITY_STATE_COORDS.get((city, state), DEFAULT_COORDS)
        if coords == DEFAULT_COORDS:
            logger.warning(f"No coordinates found for {city}, {state}. Using default.")
        return coords

    def get_location_coordinates(self, location_str: str) -> Tuple[float, float]:
        """Get coordinates for a location string"""
        try:
            # Extract city and state from location string
            city, state = self.extract_city_state(location_str)
            
            if not city or not state:
                logger.warning(f"Could not extract city/state from: {location_str}")
                return DEFAULT_COORDS
            
            # Look up coordinates
            coords = CITY_STATE_COORDS.get((city, state))
            if coords:
                return coords
            else:
                logger.warning(f"No coordinates found for {city}, {state}. Using default.")
                # Return default coordinates (center of India)
                return DEFAULT_COORDS
                
        except Exception as e:
            logger.error(f"Error getting coordinates for {location_str}: {e}")
            return DEFAULT_COORDS

    def process_warehouse_locations(self):
        if self.warehouses_df is None:
            logger.error("Warehouses data not loaded")
            return
        unique_warehouses = self.warehouses_df[['warehouse_id', 'location_of_warehouse']].drop_duplicates()
        for _, row in unique_warehouses.iterrows():
            warehouse_id = str(row['warehouse_id'])
            location = str(row['location_of_warehouse'])
            if warehouse_id not in self.warehouse_locations:
                coords = self.extract_location_coordinates(location)
                self.warehouse_locations[warehouse_id] = {
                    'location': location,
                    'coordinates': coords
                }
        logger.info(f"Processed {len(self.warehouse_locations)} warehouse locations")

    def calculate_distance_cost(self, user_location: str, warehouse_id: str) -> Dict:
        try:
            user_coords = self.extract_location_coordinates(user_location)
            warehouse_coords = self.warehouse_locations[warehouse_id]['coordinates']
            distance = geodesic(user_coords, warehouse_coords).kilometers
            base_cost = 50
            cost_per_km = 2
            estimated_cost = base_cost + (distance * cost_per_km)
            return {
                'distance_km': round(distance, 2),
                'estimated_cost': round(estimated_cost, 2),
                'user_coords': user_coords,
                'warehouse_coords': warehouse_coords
            }
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return {
                'distance_km': 1000,
                'estimated_cost': 500,
                'user_coords': DEFAULT_COORDS,
                'warehouse_coords': DEFAULT_COORDS
            }

    def get_available_zipcodes(self) -> List[str]:
        """Return a sorted list of unique zip codes as strings"""
        if self.zipcode_location_map is None:
            return []
        return sorted([str(z) for z in self.zipcode_location_map.keys()])

    def get_location_for_zipcode(self, zipcode: str) -> str:
        """Return the location string for a given zip code, or empty string if not found"""
        if self.zipcode_location_map is None:
            return ""
        loc = self.zipcode_location_map.get(int(zipcode)) if zipcode.isdigit() else self.zipcode_location_map.get(zipcode)
        if loc is None:
            loc = self.zipcode_location_map.get(str(zipcode))
        return loc if loc is not None else ""

    def get_available_products(self, user_location: str) -> pd.DataFrame:
        if self.warehouses_df is None or self.products_df is None:
            logger.error("Data not loaded")
            return pd.DataFrame()
        available_products = []
        for _, row in self.warehouses_df.iterrows():
            warehouse_id = str(row['warehouse_id'])
            product_id = str(row['product_id'])
            stock_count = int(row['stock_count'])
            if stock_count > 0:
                distance_info = self.calculate_distance_cost(user_location, warehouse_id)
                product_info = self.products_df[self.products_df['product_id'] == product_id]
                if not product_info.empty:
                    product_row = product_info.iloc[0]
                    available_products.append({
                        'product_id': product_id,
                        'product_name': str(product_row['product_name']),
                        'product_category': str(product_row['product_category']),
                        'product_price': float(product_row['product_price']),
                        'product_rating': float(product_row['product_rating']),
                        'warehouse_id': warehouse_id,
                        'stock_count': stock_count,
                        'distance_km': distance_info['distance_km'],
                        'shipping_cost': distance_info['estimated_cost'],
                        'total_cost': float(product_row['product_price']) + distance_info['estimated_cost']
                    })
        return pd.DataFrame(available_products)

    def get_product_embeddings_data(self) -> List[Dict]:
        if self.products_df is None:
            return []
        products_data = []
        for _, row in self.products_df.iterrows():
            text = f"{row['product_name']} {row['product_category']}"
            products_data.append({
                'id': row['product_id'],
                'text': text,
                'metadata': {
                    'product_name': row['product_name'],
                    'category': row['product_category'],
                    'price': row['product_price'],
                    'rating': row['product_rating']
                }
            })
        return products_data

    def get_user_recommendations(self, user_location: str, user_history: List[str] = None, top_k: int = 10) -> pd.DataFrame:
        if user_history is None:
            user_history = []
        available_products = self.get_available_products(user_location)
        if available_products is None or not isinstance(available_products, pd.DataFrame) or available_products.empty:
            return pd.DataFrame()
        available_products['distance_score'] = 1 / (1 + available_products['distance_km'])
        available_products['rating_score'] = available_products['product_rating'] / 5.0
        available_products['price_score'] = 1 / (1 + available_products['product_price'] / 10000)
        available_products['final_score'] = (
            0.4 * available_products['distance_score'] +
            0.4 * available_products['rating_score'] +
            0.2 * available_products['price_score']
        )
        recommendations = available_products.sort_values('final_score', ascending=False).head(top_k)
        return recommendations[['product_id', 'product_name', 'product_category', 'product_price', 'product_rating', 'warehouse_id', 'distance_km', 'shipping_cost', 'total_cost', 'final_score']].copy()

    def get_available_locations(self) -> List[str]:
        """Get list of available locations for dropdown"""
        if self.warehouses_df is None:
            return []
        
        try:
            # Get unique warehouse locations
            locations = self.warehouses_df['location_of_warehouse'].dropna().unique()
            
            # Format locations for display
            formatted_locations = []
            for location in locations:
                city, state = self.extract_city_state(location)
                if city and state:
                    formatted_locations.append(f"{city.title()}, {state.title()}")
                else:
                    formatted_locations.append(str(location))
            
            # Add some common Indian cities
            common_cities = [
                "Mumbai, Maharashtra",
                "Delhi, Delhi",
                "Bangalore, Karnataka", 
                "Hyderabad, Telangana",
                "Chennai, Tamil Nadu",
                "Kolkata, West Bengal",
                "Pune, Maharashtra",
                "Ahmedabad, Gujarat",
                "Jaipur, Rajasthan",
                "Lucknow, Uttar Pradesh"
            ]
            
            return sorted(list(set(formatted_locations + common_cities)))
        except Exception as e:
            logger.error(f"Error getting available locations: {e}")
            return [] 