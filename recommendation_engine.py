import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Tuple
from data_processor import DataProcessor
from genai_module import GenAIProductSearch
from geopy.distance import geodesic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine with transformer model and ChromaDB"""
        try:
            # Initialize transformer model for semantic search
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            self.model = None
        
        # Initialize ChromaDB for storing embeddings and user history
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection("product_embeddings")
            self.user_history_collection = self.chroma_client.get_or_create_collection("user_history")
            logger.info("ChromaDB initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
            self.user_history_collection = None
        
        # Initialize data processor and GenAI module
        self.data_processor = DataProcessor()
        self.genai_search = GenAIProductSearch()
        
        self.products_df = None
        self.warehouses_df = None
        self.product_embeddings = None
        
    def load_data(self):
        """Load all data and initialize embeddings"""
        if not self.data_processor.load_data():
            logger.error("Failed to load data")
            return False
            
        self.products_df = self.data_processor.products_df
        self.warehouses_df = self.data_processor.warehouses_df
        
        # Process warehouse locations
        self.data_processor.process_warehouse_locations()
        
        # Load products into GenAI module
        self.genai_search.load_products(self.products_df)
        
        # Generate embeddings for products
        self._generate_product_embeddings()
        
        logger.info("Data loaded and embeddings generated successfully")
        return True
    
    def _generate_product_embeddings(self):
        """Generate embeddings for all products"""
        if self.products_df is None or self.model is None:
            return
            
        # Create text representations for embeddings
        product_texts = []
        product_ids = []
        product_metadata = []
        
        for _, row in self.products_df.iterrows():
            product_id = str(row['product_id'])
            name = str(row['product_name'])
            category = str(row['product_category'])
            
            # Create text representation
            text = f"{name} {category}"
            
            product_texts.append(text)
            product_ids.append(product_id)
            product_metadata.append({
                'product_name': name,
                'category': category,
                'price': float(row['product_price']),
                'rating': float(row['product_rating'])
            })
        
        # Generate embeddings
        try:
            embeddings = self.model.encode(product_texts)
            
            # Store in ChromaDB
            if self.collection:
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    ids=product_ids,
                    metadatas=product_metadata
                )
            
            self.product_embeddings = embeddings
            logger.info(f"Generated embeddings for {len(product_texts)} products")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self.product_embeddings = None
    
    def get_distance_optimized_recommendations(self, user_location: str, user_id: str = None, top_k: int = 10) -> pd.DataFrame:
        """Get distance-optimized recommendations with weightage for distance, rating, and history"""
        if self.products_df is None or self.warehouses_df is None:
            return pd.DataFrame()
        
        try:
            # Get user's location coordinates
            user_coords = self.data_processor.get_location_coordinates(user_location)
            
            # Get user purchase history for weightage
            user_history = []
            if user_id and self.user_history_collection:
                try:
                    results = self.user_history_collection.get(
                        where={"user_id": {"$eq": user_id}},
                        include=['metadatas']
                    )
                    if results['ids']:
                        user_history = [metadata['product_id'] for metadata in results['metadatas']]
                except Exception as e:
                    logger.warning(f"Could not retrieve user history: {e}")
            
            # Calculate recommendations with weightage
            recommendations = []
            
            for _, warehouse in self.warehouses_df.iterrows():
                # Get warehouse location coordinates
                warehouse_location = str(warehouse['location_of_warehouse'])
                warehouse_coords = self.data_processor.get_location_coordinates(warehouse_location)
                
                # Calculate distance
                distance = geodesic(user_coords, warehouse_coords).kilometers
                
                # Get product details
                product_id = str(warehouse['product_id'])
                product_info = self.products_df[self.products_df['product_id'] == product_id]
                
                if product_info.empty:
                    continue
                
                product = product_info.iloc[0]
                
                # Calculate shipping cost based on distance
                shipping_cost = max(50, distance * 2)  # Base cost + distance factor
                
                # Calculate weightage scores
                distance_score = max(0, 1 - (distance / 1000))  # Higher score for closer warehouses
                rating_score = product['product_rating'] / 5.0  # Normalize rating
                history_score = 1.0 if product_id in user_history else 0.5  # Boost if user bought similar
                
                # Combined score with weightage
                final_score = (0.4 * distance_score + 0.4 * rating_score + 0.2 * history_score)
                
                recommendations.append({
                    'product_id': product_id,
                    'product_name': product['product_name'],
                    'product_category': product['product_category'],
                    'product_price': product['product_price'],
                    'product_rating': product['product_rating'],
                    'warehouse_id': warehouse['warehouse_id'],
                    'distance_km': distance,
                    'shipping_cost': shipping_cost,
                    'total_cost': product['product_price'] + shipping_cost,
                    'final_score': final_score,
                    'distance_score': distance_score,
                    'rating_score': rating_score,
                    'history_score': history_score
                })
            
            # Sort by final score and return top_k
            if recommendations:
                df = pd.DataFrame(recommendations)
                df = df.sort_values('final_score', ascending=False).head(top_k)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return pd.DataFrame()
    
    def semantic_search(self, query: str, top_k: int = 20, user_location: str = None, user_id: str = None) -> List[Dict]:
        """Perform semantic search using GenAI module, then re-rank by distance/rating/history/category"""
        # Get initial semantic results
        results = self.genai_search.semantic_search(query, top_k=top_k*2)  # get more to re-rank
        if not results:
            return []
        # Get user info
        user_coords = self.data_processor.get_location_coordinates(user_location) if user_location else None
        user_history = []
        if user_id and self.user_history_collection:
            try:
                db_results = self.user_history_collection.get(
                    where={"user_id": {"$eq": user_id}},
                    include=['metadatas']
                )
                if db_results['ids']:
                    user_history = [metadata['product_id'] for metadata in db_results['metadatas']]
            except Exception as e:
                logger.warning(f"Could not retrieve user history: {e}")
        # Re-rank with priority: Search Query (50%) > Distance (20%) > Rating (20%) > History (10%)
        reranked = []
        for r in results:
            product_id = r['product_id']
            # Search query similarity (highest priority)
            query_score = r['similarity_score']
            # Distance
            distance_score = 0.5
            if user_coords:
                warehouse_info = self.warehouses_df[self.warehouses_df['product_id'] == product_id]
                if not warehouse_info.empty:
                    warehouse = warehouse_info.iloc[0]
                    warehouse_coords = self.data_processor.get_location_coordinates(str(warehouse['location_of_warehouse']))
                    distance = geodesic(user_coords, warehouse_coords).kilometers
                    distance_score = max(0, 1 - (distance / 1000))
            # Rating
            rating_score = r['product_rating'] / 5.0
            # History
            history_score = 1.0 if product_id in user_history else 0.5
            # Category boost
            query_lower = query.lower()
            category_boost = 1.0
            if r['product_category'].lower() in query_lower:
                category_boost = 1.2  # boost if query matches category
            # Final score with new weights: Query(50%) > Distance(20%) > Rating(20%) > History(10%)
            final_score = (0.5 * query_score + 0.2 * distance_score + 0.2 * rating_score + 0.1 * history_score) * category_boost
            reranked.append({**r, 'distance_score': distance_score, 'rating_score': rating_score, 'history_score': history_score, 'final_score': final_score})
        reranked = sorted(reranked, key=lambda x: x['final_score'], reverse=True)[:top_k]
        return reranked
    
    def search_by_tags(self, tags: List[str], top_k: int = 20, user_location: str = None, user_id: str = None) -> List[Dict]:
        """Search products by tags using GenAI module, then re-rank by category/distance/rating/history"""
        # Get initial tag-based results
        results = self.genai_search.search_by_tags(tags, top_k=top_k*2)  # get more to re-rank
        if not results:
            return []
        # Get user info
        user_coords = self.data_processor.get_location_coordinates(user_location) if user_location else None
        user_history = []
        if user_id and self.user_history_collection:
            try:
                db_results = self.user_history_collection.get(
                    where={"user_id": {"$eq": user_id}},
                    include=['metadatas']
                )
                if db_results['ids']:
                    user_history = [metadata['product_id'] for metadata in db_results['metadatas']]
            except Exception as e:
                logger.warning(f"Could not retrieve user history: {e}")
        # Re-rank with priority: Category Match (50%) > Distance (20%) > Rating (20%) > History (10%)
        reranked = []
        for r in results:
            product_id = r['product_id']
            # Category match score (highest priority)
            category_score = r['tag_match_score']
            # Distance
            distance_score = 0.5
            if user_coords:
                warehouse_info = self.warehouses_df[self.warehouses_df['product_id'] == product_id]
                if not warehouse_info.empty:
                    warehouse = warehouse_info.iloc[0]
                    warehouse_coords = self.data_processor.get_location_coordinates(str(warehouse['location_of_warehouse']))
                    distance = geodesic(user_coords, warehouse_coords).kilometers
                    distance_score = max(0, 1 - (distance / 1000))
            # Rating
            rating_score = r['product_rating'] / 5.0
            # History
            history_score = 1.0 if product_id in user_history else 0.5
            # Final score with new weights: Category(50%) > Distance(20%) > Rating(20%) > History(10%)
            final_score = 0.5 * category_score + 0.2 * distance_score + 0.2 * rating_score + 0.1 * history_score
            reranked.append({**r, 'distance_score': distance_score, 'rating_score': rating_score, 'history_score': history_score, 'final_score': final_score})
        reranked = sorted(reranked, key=lambda x: x['final_score'], reverse=True)[:top_k]
        return reranked
    
    def get_user_history_recommendations(self, user_id: str, user_location: str = None, top_k: int = 10) -> List[Dict]:
        """Get recommendations based on user purchase history"""
        if not self.user_history_collection:
            logger.warning("ChromaDB not available for history recommendations")
            return []
        
        try:
            # Get user history from ChromaDB
            results = self.user_history_collection.get(
                where={"user_id": {"$eq": user_id}},
                include=['metadatas']
            )
            
            if not results['ids']:
                logger.info(f"No purchase history found for user {user_id}")
                # Return popular products as fallback
                return self._get_popular_products(top_k)
            
            # Get product IDs from user history
            purchased_products = [metadata['product_id'] for metadata in results['metadatas']]
            logger.info(f"Found {len(purchased_products)} purchases for user {user_id}")
            
            # Find similar products to purchased ones
            similar_products = []
            for product_id in purchased_products:
                try:
                    similar = self.genai_search.get_product_recommendations(product_id, top_k=3)
                    similar_products.extend(similar)
                    logger.info(f"Found {len(similar)} similar products for {product_id}")
                except Exception as e:
                    logger.warning(f"Error getting recommendations for product {product_id}: {e}")
            
            # Remove duplicates and sort by similarity
            unique_products = {}
            for product in similar_products:
                pid = product['product_id']
                if pid not in unique_products or product['similarity_score'] > unique_products[pid]['similarity_score']:
                    unique_products[pid] = product
            
            # Sort by similarity score and return top-k
            sorted_products = sorted(unique_products.values(), key=lambda x: x['similarity_score'], reverse=True)
            
            if sorted_products:
                logger.info(f"Returning {len(sorted_products[:top_k])} history-based recommendations")
                return sorted_products[:top_k]
            else:
                logger.info("No similar products found, returning popular products")
                return self._get_popular_products(top_k)
            
        except Exception as e:
            logger.error(f"Error getting user history recommendations: {e}")
            return self._get_popular_products(top_k)
    
    def _get_popular_products(self, top_k: int = 10) -> List[Dict]:
        """Get popular products as fallback recommendations"""
        try:
            if self.products_df is None:
                return []
            
            # Get top-rated products
            popular_products = self.products_df.nlargest(top_k, 'product_rating')
            
            results = []
            for _, row in popular_products.iterrows():
                results.append({
                    'product_id': row['product_id'],
                    'product_name': row['product_name'],
                    'product_category': row['product_category'],
                    'product_price': row['product_price'],
                    'product_rating': row['product_rating'],
                    'similarity_score': 0.5,  # Default score for popular products
                    'common_tags': self.genai_search.product_tags.get(row['product_id'], [])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting popular products: {e}")
            return []
    
    def add_user_purchase(self, user_id: str, product_id: str, location: str):
        """Add a purchase to user history"""
        if not self.user_history_collection or self.products_df is None:
            logger.warning("ChromaDB or products not available for purchase history")
            return
        
        try:
            # Get product details
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if product_info.empty:
                logger.warning(f"Product {product_id} not found")
                return
            
            product_row = product_info.iloc[0]
            
            # Create unique ID for the purchase with timestamp
            timestamp = int(pd.Timestamp.now().timestamp())
            purchase_id = f"{user_id}_{product_id}_{timestamp}"
            
            # Create a proper embedding (384 dimensions to match the collection)
            dummy_embedding = [0.0] * 384
            
            # Add to ChromaDB with correct embedding dimension
            self.user_history_collection.add(
                embeddings=[dummy_embedding],
                ids=[purchase_id],
                metadatas=[{
                    'user_id': user_id,
                    'product_id': product_id,
                    'product_name': product_row['product_name'],
                    'category': product_row['product_category'],
                    'price': float(product_row['product_price']),
                    'location': location,
                    'timestamp': pd.Timestamp.now().isoformat()
                }]
            )
            
            logger.info(f"✅ Added purchase for user {user_id}: {product_row['product_name']}")
            
            # Verify the purchase was stored
            try:
                results = self.user_history_collection.get(
                    where={"user_id": {"$eq": user_id}},
                    include=['metadatas']
                )
                logger.info(f"Verified: User {user_id} now has {len(results['ids'])} purchases")
            except Exception as e:
                logger.warning(f"Could not verify purchase storage: {e}")
            
        except Exception as e:
            logger.error(f"Error adding user purchase: {e}")
    
    def get_product_distance_info(self, product_id: str, user_location: str) -> Dict:
        """Get distance information for a specific product"""
        try:
            # Find the warehouse that has this product
            warehouse_info = self.warehouses_df[self.warehouses_df['product_id'] == product_id]
            if warehouse_info.empty:
                return None
            
            # Get the first warehouse (closest one)
            warehouse = warehouse_info.iloc[0]
            warehouse_location = str(warehouse['location_of_warehouse'])
            warehouse_coords = self.data_processor.get_location_coordinates(warehouse_location)
            user_coords = self.data_processor.get_location_coordinates(user_location)
            
            # Calculate distance
            distance = geodesic(user_coords, warehouse_coords).kilometers
            
            return {
                'distance_km': distance,
                'warehouse_id': warehouse['warehouse_id'],
                'shipping_cost': max(50, distance * 2)
            }
        except Exception as e:
            logger.error(f"Error getting distance info for product {product_id}: {e}")
            return None
    
    def get_available_tags(self) -> List[str]:
        """Get all available tags for filtering"""
        return self.genai_search.get_available_tags()
    
    def get_product_details(self, product_id: str) -> Dict:
        """Get detailed information about a product"""
        product_info = self.products_df[self.products_df['product_id'] == product_id]
        if product_info.empty:
            return {}
        
        product_row = product_info.iloc[0]
        return {
            'product_id': product_id,
            'product_name': product_row['product_name'],
            'product_category': product_row['product_category'],
            'product_price': float(product_row['product_price']),
            'product_rating': float(product_row['product_rating']),
            'tags': self.genai_search.product_tags.get(product_id, [])
        } 