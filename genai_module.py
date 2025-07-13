import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenAIProductSearch:
    def __init__(self):
        """Initialize the GenAI product search module"""
        try:
            # Load the sentence transformer model for semantic search
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("GenAI model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading GenAI model: {e}")
            self.model = None
        
        self.products_df = None
        self.product_embeddings = None
        self.product_tags = {}
        
    def load_products(self, products_df: pd.DataFrame):
        """Load products and generate embeddings"""
        self.products_df = products_df
        self._generate_product_tags()
        self._generate_embeddings()
        
    def _generate_product_tags(self):
        """Generate semantic tags for each product based on name and category"""
        if self.products_df is None:
            return
            
        for _, row in self.products_df.iterrows():
            product_id = row['product_id']
            name = str(row['product_name']).lower()
            category = str(row['product_category']).lower()
            
            # Extract key features from product name
            tags = []
            
            # Add category as primary tag
            tags.append(category)
            
            # Extract common product features
            if any(word in name for word in ['phone', 'mobile', 'smartphone']):
                tags.extend(['mobile', 'electronics', 'communication'])
            elif any(word in name for word in ['laptop', 'computer', 'pc']):
                tags.extend(['computer', 'electronics', 'work'])
            elif any(word in name for word in ['watch', 'smart watch']):
                tags.extend(['watch', 'accessories', 'fitness'])
            elif any(word in name for word in ['headphone', 'earphone', 'speaker']):
                tags.extend(['audio', 'electronics', 'music'])
            elif any(word in name for word in ['shirt', 't-shirt', 'dress', 'jeans']):
                tags.extend(['clothing', 'fashion', 'apparel'])
            elif any(word in name for word in ['shoes', 'footwear', 'sneaker']):
                tags.extend(['footwear', 'fashion', 'comfort'])
            elif any(word in name for word in ['book', 'novel', 'guide']):
                tags.extend(['books', 'education', 'reading'])
            elif any(word in name for word in ['food', 'snack', 'chips', 'chocolate']):
                tags.extend(['food', 'snacks', 'consumables'])
            elif any(word in name for word in ['toy', 'game', 'car', 'doll']):
                tags.extend(['toys', 'entertainment', 'children'])
            elif any(word in name for word in ['beauty', 'cosmetic', 'makeup']):
                tags.extend(['beauty', 'cosmetics', 'personal care'])
            elif any(word in name for word in ['health', 'medicine', 'vitamin']):
                tags.extend(['health', 'medicine', 'wellness'])
            elif any(word in name for word in ['sports', 'fitness', 'gym']):
                tags.extend(['sports', 'fitness', 'exercise'])
            elif any(word in name for word in ['home', 'kitchen', 'furniture']):
                tags.extend(['home', 'kitchen', 'furniture'])
            elif any(word in name for word in ['pet', 'dog', 'cat']):
                tags.extend(['pets', 'animals', 'care'])
            elif any(word in name for word in ['baby', 'infant', 'diaper']):
                tags.extend(['baby', 'infant', 'care'])
            elif any(word in name for word in ['automotive', 'car', 'bike']):
                tags.extend(['automotive', 'vehicle', 'transport'])
            elif any(word in name for word in ['jewelry', 'necklace', 'ring']):
                tags.extend(['jewelry', 'accessories', 'fashion'])
            elif any(word in name for word in ['bag', 'backpack', 'purse']):
                tags.extend(['bags', 'accessories', 'storage'])
            elif any(word in name for word in ['stationery', 'pen', 'paper']):
                tags.extend(['stationery', 'office', 'writing'])
            elif any(word in name for word in ['garden', 'plant', 'flower']):
                tags.extend(['gardening', 'plants', 'outdoor'])
            elif any(word in name for word in ['cleaning', 'detergent', 'soap']):
                tags.extend(['cleaning', 'hygiene', 'household'])
            elif any(word in name for word in ['beverage', 'drink', 'juice', 'tea']):
                tags.extend(['beverages', 'drinks', 'refreshment'])
            elif any(word in name for word in ['frozen', 'ice', 'cold']):
                tags.extend(['frozen', 'preserved', 'cold storage'])
            elif any(word in name for word in ['organic', 'natural', 'healthy']):
                tags.extend(['organic', 'natural', 'healthy'])
            elif any(word in name for word in ['pulse', 'dal', 'lentil']):
                tags.extend(['pulses', 'legumes', 'protein'])
            elif any(word in name for word in ['spice', 'masala', 'powder']):
                tags.extend(['spices', 'seasoning', 'cooking'])
            elif any(word in name for word in ['bakery', 'bread', 'cake']):
                tags.extend(['bakery', 'baked goods', 'dessert'])
            elif any(word in name for word in ['dairy', 'milk', 'curd', 'butter']):
                tags.extend(['dairy', 'milk products', 'fresh'])
            elif any(word in name for word in ['grocery', 'rice', 'wheat']):
                tags.extend(['groceries', 'staples', 'food'])
            
            # Add price-based tags
            price = float(row['product_price'])
            if price < 1000:
                tags.append('budget')
            elif price < 5000:
                tags.append('mid-range')
            else:
                tags.append('premium')
                
            # Add rating-based tags
            rating = float(row['product_rating'])
            if rating >= 4.5:
                tags.append('high-rated')
            elif rating >= 3.5:
                tags.append('well-rated')
            else:
                tags.append('average-rated')
            
            self.product_tags[product_id] = list(set(tags))  # Remove duplicates
            
        logger.info(f"Generated tags for {len(self.product_tags)} products")
    
    def _generate_embeddings(self):
        """Generate embeddings for all products"""
        if self.products_df is None or self.model is None:
            return
            
        # Create text representations for embeddings
        product_texts = []
        for _, row in self.products_df.iterrows():
            product_id = row['product_id']
            name = str(row['product_name'])
            category = str(row['product_category'])
            tags = ' '.join(self.product_tags.get(product_id, []))
            
            # Combine name, category, and tags for embedding
            text = f"{name} {category} {tags}"
            product_texts.append(text)
        
        # Generate embeddings
        try:
            self.product_embeddings = self.model.encode(product_texts)
            logger.info(f"Generated embeddings for {len(product_texts)} products")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self.product_embeddings = None
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform semantic search on products"""
        if self.model is None or self.product_embeddings is None:
            logger.error("Model or embeddings not available")
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                product_row = self.products_df.iloc[idx]
                results.append({
                    'product_id': product_row['product_id'],
                    'product_name': product_row['product_name'],
                    'product_category': product_row['product_category'],
                    'product_price': product_row['product_price'],
                    'product_rating': product_row['product_rating'],
                    'similarity_score': float(similarities[idx]),
                    'tags': self.product_tags.get(product_row['product_id'], [])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def search_by_tags(self, tags: List[str], top_k: int = 10) -> List[Dict]:
        """Search products by specific tags"""
        if not self.product_tags:
            return []
        
        # Convert tags to lowercase for matching
        search_tags = [tag.lower() for tag in tags]
        
        matching_products = []
        for product_id, product_tags in self.product_tags.items():
            product_tags_lower = [tag.lower() for tag in product_tags]
            
            # Calculate tag match score
            matches = sum(1 for tag in search_tags if tag in product_tags_lower)
            if matches > 0:
                product_row = self.products_df[self.products_df['product_id'] == product_id].iloc[0]
                matching_products.append({
                    'product_id': product_id,
                    'product_name': product_row['product_name'],
                    'product_category': product_row['product_category'],
                    'product_price': product_row['product_price'],
                    'product_rating': product_row['product_rating'],
                    'tag_match_score': matches,
                    'tags': product_tags
                })
        
        # Sort by tag match score and return top-k
        matching_products.sort(key=lambda x: x['tag_match_score'], reverse=True)
        return matching_products[:top_k]
    
    def get_product_recommendations(self, product_id: str, top_k: int = 5) -> List[Dict]:
        """Get similar product recommendations based on a product ID"""
        if product_id not in self.product_tags:
            return []
        
        # Get tags of the target product
        target_tags = self.product_tags[product_id]
        
        # Find products with similar tags
        similar_products = []
        for pid, tags in self.product_tags.items():
            if pid != product_id:
                # Calculate tag similarity
                common_tags = set(target_tags) & set(tags)
                similarity = len(common_tags) / max(len(target_tags), len(tags))
                
                if similarity > 0:
                    product_row = self.products_df[self.products_df['product_id'] == pid].iloc[0]
                    similar_products.append({
                        'product_id': pid,
                        'product_name': product_row['product_name'],
                        'product_category': product_row['product_category'],
                        'product_price': product_row['product_price'],
                        'product_rating': product_row['product_rating'],
                        'similarity_score': similarity,
                        'common_tags': list(common_tags)
                    })
        
        # Sort by similarity and return top-k
        similar_products.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_products[:top_k]
    
    def get_available_tags(self) -> List[str]:
        """Get all available tags for filtering"""
        all_tags = set()
        for tags in self.product_tags.values():
            all_tags.update(tags)
        return sorted(list(all_tags)) 