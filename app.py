import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from recommendation_engine import RecommendationEngine
import logging

# NEW: import our tiny HTTP client for ranking
from utils.api_client import rank_products_by_relevance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

def initialize_engine():
    """Initialize the recommendation engine"""
    if st.session_state.recommendation_engine is None:
        with st.spinner("Loading recommendation engine..."):
            engine = RecommendationEngine()
            if engine.load_data():
                st.session_state.recommendation_engine = engine
                st.success("✅ Recommendation engine loaded successfully!")
            else:
                st.error("❌ Failed to load recommendation engine")
                return None
    return st.session_state.recommendation_engine

def show_general_recommendations(engine, user_location, user_id):
    """(unchanged)"""
    # … existing code …
    """Show general distance-optimized recommendations"""
    st.header("📍 Distance-Optimized Recommendations")
    
    # Get recommendations with user history weightage
    recommendations = engine.get_distance_optimized_recommendations(user_location, user_id, 20)
    
    if recommendations.empty:
        st.warning("No recommendations available for your location.")
        st.info("💡 Try selecting a different location or check if the location is supported.")
        return
    
    # Display recommendations
    st.subheader(f"Top 10 Recommendations for {user_location}")
    
    # Create a DataFrame for display
    display_df = recommendations[['product_name', 'product_category', 'product_price', 
                                'product_rating', 'distance_km', 'shipping_cost', 'total_cost', 'final_score']].copy()
    display_df.columns = ['Product', 'Category', 'Price (₹)', 'Rating', 'Distance (km)', 'Shipping (₹)', 'Total Cost (₹)', 'Score']
    
    # Format the display
    for col in ['Price (₹)', 'Shipping (₹)', 'Total Cost (₹)']:
        display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.2f}")
    display_df['Rating'] = display_df['Rating'].apply(lambda x: f"{x:.1f} ⭐")
    display_df['Distance (km)'] = display_df['Distance (km)'].apply(lambda x: f"{x:.1f} km")
    display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Show weightage breakdown
    if not recommendations.empty:
        st.subheader("🎯 Recommendation Weightage")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_distance = recommendations['distance_score'].mean()
            st.metric("Distance Score", f"{avg_distance:.3f}")
        
        with col2:
            avg_rating = recommendations['rating_score'].mean()
            st.metric("Rating Score", f"{avg_rating:.3f}")
        
        with col3:
            avg_history = recommendations['history_score'].mean()
            st.metric("History Score", f"{avg_history:.3f}")
    
    # Purchase simulation
    st.subheader("🛒 Simulate Purchase")
    selected_product = st.selectbox(
        "Select a product to purchase from General Recommendations:",
        options=recommendations['product_name'].tolist()
    )
    
    if st.button("Purchase from General Recommendations"):
        try:
            # Get product details including distance
            product_row = recommendations[recommendations['product_name'] == selected_product].iloc[0]
            product_id = product_row['product_id']
            distance = product_row['distance_km']
            warehouse_id = product_row['warehouse_id']
            
            # Add purchase to history
            engine.add_user_purchase(user_id, product_id, user_location)
            
            # Show detailed purchase confirmation
            st.success(f"✅ Purchased: {selected_product}")
            st.info(f"📦 Warehouse: {warehouse_id}")
            st.info(f"📍 Distance: {distance:.1f} km from your location")
            st.info(f"💰 Shipping Cost: ₹{product_row['shipping_cost']:.2f}")
            st.info(f"💳 Total Cost: ₹{product_row['total_cost']:.2f}")
            st.info("💡 Your purchase history will now influence future recommendations!")
            st.balloons()
            
            # Force refresh of recommendations
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing purchase: {e}")


def show_semantic_search(engine, user_location):
    """Show semantic search functionality with per-category relevance sorting"""
    st.header("🔍 Semantic Product Search")
    search_query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'wireless headphones', 'organic food', 'gaming laptop'"
    )
    user_id = st.session_state.user_id

    if not search_query:
        return

    with st.spinner("Searching for products..."):
        results = engine.semantic_search(
            search_query, top_k=50,
            user_location=user_location,
            user_id=user_id
        )

    if not results:
        st.warning("No products found matching your query.")
        return

    st.subheader(f"Search Results for: '{search_query}'")

    # Build a DataFrame from results
    df = pd.DataFrame(results)

    # Prepare for final ordered list
    ordered_frames = []
    # Group by category in original order
    for cat, grp in df.groupby("product_category", sort=False):
        # Extract product IDs for this category
        ids = grp["product_id"].tolist()
        # Call XGB service to sort by [distance + rating]
        sorted_ids = rank_products_by_relevance(ids, st.session_state.user_id)
        # Reorder this subgroup
        sorted_grp = grp.set_index("product_id").loc[sorted_ids].reset_index()
        ordered_frames.append(sorted_grp)

    # Combine back
    final_df = pd.concat(ordered_frames, ignore_index=True)

    # Build display DataFrame
    display_data = []
    for _, row in final_df.iterrows():
        distance_info = engine.get_product_distance_info(
            row['product_id'], user_location
        )
        display_data.append({
            'Product': row['product_name'],
            'Category': row['product_category'],
            'Price (₹)': f"₹{row['product_price']:,.2f}",
            'Rating': f"{row['product_rating']:.1f} ⭐",
            'Similarity': f"{row['similarity_score']:.3f}",
            'Distance': f"{distance_info['distance_km']:.1f} km" if distance_info else "N/A",
            'Warehouse': distance_info['warehouse_id'] if distance_info else "N/A",
            'Tags': ', '.join(row['tags'][:5])
        })

    st.dataframe(pd.DataFrame(display_data), use_container_width=True)

    # Purchase option (unchanged)
    if st.button("Purchase Selected Product"):
        selected_product = st.selectbox(
            "Select a product to purchase:",
            options=final_df['product_name'].tolist()
        )
        try:
            product_id = final_df.loc[
                final_df['product_name'] == selected_product, 'product_id'
            ].iloc[0]
            engine.add_user_purchase(
                st.session_state.user_id,
                product_id,
                user_location
            )
            st.success(f"✅ Purchased: {selected_product}")
            st.info("💡 Your purchase history will now influence future recommendations!")
            st.balloons()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error processing purchase: {e}")

def show_tag_based_search(engine, user_location):
    """Show tag-based search with per-category relevance sorting"""
    st.header("🏷️ Tag-based Product Search")
    available_tags = engine.get_available_tags()
    selected_tags = st.multiselect(
        "Select tags to filter products:",
        options=available_tags
    )
    if not selected_tags:
        return

    with st.spinner("Searching for products with selected tags..."):
        results = engine.search_by_tags(
            selected_tags, top_k=50,
            user_location=user_location,
            user_id=st.session_state.user_id
        )

    if not results:
        st.warning("No products found matching the selected tags.")
        return

    st.subheader(f"Products matching tags: {', '.join(selected_tags)}")

    # Build DataFrame
    df = pd.DataFrame(results)
    ordered = []
    for cat, grp in df.groupby("product_category", sort=False):
        ids = grp["product_id"].tolist()
        sorted_ids = rank_products_by_relevance(ids, st.session_state.user_id)
        sorted_grp = grp.set_index("product_id").loc[sorted_ids].reset_index()
        ordered.append(sorted_grp)
    final_df = pd.concat(ordered, ignore_index=True)

    # Display
    display_data = []
    for _, row in final_df.iterrows():
        distance_info = engine.get_product_distance_info(
            row['product_id'], user_location
        )
        display_data.append({
            'Product': row['product_name'],
            'Category': row['product_category'],
            'Price (₹)': f"₹{row['product_price']:,.2f}",
            'Rating': f"{row['product_rating']:.1f} ⭐",
            'Tag Matches': row['tag_match_score'],
            'Distance': f"{distance_info['distance_km']:.1f} km" if distance_info else "N/A",
            'Warehouse': distance_info['warehouse_id'] if distance_info else "N/A",
            'Tags': ', '.join(row['tags'][:5])
        })
    st.dataframe(pd.DataFrame(display_data), use_container_width=True)

    # Purchase button (unchanged)…
    if st.button("Purchase Selected Product"):
        selected_product = st.selectbox(
            "Select a product to purchase:",
            options=final_df['product_name'].tolist()
        )
        try:
            pid = final_df.loc[
                final_df['product_name'] == selected_product, 'product_id'
            ].iloc[0]
            engine.add_user_purchase(
                st.session_state.user_id, pid, user_location
            )
            st.success(f"✅ Purchased: {selected_product}")
            st.balloons()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {e}")

def show_history_recommendations(engine, user_id, user_location):
    """(unchanged)"""
    # … existing code …
    """Show history-based recommendations"""
    st.header("📚 History-based Recommendations")
    
    with st.spinner("Analyzing your purchase history..."):
        results = engine.get_user_history_recommendations(user_id, user_location, top_k=20)
    
    if results:
        st.subheader(f"Personalized Recommendations for {user_id}")
        
        # Create display DataFrame with distance information
        display_data = []
        for result in results:
            # Get distance information for this product
            product_id = result['product_id']
            distance_info = engine.get_product_distance_info(product_id, user_location)
            
            display_data.append({
                'Product': result['product_name'],
                'Category': result['product_category'],
                'Price (₹)': f"₹{result['product_price']:,.2f}",
                'Rating': f"{result['product_rating']:.1f} ⭐",
                'Similarity': f"{result['similarity_score']:.3f}",
                'Distance': f"{distance_info['distance_km']:.1f} km" if distance_info else "N/A",
                'Warehouse': distance_info['warehouse_id'] if distance_info else "N/A",
                'Common Tags': ', '.join(result.get('common_tags', [])[:3])
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True)
        
        # Purchase option
        if st.button("Purchase from History Recommendations"):
            selected_product = st.selectbox(
                "Select a product to purchase from History Recommendations:",
                options=[r['product_name'] for r in results]
            )
            try:
                product_id = next(r['product_id'] for r in results if r['product_name'] == selected_product)
                engine.add_user_purchase(user_id, product_id, user_location)
                st.success(f"✅ Purchased: {selected_product}")
                st.info("💡 Your purchase history will now influence future recommendations!")
                st.balloons()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error processing purchase: {e}")
    else:
        st.info("📝 No purchase history found. Try making some purchases in the 'General Recommendations' section to get personalized recommendations!")
        st.info("💡 The system will learn from your purchases to provide better recommendations.")


def show_analytics(engine, user_location, user_id):
    """(unchanged)"""
    # … existing code …
    st.header("📊 Analytics & Insights")
    
    # Get recommendations for analytics
    recommendations = engine.get_distance_optimized_recommendations(user_location, user_id, 50)
    
    if not recommendations.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Price vs Distance Analysis")
            fig = px.scatter(
                recommendations,
                x='distance_km',
                y='product_price',
                color='product_category',
                size='product_rating',
                hover_data=['product_name'],
                title="Product Price vs Distance from Warehouse"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏷️ Category Distribution")
            category_counts = recommendations['product_category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Product Categories Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analytics
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("💰 Shipping Cost Distribution")
            fig = px.histogram(
                recommendations,
                x='shipping_cost',
                nbins=20,
                title="Distribution of Shipping Costs"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("⭐ Rating vs Distance")
            fig = px.scatter(
                recommendations,
                x='distance_km',
                y='product_rating',
                color='product_category',
                title="Product Rating vs Distance"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📋 Summary Statistics")
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Total Products", len(recommendations))
        with col6:
            st.metric("Avg Distance", f"{recommendations['distance_km'].mean():.1f} km")
        with col7:
            st.metric("Avg Shipping Cost", f"₹{recommendations['shipping_cost'].mean():.2f}")
        with col8:
            st.metric("Avg Rating", f"{recommendations['product_rating'].mean():.1f}")


def main():
    st.set_page_config(
        page_title="Distance-Optimized Product Recommendations",
        page_icon="🛍️",
        layout="wide"
    )
    st.title("🛍️ Distance-Optimized Product Recommendation System")
    st.markdown("---")

    engine = initialize_engine()
    if engine is None:
        st.error("Please check your data files and try again.")
        return

    st.sidebar.header("🎯 User Preferences")
    available_zipcodes = engine.data_processor.get_available_zipcodes()
    if available_zipcodes:
        selected_zipcode = st.sidebar.selectbox(
            "Select Your Zip Code",
            options=available_zipcodes
        )
        st.session_state.user_id = str(selected_zipcode)
        user_location = engine.data_processor.get_location_for_zipcode(
            st.session_state.user_id
        )
        st.sidebar.text_input(
            "Your Location (auto-selected)",
            value=user_location,
            disabled=True
        )
    else:
        st.sidebar.warning("No zip codes available.")
        user_location = ""
        st.session_state.user_id = ""

    recommendation_type = st.sidebar.selectbox(
        "Recommendation Type",
        [
            "General Recommendations",
            "Semantic Search",
            "Tag-based Search",
            "History-based Recommendations"
        ]
    )

    if recommendation_type == "General Recommendations":
        show_general_recommendations(
            engine, user_location, st.session_state.user_id
        )
    elif recommendation_type == "Semantic Search":
        show_semantic_search(engine, user_location)
    elif recommendation_type == "Tag-based Search":
        show_tag_based_search(engine, user_location)
    elif recommendation_type == "History-based Recommendations":
        show_history_recommendations(
            engine, st.session_state.user_id, user_location
        )

    st.markdown("---")
    show_analytics(
        engine, user_location, st.session_state.user_id
    )

    with st.expander("🔧 Debug Information"):
        # … existing debug code …
        st.subheader("Purchase History Debug")
        
        # Show current purchase history
        try:
            if engine.user_history_collection:
                results = engine.user_history_collection.get(
                    where={"user_id": {"$eq": user_id}},
                    include=['metadatas']
                )
                if results and results['ids']:
                    st.write(f"**User {user_id} has {len(results['ids'])} purchases:**")
                    if results['metadatas']:
                        for i, metadata in enumerate(results['metadatas'] or []):
                            st.write(f"{i+1}. {metadata['product_name']} (ID: {metadata['product_id']})")
                    else:
                        st.write("No purchases found")
                else:
                    st.write("No purchases found")
            else:
                st.write("❌ ChromaDB not available")
        except Exception as e:
            st.write(f"❌ Error retrieving purchase history: {e}")
        
        # Show recommendation engine status
        st.subheader("Engine Status")
        st.write(f"Products loaded: {engine.products_df is not None}")
        st.write(f"Warehouses loaded: {engine.warehouses_df is not None}")
        st.write(f"ChromaDB available: {engine.user_history_collection is not None}")
        st.write(f"GenAI module loaded: {engine.genai_search is not None}")


if __name__ == "__main__":
    main()
