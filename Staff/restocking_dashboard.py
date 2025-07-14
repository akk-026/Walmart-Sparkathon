# file: restocking_dashboard.py

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import numpy as np
import pgeocode
import math

st.set_page_config(layout="wide")
st.title("📦 Product Restocking Optimization Dashboard")

# Sidebar input
shipping_file = st.sidebar.file_uploader("Upload Shipping CSV", type=["csv"])
warehouse_file = st.sidebar.file_uploader("Upload Warehouse CSV", type=["csv"])

if shipping_file and warehouse_file:
    nomi = pgeocode.Nominatim('IN')
    
    # Load files
    shipping = pd.read_csv(shipping_file)
    warehouse = pd.read_csv(warehouse_file)

    # Normalize and prepare data
    warehouse['zip_code'] = warehouse['location_of_warehouse'].str.extract(r'(\d{6})')
    shipping['product_id'] = shipping['product_id'].astype(str).str.strip()
    warehouse['product_id'] = warehouse['product_id'].astype(str).str.strip()
    warehouse['warehouse_id'] = warehouse['warehouse_id'].astype(str).str.strip()
    warehouse['zip_code'] = warehouse['zip_code'].astype(str).str.strip()

    # Get coordinates
    shipping[['latitude', 'longitude']] = shipping['customer_zipcode'].apply(
        lambda z: nomi.query_postal_code(str(z))[['latitude', 'longitude']]
    ).apply(pd.Series)

    warehouse[['latitude', 'longitude']] = warehouse['zip_code'].apply(
        lambda z: nomi.query_postal_code(str(z))[['latitude', 'longitude']]
    ).apply(pd.Series)

    # Drop invalid entries
    shipping.dropna(subset=['latitude', 'longitude'], inplace=True)
    warehouse.dropna(subset=['latitude', 'longitude'], inplace=True)

    # Create lookup tables
    warehouse_locations = warehouse.groupby('warehouse_id').agg({
        'latitude': 'first',
        'longitude': 'first',
        'location_of_warehouse': 'first'
    }).reset_index()

    product_warehouse_stock = warehouse.groupby(['warehouse_id', 'product_id']).agg({
        'stock_count': 'sum',
        'storage_cost_per_unit_perMonth': 'first'
    }).reset_index()

    # Get common products
    common_products = set(shipping['product_id']).intersection(set(warehouse['product_id']))

    # Select product to analyze
    selected_product = st.selectbox("Select Product", sorted(common_products))

    if selected_product:
        product_orders = shipping[shipping['product_id'] == selected_product]

        if len(product_orders) < 2:
            st.warning("Not enough orders to perform clustering.")
        else:
            # Cluster customers
            k = min(5, len(product_orders))
            coords = product_orders[['latitude', 'longitude']]
            kmeans = KMeans(n_clusters=k, random_state=42)
            product_orders = product_orders.copy()
            product_orders['cluster'] = kmeans.fit_predict(coords)

            cluster_centroids = kmeans.cluster_centers_

            # Run optimization
            restocking_results = []
            product_warehouses = product_warehouse_stock[
                product_warehouse_stock['product_id'] == selected_product
            ]

            for cluster_id, (lat, lon) in enumerate(cluster_centroids):
                demand = len(product_orders[product_orders['cluster'] == cluster_id])
                min_dist = float('inf')
                closest_wh, current_stock, storage_cost = None, 0, 0

                for _, pw in product_warehouses.iterrows():
                    wh_info = warehouse_locations[warehouse_locations['warehouse_id'] == pw['warehouse_id']]
                    if wh_info.empty:
                        continue
                    wh_lat = wh_info.iloc[0]['latitude']
                    wh_lon = wh_info.iloc[0]['longitude']
                    dist = geodesic((lat, lon), (wh_lat, wh_lon)).km
                    if dist < min_dist:
                        min_dist = dist
                        closest_wh = pw['warehouse_id']
                        current_stock = pw['stock_count']
                        storage_cost = pw['storage_cost_per_unit_perMonth']

                restock_qty = math.ceil(max(demand - current_stock, 0) * 1.2)
                restocking_results.append({
                    'Cluster': cluster_id,
                    'Demand': demand,
                    'Warehouse ID': closest_wh,
                    'Distance (km)': round(min_dist, 2),
                    'Current Stock': current_stock,
                    'Restock Qty': restock_qty,
                    'Storage Cost/unit': storage_cost,
                    'Total Cost': restock_qty * storage_cost
                })

            # Show DataFrame
            result_df = pd.DataFrame(restocking_results)
            st.dataframe(result_df)

            # Total summary
            st.subheader("📊 Summary")
            st.write(f"**Total restock quantity:** {result_df['Restock Qty'].sum()}")
            st.write(f"**Estimated total cost:** ₹{result_df['Total Cost'].sum():,.2f}")
else:
    st.info("📁 Please upload both CSV files to continue.")
