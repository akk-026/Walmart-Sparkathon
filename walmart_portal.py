import streamlit as st
import pandas as pd
import numpy as np
import math
import logging
import pgeocode
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# Recommendation imports
from recommendation_engine import RecommendationEngine
from utils.api_client import rank_products_by_relevance

# ─── PAGE CONFIG & LOGGER ───────────────────────────────────────────────────────
st.set_page_config(page_title="Walmart Unified Portal", page_icon="🛍️", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── RECOMMENDATION ENGINE FUNCTIONS ───────────────────────────────────────────
def initialize_engine():
    if 'recommendation_engine' not in st.session_state or st.session_state.recommendation_engine is None:
        with st.spinner("Loading recommendation engine..."):
            engine = RecommendationEngine()
            if engine.load_data():
                st.session_state.recommendation_engine = engine
                st.success("Recommendation engine loaded.")
            else:
                st.error("Failed to load recommendation engine.")
                return None
    return st.session_state.recommendation_engine

def show_general_recommendations(engine, user_location, user_id):
    st.header("Distance-Optimized Recommendations")
    recs = engine.get_distance_optimized_recommendations(user_location, user_id, 20)
    if recs.empty:
        st.warning("No recommendations available for your location.")
        return
    st.subheader(f"Top Recommendations for {user_location}")
    df = recs.copy()
    disp = df[['product_name','product_category','product_price','product_rating','distance_km','shipping_cost','total_cost','final_score']].copy()
    disp.columns = ['Product','Category','Price (₹)','Rating','Distance (km)','Shipping (₹)','Total Cost (₹)','Score']
    disp['Price (₹)'] = disp['Price (₹)'].map(lambda x: f"₹{x:,.2f}")
    disp['Shipping (₹)'] = disp['Shipping (₹)'].map(lambda x: f"₹{x:,.2f}")
    disp['Total Cost (₹)'] = disp['Total Cost (₹)'].map(lambda x: f"₹{x:,.2f}")
    disp['Rating'] = disp['Rating'].map(lambda x: f"{x:.1f} ⭐")
    disp['Distance (km)'] = disp['Distance (km)'].map(lambda x: f"{x:.1f} km")
    disp['Score'] = disp['Score'].map(lambda x: f"{x:.3f}")
    st.dataframe(disp, use_container_width=True)
    st.subheader("Simulate Purchase")
    sel = st.selectbox("Pick product to buy:", df['product_name'].tolist(), key="gen_sel")
    if st.button("Purchase", key="gen_buy"):
        row = df[df['product_name']==sel].iloc[0]
        engine.add_user_purchase(user_id, row['product_id'], user_location)
        st.success(f"Purchased {sel}")
        st.balloons()
        st.experimental_rerun()

def show_semantic_search(engine, user_location):
    st.header("Semantic Product Search")
    query = st.text_input("Search query:", key="sem_q")
    if not query: return
    results = engine.semantic_search(query, top_k=50, user_location=user_location, user_id=st.session_state.user_id)
    if not results:
        st.warning("No products found.")
        return
    df = pd.DataFrame(results)
    ordered = []
    cats = list(dict.fromkeys(df['product_category']))
    for cat in cats:
        grp = df[df['product_category']==cat]
        ids = grp['product_id'].tolist()
        sorted_ids = rank_products_by_relevance(ids, st.session_state.user_id)
        ordered.append(grp.set_index('product_id').loc[sorted_ids].reset_index())
    final = pd.concat(ordered, ignore_index=True)
    display = []
    for _, r in final.iterrows():
        d = engine.get_product_distance_info(r['product_id'], user_location) or {}
        display.append({
            'Product': r['product_name'],
            'Category': r['product_category'],
            'Price (₹)': f"₹{r['product_price']:,.2f}",
            'Rating': f"{r['product_rating']:.1f} ⭐",
            'Similarity': f"{r['similarity_score']:.3f}",
            'Distance': f"{d.get('distance_km',0):.1f} km",
            'Warehouse': d.get('warehouse_id',"N/A")
        })
    st.dataframe(pd.DataFrame(display), use_container_width=True)
    st.subheader("Purchase Product")
    sel = st.selectbox("Choose product:", final['product_name'].tolist(), key="sem_buy_sel")
    if st.button("Buy", key="sem_buy_btn"):
        pid = final[final['product_name']==sel]['product_id'].iloc[0]
        engine.add_user_purchase(st.session_state.user_id, pid, user_location)
        st.success(f"Purchased {sel}")
        st.balloons()
        st.experimental_rerun()

def show_tag_based_search(engine, user_location):
    st.header("Tag-based Product Search")
    tags = engine.get_available_tags()
    sel_tags = st.multiselect("Select tags:", tags, key="tag_sel")
    if not sel_tags: return
    results = engine.search_by_tags(sel_tags, top_k=50, user_location=user_location, user_id=st.session_state.user_id)
    if not results:
        st.warning("No matches for selected tags.")
        return
    df = pd.DataFrame(results)
    ordered = []
    cats = list(dict.fromkeys(df['product_category']))
    for cat in cats:
        grp = df[df['product_category']==cat]
        ids = grp['product_id'].tolist()
        sorted_ids = rank_products_by_relevance(ids, st.session_state.user_id)
        ordered.append(grp.set_index('product_id').loc[sorted_ids].reset_index())
    final = pd.concat(ordered, ignore_index=True)
    display = []
    for _, r in final.iterrows():
        d = engine.get_product_distance_info(r['product_id'], user_location) or {}
        display.append({
            'Product': r['product_name'],
            'Category': r['product_category'],
            'Price (₹)': f"₹{r['product_price']:,.2f}",
            'Rating': f"{r['product_rating']:.1f} ⭐",
            'Distance': f"{d.get('distance_km',0):.1f} km",
            'Warehouse': d.get('warehouse_id',"N/A")
        })
    st.dataframe(pd.DataFrame(display), use_container_width=True)
    sel = st.selectbox("Choose product to buy:", final['product_name'].tolist(), key="tag_buy_sel")
    if st.button("Buy", key="tag_buy_btn"):
        pid = final[final['product_name']==sel]['product_id'].iloc[0]
        engine.add_user_purchase(st.session_state.user_id, pid, user_location)
        st.success(f"Purchased {sel}")
        st.balloons()
        st.experimental_rerun()

def show_history_recommendations(engine, user_id, user_location):
    st.header("History-based Recommendations")
    results = engine.get_user_history_recommendations(user_id, user_location, top_k=20)
    if not results:
        st.info("No purchase history yet.")
        return
    df = pd.DataFrame(results)
    display = []
    for _, r in df.iterrows():
        d = engine.get_product_distance_info(r['product_id'], user_location) or {}
        display.append({
            'Product': r['product_name'],
            'Category': r['product_category'],
            'Similarity': f"{r['similarity_score']:.3f}",
            'Distance': f"{d.get('distance_km',0):.1f} km"
        })
    st.dataframe(pd.DataFrame(display), use_container_width=True)
    sel = st.selectbox("Choose to buy:", df['product_name'].tolist(), key="hist_sel")
    if st.button("Buy", key="hist_buy_btn"):
        pid = df[df['product_name']==sel]['product_id'].iloc[0]
        engine.add_user_purchase(user_id, pid, user_location)
        st.success(f"Purchased {sel}")
        st.balloons()
        st.experimental_rerun()

def show_analytics(engine, user_location, user_id):
    st.header("Analytics & Insights")
    recs = engine.get_distance_optimized_recommendations(user_location, user_id, 50)
    if recs.empty:
        st.warning("No data for analytics.")
        return
    col1,col2 = st.columns(2)
    with col1:
        fig = px.scatter(recs, x='distance_km', y='product_price',
                         color='product_category', size='product_rating',
                         hover_data=['product_name'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cnt = recs['product_category'].value_counts()
        fig = px.pie(values=cnt.values, names=cnt.index)
        st.plotly_chart(fig, use_container_width=True)
    stats = st.columns(4)
    stats[0].metric("Total Products", len(recs))
    stats[1].metric("Avg Distance", f"{recs['distance_km'].mean():.1f} km")
    stats[2].metric("Avg Shipping Cost", f"₹{recs['shipping_cost'].mean():.2f}")
    stats[3].metric("Avg Rating", f"{recs['product_rating'].mean():.1f}")

# ─── STAFF PORTAL ───────────────────────────────────────────────────────────────
def run_staff_portal():
    st.title("Staff Portal: Restocking Optimization")
    shipping_file = st.sidebar.file_uploader("Shipping CSV", type=["csv"], key="s1")
    warehouse_file = st.sidebar.file_uploader("Warehouse CSV", type=["csv"], key="s2")
    if not (shipping_file and warehouse_file):
        st.info("Upload both shipping & warehouse CSVs to start.")
        return

    nomi = pgeocode.Nominatim('IN')
    shipping = pd.read_csv(shipping_file)
    warehouse = pd.read_csv(warehouse_file)
    products_df = pd.read_csv("data/Product.csv")

    warehouse["zip_code"] = warehouse["location_of_warehouse"].str.extract(r"(\d{6})")
    for df, col in [(shipping,"product_id"),(warehouse,"product_id"),(warehouse,"warehouse_id"),(warehouse,"zip_code")]:
        df[col] = df[col].astype(str).str.strip()
    products_df["product_id"] = products_df["product_id"].astype(str).str.strip()

    shipping[["latitude","longitude"]] = shipping["customer_zipcode"]\
        .apply(lambda z: nomi.query_postal_code(str(z))[["latitude","longitude"]])\
        .apply(pd.Series)
    warehouse[["latitude","longitude"]] = warehouse["zip_code"]\
        .apply(lambda z: nomi.query_postal_code(str(z))[["latitude","longitude"]])\
        .apply(pd.Series)

    shipping.dropna(subset=["latitude","longitude"], inplace=True)
    warehouse.dropna(subset=["latitude","longitude"], inplace=True)

    warehouse_locations = warehouse.groupby("warehouse_id").agg({
        "latitude":"first","longitude":"first","location_of_warehouse":"first"
    }).reset_index()
    product_warehouse_stock = warehouse.groupby(["warehouse_id","product_id"]).agg({
        "stock_count":"sum","storage_cost_per_unit_perMonth":"first"
    }).reset_index()

    common = set(shipping["product_id"]) & set(warehouse["product_id"])
    opts = [f"{pid} - {products_df[products_df['product_id']==pid]['product_name'].iloc[0]}"
            for pid in sorted(common)]
    sel = st.selectbox("Select Product", opts, key="staff_sel")
    pid = sel.split(" - ")[0]
    orders = shipping[shipping["product_id"]==pid]
    if len(orders)<2:
        st.warning("Not enough orders.")
        return

    max_k = min(20,len(orders))
    k = st.slider("Clusters", 2, max_k, value=min(5,max_k), key="staff_k")
    orders = orders.copy()
    orders["cluster"] = KMeans(n_clusters=k, random_state=42)\
        .fit_predict(orders[["latitude","longitude"]])
    centroids = KMeans(n_clusters=k, random_state=42)\
        .fit(orders[["latitude","longitude"]]).cluster_centers_

    results, conns = [], []
    pw = product_warehouse_stock[product_warehouse_stock["product_id"]==pid]
    for cid,(clat,clon) in enumerate(centroids):
        demand = (orders["cluster"]==cid).sum()
        mind=1e9; best=None; stock=0; cost=0
        for _,r in pw.iterrows():
            w = warehouse_locations[warehouse_locations["warehouse_id"]==r["warehouse_id"]].iloc[0]
            dist = geodesic((clat,clon),(w["latitude"],w["longitude"])).km
            if dist<mind:
                mind, best, stock, cost = dist, r["warehouse_id"], r["stock_count"], r["storage_cost_per_unit_perMonth"]
                wcoord=(w["latitude"],w["longitude"])
        qty = math.ceil(max(demand-stock,30)*1.2)
        results.append({
            "Cluster":cid,"Demand":demand,"Warehouse ID":best,
            "Distance (km)":round(mind,2),"Current Stock":stock,
            "Restock Qty":qty,"Storage Cost/unit":cost,
            "Total Cost":qty*cost
        })
        if best:
            conns.append({
                "from_lat":wcoord[0],"from_lon":wcoord[1],
                "to_lat":clat,"to_lon":clon,
                "warehouse_id":best,"cluster_id":cid
            })

    df_res = pd.DataFrame(results)
    st.dataframe(df_res, use_container_width=True)
    st.subheader("Cluster Map & Restock Arrows")
    fig = go.Figure()

    for cid,(lat,lon) in enumerate(centroids):
        fig.add_trace(go.Scattermapbox(
            lat=[lat],lon=[lon],mode="markers+text",
            marker=dict(size=14,color="red"),text=f"Cluster {cid}",
            textposition="top center"
        ))
        circ_lats, circ_lons = [], []
        for a in np.linspace(0,2*math.pi,100):
            dlat = 0.09*math.cos(a); dlon = 0.09*math.sin(a)/math.cos(math.radians(lat))
            circ_lats.append(lat+dlat); circ_lons.append(lon+dlon)
        fig.add_trace(go.Scattermapbox(
            lat=circ_lats+[circ_lats[0]],lon=circ_lons+[circ_lons[0]],
            mode="lines",fill="toself",fillcolor="rgba(255,0,0,0.15)",
            line=dict(color="red",width=1),showlegend=False
        ))

    wsub = warehouse_locations[warehouse_locations["warehouse_id"].isin(df_res["Warehouse ID"])]
    fig.add_trace(go.Scattermapbox(
        lat=wsub["latitude"],lon=wsub["longitude"],mode="markers+text",
        marker=dict(size=10,color="blue"),text=wsub["warehouse_id"],
        textposition="bottom right"
    ))

    for c in conns:
        angle = math.degrees(math.atan2(
            c["to_lat"]-c["from_lat"],
            (c["to_lon"]-c["from_lon"])*math.cos(math.radians(c["to_lat"]))
        ))
        fig.add_trace(go.Scattermapbox(
            lat=[c["from_lat"],c["to_lat"]],lon=[c["from_lon"],c["to_lon"]],
            mode="lines+markers",
            line=dict(color="green",width=2),
            marker=dict(size=10,symbol="arrow",angle=angle,color="green"),
            hoverinfo="text",
            text=[f'WH {c["warehouse_id"]}→C{c["cluster_id"]}']*2
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=4.5,
        mapbox_center={"lat":orders["latitude"].mean(),"lon":orders["longitude"].mean()},
        margin=dict(l=0,r=0,t=0,b=0),height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Summary")
    st.write(f"Total Restock Qty: {df_res['Restock Qty'].sum()}")
    st.write(f"Estimated Cost: ₹{df_res['Total Cost'].sum():,.2f}")

# ─── USER PORTAL ────────────────────────────────────────────────────────────────
def run_user_portal():
    engine = initialize_engine()
    if engine is None:
        return
    st.sidebar.header("User Settings")
    zips = engine.data_processor.get_available_zipcodes()
    if zips:
        uz = st.sidebar.selectbox("Zip Code", zips, key="user_zip")
        st.session_state.user_id = str(uz)
        ul = engine.data_processor.get_location_for_zipcode(str(uz))
        st.sidebar.text_input("Location", value=ul, disabled=True)
    else:
        st.sidebar.warning("No zip codes available.")
        ul = ""
        st.session_state.user_id = ""

    choice = st.sidebar.selectbox("Choose Function",[
        "General Recommendations",
        "Semantic Search",
        "Tag-based Search",
        "History-based Recommendations",
        "Analytics & Insights"
    ], key="user_choice")

    if choice == "General Recommendations":
        show_general_recommendations(engine, ul, st.session_state.user_id)
    elif choice == "Semantic Search":
        show_semantic_search(engine, ul)
    elif choice == "Tag-based Search":
        show_tag_based_search(engine, ul)
    elif choice == "History-based Recommendations":
        show_history_recommendations(engine, st.session_state.user_id, ul)
    else:
        show_analytics(engine, ul, st.session_state.user_id)

# ─── MAIN SWITCH ───────────────────────────────────────────────────────────────
def main():
    portal = st.sidebar.radio("Select Portal", ["User Portal","Staff Portal"], key="main_switch")
    st.sidebar.markdown("---")
    if portal == "User Portal":
        run_user_portal()
    else:
        run_staff_portal()

if __name__ == "__main__":
    main()
