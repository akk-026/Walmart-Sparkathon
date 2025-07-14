# 🛍️ Distance-Optimized Product Recommendation System

A sophisticated recommendation system that optimizes product suggestions based on user location, shipping costs, and product ratings to minimize delivery expenses while maximizing user satisfaction.

## 🎯 Features

### Core Functionality
- **Distance-Based Optimization**: Recommends products from the nearest warehouses to reduce shipping costs
- **Semantic Search**: Uses transformer models for intelligent product matching
- **User History Tracking**: Stores purchase history in ChromaDB for personalized recommendations
- **Real-time Analytics**: Interactive visualizations showing cost vs. distance analysis

### Recommendation Types
1. **General Recommendations**: Based on location proximity and product popularity
2. **Search-Based**: Semantic search using natural language queries
3. **History-Based**: Personalized recommendations from purchase history

### Key Optimizations
- **Shipping Cost Calculation**: Estimates delivery costs based on distance
- **Multi-factor Scoring**: Combines distance, rating, and price for optimal recommendations
- **Warehouse Proximity**: Prioritizes products from nearby warehouses

## 🏗️ Architecture

### Components
1. **Data Processor** (`data_processor.py`): Handles CSV data and geocoding
2. **Recommendation Engine** (`recommendation_engine.py`): Core recommendation logic with transformers
3. **Streamlit Frontend** (`app.py`): Beautiful web interface
4. **ChromaDB**: Vector database for storing embeddings and user history

### Data Sources
- `Product.csv`: Product catalog with ratings and categories
- `Warehouse.csv`: Warehouse locations and inventory
- `Shipping_matrix.csv`: Historical shipping data

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Application
Run a separate terminal and start the FastAPI server on port 8000

```bash
uvicorn backend.api:app --reload --port 8000
```
Then open a new terminal and start the Streamlit app
```bash
streamlit run walmart_portal.py
```

The application will be available at `http://localhost:8501`

## 📊 How It Works

### 1. Data Processing
- Loads product and warehouse data from CSV files
- Geocodes warehouse locations for distance calculations
- Creates product embeddings using sentence transformers

### 2. Recommendation Algorithm
The system uses a weighted scoring approach:

```
Final Score = (0.4 × Distance Score) + (0.4 × Rating Score) + (0.2 × Price Score)
```

Where:
- **Distance Score**: Inverse of distance (closer = higher score)
- **Rating Score**: Normalized product rating (0-5 scale)
- **Price Score**: Inverse of price (lower cost = higher score)

### 3. Distance Optimization
- Calculates geodesic distance between user and warehouses
- Estimates shipping costs based on distance
- Prioritizes products from nearest warehouses

### 4. User Experience
- **Interactive Interface**: Clean, modern Streamlit UI
- **Real-time Analytics**: Visual charts showing cost vs. distance relationships
- **Purchase Tracking**: Stores user history for future recommendations

## 🎨 Features

### Frontend Features
- **Responsive Design**: Works on desktop and mobile
- **Interactive Charts**: Plotly visualizations for data analysis
- **Product Cards**: Clean display of product information
- **Purchase Simulation**: Track user purchases for history-based recommendations

### Analytics Dashboard
- **Price vs Distance Analysis**: Scatter plots showing cost-distance relationships
- **Category Distribution**: Pie charts of product categories
- **Shipping Cost Analysis**: Histograms of delivery costs
- **Rating Analysis**: Product ratings vs. distance correlation

## 🔧 Configuration

### Environment Variables
- `CHROMA_PERSIST_DIRECTORY`: ChromaDB storage location (default: `./chroma_db`)

### Customization
- **Scoring Weights**: Adjust the weights in the recommendation algorithm
- **Distance Calculation**: Modify the shipping cost estimation formula
- **UI Styling**: Customize CSS in the Streamlit app

## 📈 Performance Metrics

The system tracks:
- **Average Distance**: Mean distance to recommended products
- **Shipping Cost Savings**: Reduction in delivery costs
- **User Satisfaction**: Based on product ratings
- **Recommendation Accuracy**: Semantic similarity scores

## 🛠️ Technical Stack

- **Backend**: Python, Pandas, NumPy
- **ML/AI**: Sentence Transformers, ChromaDB
- **Frontend**: Streamlit, Plotly
- **Geospatial**: Geopy for distance calculations
- **Database**: ChromaDB for vector storage

## 🎯 Use Cases

### E-commerce Optimization
- Reduce shipping costs for customers
- Improve delivery times
- Increase customer satisfaction

### Supply Chain Management
- Optimize warehouse locations
- Minimize transportation costs
- Improve inventory distribution

### Customer Experience
- Personalized recommendations
- Transparent pricing (product + shipping)
- Location-based product discovery

## 🔮 Future Enhancements

1. **Real-time Inventory**: Live stock updates
2. **Weather Integration**: Consider weather impact on shipping
3. **Multi-modal Transport**: Different shipping methods
4. **Predictive Analytics**: Forecast demand patterns
5. **Mobile App**: Native mobile application

## 📝 Project Structure

```
sparkathon2/
├── app.py                 # Streamlit frontend
├── data_processor.py      # Data handling and geocoding
├── recommendation_engine.py # Core recommendation logic
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── Product.csv           # Product catalog
├── Warehouse.csv         # Warehouse data
└── Shipping_matrix.csv   # Shipping information
```

## 🤝 Contributing

This is a group project for Sparkathon 2. The system demonstrates:
- Advanced recommendation algorithms
- Distance optimization techniques
- Modern web application development
- Data science and machine learning integration

## 📄 License

This project is part of the Sparkathon 2 competition and is developed for educational purposes.

---

**Built with ❤️ for Sparkathon 2** 🚀 
