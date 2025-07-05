import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
import warnings
from utils import load_or_train_model, calculate_affordability_index, generate_market_insights

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base CSS styling that will be enhanced by theme-specific CSS
st.markdown("""
<style>
/* Header styling with house icon background */
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '🏠🏢🏘️';
    position: absolute;
    top: 10px;
    right: 20px;
    font-size: 2rem;
    opacity: 0.2;
    z-index: 1;
}

.main-header > * {
    position: relative;
    z-index: 2;
}

/* Enhanced metric cards with house patterns */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin: 0.5rem 0;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '🏠';
    position: absolute;
    top: 10px;
    right: 15px;
    font-size: 3rem;
    opacity: 0.2;
    z-index: 1;
}

/* Common styles for all themes */
.insight-box {
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}

.insight-box::before {
    content: '💡';
    position: absolute;
    top: 10px;
    right: 15px;
    font-size: 1.5rem;
    opacity: 0.5;
}

.warning-box {
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}

.warning-box::before {
    content: '⚠️';
    position: absolute;
    top: 10px;
    right: 15px;
    font-size: 1.5rem;
    opacity: 0.5;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-radius: 10px;
    padding: 0.5rem;
    backdrop-filter: blur(10px);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

.sub-header {
    font-size: 1.2rem;
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
    border-radius: 10px;
    backdrop-filter: blur(10px);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: white;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load model and data
@st.cache_data
def load_data():
    """Load the house price dataset"""
    try:
        df = pd.read_csv('Bengaluru_House_Data.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'Bengaluru_House_Data.csv' is in the project directory.")
        return None

@st.cache_resource
def load_model():
    """Load or train the machine learning model"""
    return load_or_train_model()



# Main application
def main():
    # Add clean white background
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                background: #ffffff;
                z-index: -1000; pointer-events: none;">
    </div>
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; height: 120px; 
                background: linear-gradient(to top, rgba(102, 126, 234, 0.05), transparent);
                z-index: -999; pointer-events: none;">
        <div style="position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
                    font-size: 2rem; opacity: 0.05; white-space: nowrap; color: #667eea;">
            🏠 🏢 🏘️ 🏙️ 🏠 🏢 🏘️ 🏙️ 🏠 🏢 🏘️ 🏙️
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply clean white theme styling
    st.markdown("""
    <style>
    /* Ensure all backgrounds are white */
    .stApp {
        background: #ffffff !important;
    }
    
    .main {
        background: #ffffff !important;
    }
    
    .main .block-container {
        background: #ffffff !important;
        padding: 2rem 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        position: relative;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar background */
    .css-1d391kg {
        background: #ffffff !important;
    }
    
    .sidebar .sidebar-content {
        background: #ffffff !important;
    }
    
    .sub-header {
        color: #333;
        background: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.1);
        border: 1px solid rgba(40, 167, 69, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #333 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(102, 126, 234, 0.2) !important;
        color: #000 !important;
    }
    
    /* Force white background on all elements */
    div[data-testid="stSidebar"] {
        background: #ffffff !important;
    }
    
    .css-1lcbmhc {
        background: #ffffff !important;
    }
    
    .css-12oz5g7 {
        background: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏠 Bengaluru House Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive ML-powered tool for 3rd Year Computer Science Students</p>', unsafe_allow_html=True)
    
    # Load data and model
    df = load_data()
    model_data = load_model()
    
    if df is None or model_data is None:
        st.stop()
    
    model = model_data['model']
    location_encoder = model_data['location_encoder']
    locations = model_data['locations']
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Price Prediction", 
        "📊 Data Exploration", 
        "🧠 ML Insights", 
        "📈 Market Analysis", 
        "📚 Learning Resources"
    ])
    
    with tab1:
        prediction_interface(model, location_encoder, locations, df)
    
    with tab2:
        data_exploration(df)
    
    with tab3:
        ml_insights(model, location_encoder, locations, df)
    
    with tab4:
        market_analysis(df)
    
    with tab5:
        learning_resources()

def prediction_interface(model, location_encoder, locations, df):
    """Interactive prediction interface"""
    # Add prediction-specific background with house images
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0iIzY2N2VlYSIgZmlsbC1vcGFjaXR5PSIwLjAzIj4KICAgIDxwYXRoIGQ9Ik01MCAzMEw3MCA1NUgzMEw1MCAzMHogTTMwIDU1aDQwdjUwSDMwVjU1eiBNMzggNzBoMjR2MjVIMzhWNzB6IE00MiA3NWgxNnYxNUg0MlY3NXogTTEwMCA1MEwxMjAgNzVIODBMMTAwIDUwek0gODAgNzVoNDB2NTBIODBWNzV6IE04OCA5MGgyNHYyNUg4OFY5MHogTTkyIDk1aDE2djE1SDkyVjk1eiBNMTUwIDQwTDE3MCA2NUgxMzBMMTUwIDQwek0gMTMwIDY1aDQwdjUwSDEzMFY2NXogTTEzOCA4MGgyNHYyNUgxMzhWODB6IE0xNDIgODVoMTZ2MTVIMTQyVjg1eiIvPgogIDwvZz4KPC9zdmc+');
                background-size: 200px 200px;
                background-repeat: repeat;
                z-index: -1000; pointer-events: none;">
    </div>
    <div style="position: fixed; top: 10%; right: 5%; font-size: 4rem; opacity: 0.05; z-index: -999;">
        🔮✨💰🏠📊
    </div>
    """, unsafe_allow_html=True)
    
    st.header("🔮 Real-time Price Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Property Details")
        
        # Location selection
        location = st.selectbox(
            "📍 Select Location",
            options=locations,
            help="Choose from popular Bengaluru locations"
        )
        
        # Square feet input
        sqft = st.number_input(
            "📏 Total Square Feet",
            min_value=300,
            max_value=10000,
            value=1200,
            step=50,
            help="Enter the total area in square feet"
        )
        
        # Number of bathrooms
        bath = st.slider(
            "🚿 Number of Bathrooms",
            min_value=1,
            max_value=6,
            value=2,
            help="Select number of bathrooms"
        )
        
        # Number of BHK
        bhk = st.slider(
            "🏠 Number of BHK",
            min_value=1,
            max_value=5,
            value=2,
            help="Select number of bedrooms, hall, and kitchen"
        )
        
        # Real-time prediction
        if st.button("🎯 Predict Price", type="primary"):
            prediction = make_prediction(model, location_encoder, location, sqft, bath, bhk)
            
            if prediction:
                # Store in history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'location': location,
                    'sqft': sqft,
                    'bath': bath,
                    'bhk': bhk,
                    'predicted_price': prediction['price'],
                    'price_per_sqft': prediction['price_per_sqft']
                })
                
                # Display prediction in the second column
                with col2:
                    display_prediction_results(prediction, location, sqft, bath, bhk, df)
    
    # Prediction history
    if st.session_state.prediction_history:
        st.subheader("📋 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display recent predictions
        st.dataframe(
            history_df.tail(10),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv_data = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prediction History",
            data=csv_data,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def make_prediction(model, location_encoder, location, sqft, bath, bhk):
    """Make price prediction"""
    try:
        # Encode location
        location_encoded = location_encoder.transform([location])[0]
        
        # Create feature array
        features = np.array([[location_encoded, sqft, bath, bhk]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Ensure positive prediction
        if prediction < 0:
            prediction = abs(prediction)
        
        # Calculate insights
        price_per_sqft = (prediction * 100000) / sqft
        
        return {
            'price': round(prediction, 2),
            'price_per_sqft': round(price_per_sqft, 2)
        }
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def display_prediction_results(prediction, location, sqft, bath, bhk, df):
    """Display prediction results with insights"""
    st.subheader("🎯 Prediction Results")
    
    # Main prediction display
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="margin:0; color: white;">₹{prediction['price']:.2f} Lakhs</h2>
        <p style="margin:0; color: #e0e0e0;">Estimated Property Value</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced visibility for key metrics
    st.markdown("---")
    st.subheader("🔍 **Detailed Analysis**")
    
    # Price per square foot with enhanced styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 15px; margin: 10px 0;
                    text-align: center; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
            <h3 style="margin: 0; font-size: 1.2em;">💰 Price per Sq Ft</h3>
            <h1 style="margin: 10px 0; font-size: 2.5em; font-weight: bold;">₹{prediction['price_per_sqft']:.0f}</h1>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">Calculated price per square foot</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        affordability_index = calculate_affordability_index(prediction['price'])
        # Color coding for affordability
        if affordability_index >= 8:
            color = "#28a745"  # Green
            status = "Highly Affordable"
        elif affordability_index >= 6:
            color = "#ffc107"  # Yellow
            status = "Moderately Affordable"
        elif affordability_index >= 4:
            color = "#fd7e14"  # Orange
            status = "Less Affordable"
        else:
            color = "#dc3545"  # Red
            status = "Expensive"
            
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, {color}aa 100%); 
                    color: white; padding: 20px; border-radius: 15px; margin: 10px 0;
                    text-align: center; box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);">
            <h3 style="margin: 0; font-size: 1.2em;">📊 Affordability Index</h3>
            <h1 style="margin: 10px 0; font-size: 2.5em; font-weight: bold;">{affordability_index}/10</h1>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9em;">{status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Market comparison with enhanced visibility
    location_data = df[df['location'] == location]
    if not location_data.empty:
        avg_price = location_data['price'].mean()
        comparison = ((prediction['price'] - avg_price) / avg_price) * 100
        
        st.markdown("### 📈 **Market Comparison**")
        
        if comparison > 0:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
                        color: white; padding: 25px; border-radius: 15px; margin: 15px 0;
                        border-left: 5px solid #ff4757; box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);">
                <h3 style="margin: 0 0 10px 0; display: flex; align-items: center;">
                    <span style="font-size: 1.5em; margin-right: 10px;">⚠️</span>
                    Above Market Average
                </h3>
                <p style="margin: 0; font-size: 1.1em; font-weight: 500;">
                    This property is <strong>{comparison:.1f}% above average</strong> for {location}
                </p>
                <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Average price in {location}: ₹{avg_price:.2f} lakhs
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        color: white; padding: 25px; border-radius: 15px; margin: 15px 0;
                        border-left: 5px solid #28a745; box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);">
                <h3 style="margin: 0 0 10px 0; display: flex; align-items: center;">
                    <span style="font-size: 1.5em; margin-right: 10px;">✅</span>
                    Below Market Average
                </h3>
                <p style="margin: 0; font-size: 1.1em; font-weight: 500;">
                    This property is <strong>{abs(comparison):.1f}% below average</strong> for {location}
                </p>
                <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 0.9em;">
                    Average price in {location}: ₹{avg_price:.2f} lakhs
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Property insights
    insights = generate_market_insights(location, sqft, bath, bhk, prediction['price'])
    for insight in insights:
        st.info(insight)

def data_exploration(df):
    """Data exploration interface"""
    # Add data exploration background with chart patterns
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0iIzI4YTc0NSIgZmlsbC1vcGFjaXR5PSIwLjAzIj4KICAgIDxyZWN0IHg9IjEwIiB5PSI0MCIgd2lkdGg9IjEwIiBoZWlnaHQ9IjIwIi8+CiAgICA8cmVjdCB4PSIzMCIgeT0iMzAiIHdpZHRoPSIxMCIgaGVpZ2h0PSIzMCIvPgogICAgPHJlY3QgeD0iNTAiIHk9IjIwIiB3aWR0aD0iMTAiIGhlaWdodD0iNDAiLz4KICAgIDxyZWN0IHg9IjcwIiB5PSIxMCIgd2lkdGg9IjEwIiBoZWlnaHQ9IjUwIi8+CiAgPC9nPgo8L3N2Zz4=');
                background-size: 100px 100px;
                background-repeat: repeat;
                z-index: -1000; pointer-events: none;">
    </div>
    <div style="position: fixed; top: 15%; left: 5%; font-size: 3rem; opacity: 0.05; z-index: -999;">
        📊📈📉🔍📋
    </div>
    """, unsafe_allow_html=True)
    
    st.header("📊 Dataset Exploration")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df))
    with col2:
        st.metric("Unique Locations", df['location'].nunique())
    with col3:
        st.metric("Avg Price (Lakhs)", f"₹{df['price'].mean():.2f}")
    with col4:
        st.metric("Price Range", f"₹{df['price'].min():.1f} - ₹{df['price'].max():.1f}")
    
    # Interactive filters
    st.subheader("🔍 Interactive Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_locations = st.multiselect(
            "Select Locations",
            options=df['location'].unique(),
            default=df['location'].unique()[:5]
        )
    
    with col2:
        price_range = st.slider(
            "Price Range (Lakhs)",
            min_value=float(df['price'].min()),
            max_value=float(df['price'].max()),
            value=(float(df['price'].min()), float(df['price'].max()))
        )
    
    # Filter data
    filtered_df = df[
        (df['location'].isin(selected_locations)) &
        (df['price'] >= price_range[0]) &
        (df['price'] <= price_range[1])
    ]
    
    # Display filtered data
    st.subheader("📋 Filtered Dataset")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download filtered data
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv_data,
        file_name=f"filtered_bengaluru_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def ml_insights(model, location_encoder, locations, df):
    """Machine Learning insights and educational content"""
    # Add ML insights background with brain/network patterns
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjEyMCIgdmlld0JveD0iMCAwIDEyMCAxMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNzY0YmEyIiBzdHJva2Utd2lkdGg9IjEiIHN0cm9rZS1vcGFjaXR5PSIwLjAzIj4KICAgIDxjaXJjbGUgY3g9IjIwIiBjeT0iMjAiIHI9IjgiLz4KICAgIDxjaXJjbGUgY3g9IjEwMCIgY3k9IjIwIiByPSI4Ii8+CiAgICA8Y2lyY2xlIGN4PSI2MCIgY3k9IjYwIiByPSI4Ii8+CiAgICA8Y2lyY2xlIGN4PSIyMCIgY3k9IjEwMCIgcj0iOCIvPgogICAgPGNpcmNsZSBjeD0iMTAwIiBjeT0iMTAwIiByPSI4Ii8+CiAgICA8bGluZSB4MT0iMjAiIHkxPSIyMCIgeDI9IjYwIiB5Mj0iNjAiLz4KICAgIDxsaW5lIHgxPSIxMDAiIHkxPSIyMCIgeDI9IjYwIiB5Mj0iNjAiLz4KICAgIDxsaW5lIHgxPSI2MCIgeTE9IjYwIiB4Mj0iMjAiIHkyPSIxMDAiLz4KICAgIDxsaW5lIHgxPSI2MCIgeTE9IjYwIiB4Mj0iMTAwIiB5Mj0iMTAwIi8+CiAgPC9nPgo8L3N2Zz4=');
                background-size: 120px 120px;
                background-repeat: repeat;
                z-index: -1000; pointer-events: none;">
    </div>
    <div style="position: fixed; top: 10%; right: 10%; font-size: 3rem; opacity: 0.05; z-index: -999;">
        🧠⚡🔬💡🎯
    </div>
    """, unsafe_allow_html=True)
    
    st.header("🧠 Machine Learning Insights")
    
    # Model performance
    st.subheader("📈 Model Performance")
    
    # Calculate model metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    
    # Prepare data for evaluation
    X = df.copy()
    X['location_encoded'] = location_encoder.transform(X['location'])
    X_features = X[['location_encoded', 'total_sqft', 'bath', 'bhk']]
    y = X['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Absolute Error", f"₹{mae:.2f} Lakhs")
    with col2:
        st.metric("R² Score", f"{r2:.4f}")
    with col3:
        accuracy = max(0, (1 - mae/df['price'].mean()) * 100)
        st.metric("Model Accuracy", f"{accuracy:.1f}%")
    
    # Feature importance visualization
    st.subheader("🎯 Feature Importance")
    
    # Calculate feature importance (for linear regression, use coefficients)
    if hasattr(model, 'coef_'):
        feature_names = ['Location', 'Square Feet', 'Bathrooms', 'BHK']
        importance = np.abs(model.coef_)
        
        fig = px.bar(
            x=feature_names,
            y=importance,
            title="Feature Importance in Price Prediction",
            labels={'x': 'Features', 'y': 'Importance (Absolute Coefficient)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction vs Actual scatter plot
    st.subheader("📊 Prediction Accuracy Visualization")
    
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        title="Actual vs Predicted Prices",
        labels={'x': 'Actual Price (Lakhs)', 'y': 'Predicted Price (Lakhs)'}
    )
    
    # Add perfect prediction line
    min_price = min(float(y_test.min()), float(y_pred.min()))
    max_price = max(float(y_test.max()), float(y_pred.max()))
    fig.add_trace(go.Scatter(
        x=[min_price, max_price],
        y=[min_price, max_price],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Educational content
    st.subheader("📚 How the Model Works")
    
    with st.expander("🤖 Linear Regression Explained"):
        st.write("""
        **Linear Regression** is a fundamental machine learning algorithm that finds the best linear relationship 
        between input features and the target variable (price).
        
        **Formula:** `Price = β₀ + β₁×Location + β₂×SqFt + β₃×Bath + β₄×BHK + ε`
        
        Where:
        - β₀ is the intercept (base price)
        - β₁, β₂, β₃, β₄ are coefficients for each feature
        - ε is the error term
        """)
    
    with st.expander("📊 Model Evaluation Metrics"):
        st.write("""
        **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual prices.
        Lower is better.
        
        **R² Score:** Proportion of variance in the target variable that's predictable from features.
        Ranges from 0 to 1, where 1 is perfect prediction.
        
        **Model Accuracy:** Percentage accuracy calculated as (1 - MAE/Mean_Price) × 100.
        """)

def market_analysis(df):
    """Market analysis and visualization"""
    # Add market analysis background with financial patterns
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA4MCA4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBmaWxsPSJub25lIiBzdHJva2U9IiNmZmMxMDciIHN0cm9rZS13aWR0aD0iMSIgc3Ryb2tlLW9wYWNpdHk9IjAuMDMiPgogICAgPHBhdGggZD0iTTEwIDUwTDIwIDQwTDMwIDQ1TDQwIDMwTDUwIDM1TDYwIDI1TDcwIDMwIi8+CiAgICA8Y2lyY2xlIGN4PSIxMCIgY3k9IjUwIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIyMCIgY3k9IjQwIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIzMCIgY3k9IjQ1IiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSI0MCIgY3k9IjMwIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSI1MCIgY3k9IjM1IiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSI2MCIgY3k9IjI1IiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSI3MCIgY3k9IjMwIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4=');
                background-size: 80px 80px;
                background-repeat: repeat;
                z-index: -1000; pointer-events: none;">
    </div>
    <div style="position: fixed; bottom: 10%; left: 10%; font-size: 3rem; opacity: 0.05; z-index: -999;">
        📈💹📊💰🏙️
    </div>
    """, unsafe_allow_html=True)
    
    st.header("📈 Market Analysis")
    
    # Price distribution
    st.subheader("💰 Price Distribution")
    
    fig = px.histogram(
        df,
        x='price',
        title="Distribution of Property Prices",
        labels={'price': 'Price (Lakhs)', 'count': 'Number of Properties'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Location-wise analysis
    st.subheader("🏘️ Location-wise Analysis")
    
    location_stats = df.groupby('location').agg({
        'price': ['mean', 'count', 'std'],
        'total_sqft': 'mean'
    }).round(2)
    
    location_stats.columns = ['Avg Price (Lakhs)', 'Property Count', 'Price Std Dev', 'Avg SqFt']
    location_stats = location_stats.sort_values('Avg Price (Lakhs)', ascending=False)
    
    # Top 10 most expensive locations
    top_10 = location_stats.head(10)
    
    fig = px.bar(
        x=top_10.index,
        y=top_10['Avg Price (Lakhs)'],
        title="Top 10 Most Expensive Locations",
        labels={'x': 'Location', 'y': 'Average Price (Lakhs)'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # BHK vs Price analysis
    st.subheader("🏠 BHK vs Price Analysis")
    
    fig = px.box(
        df,
        x='bhk',
        y='price',
        title="Price Distribution by BHK",
        labels={'bhk': 'Number of BHK', 'price': 'Price (Lakhs)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlation")
    
    corr_matrix = df[['total_sqft', 'bath', 'bhk', 'price']].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Numerical Features"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Market insights
    st.subheader("💡 Market Insights")
    
    insights = [
        f"📊 Average property price in Bengaluru: ₹{df['price'].mean():.2f} lakhs",
        f"🏘️ Most expensive location: {location_stats.index[0]} (₹{location_stats.iloc[0]['Avg Price (Lakhs)']:.2f} lakhs)",
        f"💰 Most affordable location: {location_stats.index[-1]} (₹{location_stats.iloc[-1]['Avg Price (Lakhs)']:.2f} lakhs)",
        f"🏠 Most common BHK type: {df['bhk'].mode()[0]} BHK",
        f"📏 Average property size: {df['total_sqft'].mean():.0f} sq ft"
    ]
    
    for insight in insights:
        st.info(insight)

def learning_resources():
    """Educational resources for students"""
    # Add learning resources background with book/education patterns
    st.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgdmlld0JveD0iMCAwIDEwMCAxMDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGcgZmlsbD0iIzI4YTc0NSIgZmlsbC1vcGFjaXR5PSIwLjAyIj4KICAgIDxyZWN0IHg9IjIwIiB5PSIyMCIgd2lkdGg9IjE1IiBoZWlnaHQ9IjIwIiByeD0iMiIvPgogICAgPHJlY3QgeD0iNDAiIHk9IjE1IiB3aWR0aD0iMTUiIGhlaWdodD0iMjAiIHJ4PSIyIi8+CiAgICA8cmVjdCB4PSI2MCIgeT0iMjUiIHdpZHRoPSIxNSIgaGVpZ2h0PSIyMCIgcng9IjIiLz4KICAgIDxwYXRoIGQ9Ik0yMCA2MEwzNSA1NUw1MCA2NUw2NSA2MEw4MCA2NUw4MCA3NUw2NSA3MEw1MCA4MEwzNSA3NUwyMCA4MFoiLz4KICA8L2c+Cjwvc3ZnPg==');
                background-size: 100px 100px;
                background-repeat: repeat;
                z-index: -1000; pointer-events: none;">
    </div>
    <div style="position: fixed; top: 20%; right: 15%; font-size: 3rem; opacity: 0.05; z-index: -999;">
        📚🎓💡🔬📖
    </div>
    """, unsafe_allow_html=True)
    
    st.header("📚 Learning Resources")
    
    st.subheader("🎯 Key Concepts Covered")
    
    concepts = [
        {
            "title": "Linear Regression",
            "description": "A statistical method for modeling the relationship between a dependent variable and independent variables.",
            "applications": ["Price prediction", "Trend analysis", "Risk assessment"]
        },
        {
            "title": "Feature Engineering",
            "description": "The process of selecting and transforming variables for machine learning models.",
            "applications": ["Data preprocessing", "Model performance improvement", "Domain knowledge integration"]
        },
        {
            "title": "Model Evaluation",
            "description": "Techniques to assess model performance and reliability.",
            "applications": ["MAE, RMSE, R²", "Cross-validation", "Bias-variance tradeoff"]
        },
        {
            "title": "Data Visualization",
            "description": "Graphical representation of data to discover patterns and insights.",
            "applications": ["Exploratory data analysis", "Result presentation", "Decision making"]
        }
    ]
    
    for concept in concepts:
        with st.expander(f"📖 {concept['title']}"):
            st.write(f"**Definition:** {concept['description']}")
            st.write("**Applications:**")
            for app in concept['applications']:
                st.write(f"• {app}")
    
    st.subheader("🔬 Suggested Experiments")
    
    experiments = [
        "Try different combinations of features and observe how predictions change",
        "Compare prices across different locations for similar properties",
        "Analyze the relationship between property size and price",
        "Investigate how the number of bathrooms affects property value",
        "Study the correlation between different features"
    ]
    
    for i, experiment in enumerate(experiments, 1):
        st.write(f"{i}. {experiment}")
    
    st.subheader("📊 Data Science Workflow")
    
    workflow_steps = [
        "Data Collection & Understanding",
        "Data Cleaning & Preprocessing",
        "Exploratory Data Analysis",
        "Feature Engineering",
        "Model Selection & Training",
        "Model Evaluation & Validation",
        "Deployment & Monitoring"
    ]
    
    for i, step in enumerate(workflow_steps, 1):
        st.write(f"{i}. **{step}**")
    
    st.subheader("💡 Tips for Students")
    
    tips = [
        "Always start with data exploration before building models",
        "Understand your data - check for missing values, outliers, and patterns",
        "Feature engineering often has more impact than algorithm selection",
        "Cross-validate your models to ensure they generalize well",
        "Interpret your results in the context of the problem domain",
        "Document your work and assumptions for reproducibility"
    ]
    
    for tip in tips:
        st.success(tip)

if __name__ == "__main__":
    main()
