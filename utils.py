"""
Utility functions for the Bengaluru House Price Prediction App
Educational tool for 3rd year Computer Science students
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def load_or_train_model():
    """Load existing model or train a new one if not found"""
    model_path = 'house_price_model.pkl'
    
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            print("Model loaded successfully from existing file")
            return model_data
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Training new model...")
    
    # Train new model
    return train_new_model()

def train_new_model():
    """Train a new machine learning model"""
    try:
        # Load data
        df = pd.read_csv('Bengaluru_House_Data.csv')
        
        # Clean data
        df = df.dropna()
        
        # Convert total_sqft to numeric
        def convert_sqft_to_num(x):
            if isinstance(x, str):
                if '-' in x:
                    try:
                        parts = x.split('-')
                        return (float(parts[0]) + float(parts[1])) / 2
                    except:
                        return None
                else:
                    try:
                        return float(x)
                    except:
                        return None
            return float(x)
        
        df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
        df = df.dropna()
        
        # Remove outliers
        df = df[
            (df['total_sqft'] >= 300) & 
            (df['total_sqft'] <= 10000) &
            (df['bath'] >= 1) & 
            (df['bath'] <= 10) &
            (df['bhk'] >= 1) & 
            (df['bhk'] <= 10) &
            (df['price'] > 0) &
            (df['price'] <= 1000)
        ]
        
        # Prepare features
        X = df[['location', 'total_sqft', 'bath', 'bhk']].copy()
        y = df['price'].copy()
        
        # Encode location
        location_encoder = LabelEncoder()
        X['location_encoded'] = location_encoder.fit_transform(X['location'])
        
        # Feature matrix
        X_final = X[['location_encoded', 'total_sqft', 'bath', 'bhk']]
        
        # Train model
        model = LinearRegression()
        model.fit(X_final, y)
        
        # Get locations
        locations = df['location'].unique()
        locations.sort()
        
        # Save model
        model_data = {
            'model': model,
            'location_encoder': location_encoder,
            'locations': locations.tolist()
        }
        
        joblib.dump(model_data, 'house_price_model.pkl')
        print("New model trained and saved successfully")
        
        return model_data
        
    except Exception as e:
        print(f"Error training new model: {e}")
        return None

def calculate_affordability_index(price_lakhs):
    """Calculate affordability index based on typical income levels"""
    # Assuming average household income in Bengaluru is around 8-12 lakhs per annum
    # Affordable housing is typically 3-5 times annual income
    
    avg_income = 10  # 10 lakhs per annum
    affordable_ratio = 4  # 4x annual income
    
    affordable_price = avg_income * affordable_ratio
    
    if price_lakhs <= affordable_price * 0.5:
        return 10  # Very affordable
    elif price_lakhs <= affordable_price * 0.75:
        return 8   # Affordable
    elif price_lakhs <= affordable_price:
        return 6   # Moderately affordable
    elif price_lakhs <= affordable_price * 1.5:
        return 4   # Expensive
    else:
        return 2   # Very expensive

def generate_market_insights(location, sqft, bath, bhk, predicted_price):
    """Generate market insights for the prediction"""
    insights = []
    
    # Size-based insights
    if sqft < 800:
        insights.append("💡 This is a compact property, ideal for first-time buyers or young professionals.")
    elif sqft > 2000:
        insights.append("💡 This is a spacious property, suitable for large families or luxury living.")
    
    # BHK-based insights
    if bhk == 1:
        insights.append("🏠 1 BHK properties are popular among young professionals and students.")
    elif bhk >= 4:
        insights.append("🏠 4+ BHK properties are considered luxury and suitable for large families.")
    
    # Price-based insights
    if predicted_price > 200:
        insights.append("💰 This property falls in the premium segment of the market.")
    elif predicted_price < 80:
        insights.append("💰 This property offers good value for money in the current market.")
    
    # Location-based insights
    premium_locations = ['Koramangala', 'Indiranagar', 'Whitefield', 'HSR Layout']
    if location in premium_locations:
        insights.append(f"📍 {location} is considered a premium location with good connectivity and amenities.")
    
    return insights

def get_location_statistics(df, location):
    """Get statistics for a specific location"""
    location_data = df[df['location'] == location]
    
    if location_data.empty:
        return None
    
    stats = {
        'count': len(location_data),
        'avg_price': location_data['price'].mean(),
        'min_price': location_data['price'].min(),
        'max_price': location_data['price'].max(),
        'avg_sqft': location_data['total_sqft'].mean(),
        'popular_bhk': location_data['bhk'].mode()[0] if not location_data['bhk'].mode().empty else 2
    }
    
    return stats

def validate_inputs(location, sqft, bath, bhk, available_locations):
    """Validate user inputs"""
    errors = []
    
    if location not in available_locations:
        errors.append("Please select a valid location from the available options.")
    
    if sqft < 300 or sqft > 10000:
        errors.append("Square feet should be between 300 and 10,000.")
    
    if bath < 1 or bath > 10:
        errors.append("Number of bathrooms should be between 1 and 10.")
    
    if bhk < 1 or bhk > 10:
        errors.append("Number of BHK should be between 1 and 10.")
    
    return errors

def calculate_price_per_sqft_market_position(price_per_sqft, location, df):
    """Calculate market position based on price per square foot"""
    location_data = df[df['location'] == location]
    
    if location_data.empty:
        return "No market data available for comparison"
    
    # Calculate price per sqft for location data
    location_data = location_data.copy()
    location_data['price_per_sqft'] = (location_data['price'] * 100000) / location_data['total_sqft']
    
    avg_price_per_sqft = location_data['price_per_sqft'].mean()
    
    if price_per_sqft < avg_price_per_sqft * 0.8:
        return "Below market average - Good value"
    elif price_per_sqft < avg_price_per_sqft * 1.2:
        return "Around market average"
    else:
        return "Above market average - Premium pricing"

def get_similar_properties(df, location, sqft, bhk, tolerance=0.2):
    """Find similar properties in the dataset"""
    sqft_min = sqft * (1 - tolerance)
    sqft_max = sqft * (1 + tolerance)
    
    similar = df[
        (df['location'] == location) &
        (df['total_sqft'] >= sqft_min) &
        (df['total_sqft'] <= sqft_max) &
        (df['bhk'] == bhk)
    ]
    
    return similar.head(5)  # Return top 5 similar properties

def format_price_display(price):
    """Format price for display"""
    if price >= 100:
        return f"₹{price:.1f} Cr"
    else:
        return f"₹{price:.2f} L"

def get_investment_insights(predicted_price, location, bhk):
    """Generate investment insights"""
    insights = []
    
    # ROI potential based on location and property type
    high_growth_locations = ['Whitefield', 'Electronic City', 'Sarjapur Road', 'Marathahalli']
    stable_locations = ['Koramangala', 'Indiranagar', 'Jayanagar', 'Malleshwaram']
    
    if location in high_growth_locations:
        insights.append("📈 This location has shown consistent growth and good rental potential.")
    elif location in stable_locations:
        insights.append("🏛️ This is a well-established location with stable property values.")
    
    # BHK-based insights
    if bhk == 2:
        insights.append("🎯 2 BHK properties have good resale value and rental demand.")
    elif bhk == 3:
        insights.append("🎯 3 BHK properties are popular among families and have good appreciation potential.")
    
    return insights
