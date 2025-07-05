#!/usr/bin/env python3
"""
Machine Learning Model Training Script for Bengaluru House Price Prediction
Educational tool for 3rd year Computer Science students
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path='Bengaluru_House_Data.csv'):
    """Load and clean the dataset"""
    print("Loading and preparing data...")
    
    # Load the data
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df = df.dropna()
    
    # Convert total_sqft to numeric (handle range values like "1200-1300")
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
    
    # Remove outliers and invalid data
    df = df[
        (df['total_sqft'] >= 300) & 
        (df['total_sqft'] <= 10000) &
        (df['bath'] >= 1) & 
        (df['bath'] <= 10) &
        (df['bhk'] >= 1) & 
        (df['bhk'] <= 10) &
        (df['price'] > 0) &
        (df['price'] <= 1000)  # Price in lakhs
    ]
    
    print(f"After cleaning: {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

def train_model(df):
    """Train the machine learning model"""
    print("\nTraining the model...")
    
    # Select features and target
    features = ['location', 'total_sqft', 'bath', 'bhk']
    X = df[features].copy()
    y = df['price'].copy()
    
    # Encode categorical variables (location)
    location_encoder = LabelEncoder()
    X['location_encoded'] = location_encoder.fit_transform(X['location'])
    
    # Prepare feature matrix
    X_final = X[['location_encoded', 'total_sqft', 'bath', 'bhk']]
    
    print(f"Feature matrix shape: {X_final.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: {mae:.2f} lakhs")
    print(f"Root Mean Squared Error: {rmse:.2f} lakhs")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance (coefficients for linear regression)
    feature_names = ['Location', 'Total SqFt', 'Bathrooms', 'BHK']
    coefficients = model.coef_
    
    print(f"\nFeature Coefficients:")
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.4f}")
    
    return model, location_encoder, X_test, y_test, y_pred

def save_model(model, location_encoder, locations):
    """Save the trained model and metadata"""
    print("\nSaving the model...")
    
    model_data = {
        'model': model,
        'location_encoder': location_encoder,
        'locations': locations.tolist()
    }
    
    # Save the model
    joblib.dump(model_data, 'house_price_model.pkl')
    print("Model saved as 'house_price_model.pkl'")
    
    # Save location list
    with open('locations.txt', 'w') as f:
        for loc in locations:
            f.write(f"{loc}\n")
    print("Location list saved as 'locations.txt'")

def test_model(model, location_encoder, locations):
    """Test the model with sample predictions"""
    print("\nTesting the model...")
    
    # Test cases
    test_cases = [
        {'location': 'Whitefield', 'total_sqft': 1200, 'bath': 2, 'bhk': 3},
        {'location': 'Indiranagar', 'total_sqft': 1500, 'bath': 3, 'bhk': 3},
        {'location': 'Electronic City', 'total_sqft': 1000, 'bath': 2, 'bhk': 2},
        {'location': 'Koramangala', 'total_sqft': 1800, 'bath': 3, 'bhk': 4},
        {'location': 'HSR Layout', 'total_sqft': 900, 'bath': 2, 'bhk': 2}
    ]
    
    print("\nSample Predictions:")
    print("-" * 60)
    
    for test_case in test_cases:
        try:
            # Encode location
            location_encoded = location_encoder.transform([test_case['location']])[0]
            
            # Create feature array
            features = np.array([[
                location_encoded,
                test_case['total_sqft'],
                test_case['bath'],
                test_case['bhk']
            ]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            # Ensure positive prediction
            if prediction < 0:
                prediction = abs(prediction)
            
            price_per_sqft = (prediction * 100000) / test_case['total_sqft']
            
            print(f"Location: {test_case['location']}")
            print(f"Size: {test_case['total_sqft']} sqft, {test_case['bath']} bath, {test_case['bhk']} BHK")
            print(f"Predicted Price: ₹{prediction:.2f} lakhs")
            print(f"Price per SqFt: ₹{price_per_sqft:.0f}")
            print("-" * 60)
            
        except Exception as e:
            print(f"Error predicting for {test_case['location']}: {e}")

def main():
    """Main function to orchestrate the training process"""
    print("=== Bengaluru House Price Prediction Model Training ===")
    print("Educational ML tool for 3rd year Computer Science students")
    print("=" * 60)
    
    try:
        # Load and clean data
        df = load_and_clean_data()
        
        # Train model
        model, location_encoder, X_test, y_test, y_pred = train_model(df)
        
        # Get unique locations
        locations = df['location'].unique()
        locations.sort()
        
        # Save model
        save_model(model, location_encoder, locations)
        
        # Test model
        test_model(model, location_encoder, locations)
        
        print("\n" + "=" * 60)
        print("Model training completed successfully!")
        print("Files created:")
        print("- house_price_model.pkl (trained model)")
        print("- locations.txt (available locations)")
        print("\nYou can now run the Streamlit app: streamlit run app.py")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nTraining failed. Please check the error messages above.")
