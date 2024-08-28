import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Load model, scaler, and encoder
model_path = './model.pkl'
encoder_path = './encoder.pkl'
scaler_path = './scaler.pkl'
csv_path = './data_cleaned.csv'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    encoder = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Define the cities dictionary
cities = {
    'Cairo': ['Hay Sharq', 'Hay El Maadi', 'Mokattam', 'Mostakbal City', 'Future City', 'New Cairo City', 'New Capital City', 'New Heliopolis', 'Shorouk City', 'Madinaty', '6 October City', 'Sheikh Zayed City'],
    'Giza': ['6 October City', 'Sheikh Zayed City', 'Kafr El Sheikh', 'El Giza'],
    'Alexandria': ['Alexandria', 'Kafr El Sheikh'],
    'North Coast': ['Marina', 'Sidi Abdel Rahman', 'Al Alamein'],
    'Red Sea': ['Hurghada', 'El Gouna', 'Sahl Hasheesh'],
    'Suez': ['Suez', 'Ain Sokhna']
}

# Function to make predictions
def predict_price(area, bedroom_number, bathroom_number, property_type, governorate, city):
    # Prepare the data for prediction
    new_data = pd.DataFrame([[area, bedroom_number, bathroom_number, property_type, governorate, city]],
                            columns=['area', 'bedroom_number', 'bathroom_number', 'property_type', 'governorate', 'City'])

    new_data_encoded = encoder.transform(new_data[['property_type', 'governorate', 'City']])
    new_data_scaled = scaler.transform(new_data[['area', 'bedroom_number', 'bathroom_number']])
    combined_features = np.hstack((new_data_scaled, new_data_encoded.toarray()))

    # Predict the price
    predicted_price = model.predict(combined_features)
    return predicted_price[0]  # Return the predicted price

# Streamlit app layout
st.title('Property Price Prediction')

area = st.number_input('Area (in square meters):', min_value=20)
bedroom_number = st.number_input('Number of Bedrooms:', min_value=1, max_value=7)
bathroom_number = st.number_input('Number of Bathrooms:', min_value=1, max_value=7)
property_type = st.selectbox('Property Type:', ['Apartment', 'Chalet', 'Duplex', 'Penthouse', 'Townhouse', 'Twin House', 'Villa'])
governorate = st.selectbox('Governorate:', list(cities.keys()))

if governorate:
    city = st.selectbox('City:', cities[governorate])
else:
    city = st.selectbox('City:', [])

# Prediction and Visualization
if st.button('Predict'):
    predicted_price = predict_price(area, bedroom_number, bathroom_number, property_type, governorate, city)
    st.write(f'The predicted price is: ${predicted_price:.2f}')

    # Visualization
    st.header('Visualization')
    st.subheader('Property Price in Egypt vs. Predicted Price')

    data = pd.read_csv(csv_path)
    data['predicted_price'] = predicted_price[0]
    st.scatter_chart(data[['price_in_million', 'predicted_price']])

