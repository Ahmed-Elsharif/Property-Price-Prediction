import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Define paths to files
model_path = './model.pkl'
encoder_path = './encoder.pkl'
scaler_path = './scaler.pkl'
csv_path = './data_cleaned.csv'

# Load your model, scaler, and encoder
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(encoder_path, 'rb') as f:  
    encoder = pickle.load(f)

# Define the cities dictionary
cities = {
    'Cairo': ['Hay Sharq', 'Hay El Maadi', 'Mokattam', 'Mostakbal City', 'Future City', 'New Cairo City', 'New Capital City', 'New Heliopolis', 'Shorouk City', 'Madinaty', '6 October City', 'Sheikh Zayed City'],
    'Giza': ['6 October City', 'Sheikh Zayed City', 'Kafr El Sheikh', 'El Giza'],
    'Alexandria': ['Alexandria', 'Kafr El Sheikh'],
    'North Coast': ['Marina', 'Sidi Abdel Rahman', 'Al Alamein'],
    'Red Sea': ['Hurghada', 'El Gouna', 'Sahl Hasheesh'],
    'Suez': ['Suez', 'Ain Sokhna']
}

def predict_price(area, bedroom_number, bathroom_number, property_type, governorate, city):
    # Create a DataFrame for the input data
    new_data = pd.DataFrame([[area, bedroom_number, bathroom_number, property_type, governorate, city]],
                            columns=['area', 'bedroom_number', 'bathroom_number', 'property_type', 'governorate', 'City'])

    # Encode categorical features
    new_data_encoded = encoder.transform(new_data[['property_type', 'governorate', 'City']])

    # Scale numerical features
    new_data_scaled = scaler.transform(new_data[['area', 'bedroom_number', 'bathroom_number']])

    # Combine the scaled numerical data with the encoded categorical data
    combined_features = np.hstack((new_data_scaled, new_data_encoded.toarray()))

    # Create a DataFrame for the combined features 
    feature_names = (list(new_data[['area', 'bedroom_number', 'bathroom_number']].columns) +
                     list(encoder.get_feature_names_out(['property_type', 'governorate', 'City'])))

    new_data_prepared = pd.DataFrame(combined_features, columns=feature_names)

   
    prediction = model.predict(new_data_prepared)
    return prediction


# Streamlit app
st.title('Property Price Prediction')

st.header('Enter the details below:')

area = st.number_input('Area (in square meters):', min_value=10)
bedroom_number = st.number_input('Number of Bedrooms:', min_value=1, max_value=7)
bathroom_number = st.number_input('Number of Bathrooms:', min_value=1, max_value=7)

property_type = st.selectbox('Property Type:', ['Apartment', 'Chalet', 'Duplex', 'Penthouse', 'Townhouse', 'Twin House', 'Villa'])
governorate = st.selectbox('Governorate:', list(cities.keys()))

# Update city options based on selected governorate
if governorate:
    city = st.selectbox('City:', cities[governorate])
else:
    city = st.selectbox('City:', [])

if st.button('Predict'):
    predicted_price = predict_price(area, bedroom_number, bathroom_number, property_type, governorate, city)
    st.write(f'The predicted price is: ${predicted_price[0]:.2f}')

#  DataFrame
data = pd.read_csv(csv_path)
# Visualization 
st.header('Visualization')
st.subheader('Property Price in Egypt vs. predicted_price')
data['predicted_price'] = predicted_price[0]
st.scatter_chart(data[['price_in_million','predicted_price']])
