import streamlit as st
import pandas as pd
import pickle
import numpy as np

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

# Define the hierarchical dictionary
cities = {
    'Cairo': {
        'Hay El Maadi': {
            'Zahraa El Maadi': ['2nd Sector', '90 Avenue'],
            'South Investors Area': ['9th District', 'Aeon']
        },
        'New Cairo City': {
            'The 5th Settlement': ['Al Rabwa', 'Al Andalus Buildings'],
            'The 1st Settlement': ['Al Gouna', 'Al Hadaba Al Wosta']
        },
        'Mokattam': {
            'Al Hadaba Al Wosta': ['District 1', 'District 5'],
            'Faisal': ['El Motamayez District', 'El Narges Buildings']
        }
    },
    'Giza': {
        '6 October City': {
            '9th District': ['Al Wahat Road', 'Al Gouna'],
            '6 October Compounds': ['Cairo Alexandria Desert Road', 'Al Khamayel city']
        },
        'Sheikh Zayed City': {
            'Sheikh Zayed Compounds': ['Beverly Hills', 'Blanca Gardens'],
            'Green Belt': ['Aeon', 'Al Karma 4']
        }
    },
    'Alexandria': {
        'Alexandria': {
            'Alex West': ['Alex West Compound', 'Al Wahat Road'],
            'Al Andalus District': ['Zahra Al Maadi', 'Zahra Al Mansoura']
        }
    },
    'North Coast': {
        'Marina': {
            'Marina 1': ['Palm Hills', 'Fifth Square'],
            'Marina 5': ['Eastown', 'Cairo Festival City']
        },
        'Sidi Abdel Rahman': {
            'Hacienda Bay': ['Hacienda Bay', 'Hacienda White'],
            'Marassi': ['Marassi 1', 'Marassi 2']
        }
    },
    'Red Sea': {
        'Hurghada': {
            'Al Ahyaa District': ['El Gouna', 'El Kawther District'],
            'El Hadaba Al Wosta': ['El Motamayez District', 'El Narges Buildings']
        }
    },
    'Suez': {
        'Suez': {
            'Al Wahat Road': ['Palm Hills', 'Fifth Square'],
            'Downtown': ['Eastown', 'Cairo Festival City']
        },
        'Ain Sokhna': {
            'Azha North': ['Telal Al Sokhna', 'La Vista Gardens'],
            'Palm Hills': ['Al Wahat Road', 'Al Karma 4']
        }
    }
}

# Function to make predictions
def predict_price(area, bedroom_number, bathroom_number, property_type, governorate, city, location, address):
    # Prepare the data for prediction
    new_data = pd.DataFrame([[area, bedroom_number, bathroom_number, property_type, governorate, city, location, address]],
                            columns=['area', 'bedroom_number', 'bathroom_number', 'property_type', 'governorate', 'City', 'Location', 'Address'])

    new_data_encoded = encoder.transform(new_data[['property_type', 'governorate', 'City', 'Location', 'Address']])
    new_data_scaled = scaler.transform(new_data[['area', 'bedroom_number', 'bathroom_number']])
    combined_features = np.hstack((new_data_scaled, new_data_encoded.toarray()))

    # Predict the price
    predicted_price = model.predict(combined_features)
    return predicted_price[0]  # Return the predicted price

# Streamlit app layout
st.title('Property Price Prediction')

area = st.number_input('Area (in square meters):', min_value=30)
bedroom_number = st.number_input('Number of Bedrooms:', min_value=1, max_value=7)
bathroom_number = st.number_input('Number of Bathrooms:', min_value=1, max_value=7)
property_type = st.selectbox('Property Type:', ['Apartment', 'Chalet', 'Duplex', 'Penthouse', 'Townhouse', 'Twin House', 'Villa'])
governorate = st.selectbox('Governorate:', list(cities.keys()))

if governorate:
    city = st.selectbox('City:', list(cities[governorate].keys()))
    if city:
        location = st.selectbox('Location:', list(cities[governorate][city].keys()))
        if location:
            address = st.selectbox('Address:', cities[governorate][city][location])
else:
    city = st.selectbox('City:', [])
    location = st.selectbox('Location:', [])
    address = st.selectbox('Address:', [])

# Prediction and Visualization
if st.button('Predict'):
    predicted_price = predict_price(area, bedroom_number, bathroom_number, property_type, governorate, city, location, address)
    st.write(f'The predicted price in million is: EGP{predicted_price:.2f}')

    # Visualization
    st.header('Visualization')
    st.subheader('Property Price in million in Egypt vs. Predicted Price')

    data = pd.read_csv(csv_path)
    data['predicted_price'] = predicted_price
    st.scatter_chart(data[['price_in_million', 'predicted_price']])
