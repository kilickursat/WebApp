import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import streamlit.components.v1 as components
import joblib
import requests
from tensorflow.keras.models import load_model
from io import BytesIO
import tempfile
from PIL import Image
import os

# Function to download a file from a URL and save it temporarily
def download_file(url, is_model=False, is_excel=False):
    response = requests.get(url)
    response.raise_for_status()
    if is_model:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    elif is_excel:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    else:
        return BytesIO(response.content)

# Function to load images
def load_images():
    image_scientist = 'https://github.com/kilickursat/WebApp/blob/main/Leonardo_Diffusion_XL_An_AI_scientist_with_his_cuttingedge_tec_1.jpg'
    image_tunnel = 'https://github.com/kilickursat/WebApp/blob/main/A__high_tech_tunnel_boring_machine_excavates_under_a_city_with_cross_sectional_view_GIF_format__Style-_Anime_seed-0ts-1705818285_idx-0.png'
    return image_scientist, image_tunnel

# Display images at the top of the main page
def display_images(image_scientist, image_tunnel):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_scientist, width=300, caption='AI Scientist')
    with col2:
        st.image(image_tunnel, width=300, caption='Tunnel Boring Machine')

# Function to scale input features
def scale_input(input_data, scaler, FEATURE_NAMES):
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    return scaler.transform(input_df)

# Main function
def main():
    image_scientist, image_tunnel = load_images()
    display_images(image_scientist, image_tunnel)

    # Download and load the trained model and scaler
    model_url = 'https://github.com/kilickursat/WebApp/raw/main/ann_model.h5'
    scaler_url = 'https://github.com/kilickursat/WebApp/raw/main/scaler.pkl'
    dataset_url = 'https://github.com/kilickursat/WebApp/raw/main/TBM_Performance.xlsx'

    model_path = download_file(model_url, is_model=True)
    model = load_model(model_path)
    scaler = joblib.load(download_file(scaler_url))
    dataset_path = download_file(dataset_url, is_excel=True)

    # Load the dataset and drop unnecessary columns
    df = pd.read_excel(dataset_path)
    df.drop(columns=['Type of rock and descriptions', 'Measured ROP (m/h)'], inplace=True)

    # Display descriptive statistics of the dataset
    st.write("Dataset Descriptive Statistics:")
    st.write(df.describe())

    # Define feature names and input data
    FEATURE_NAMES = ['UCS (MPa)', 'BTS (MPa)', 'PSI (kN/mm)', 'DPW (m)', 'Alpha angle (degrees)']
    input_data = {}
    for feature in FEATURE_NAMES:
        input_data[feature] = st.sidebar.number_input(feature, min_value=0.0, max_value=100.0, value=50.0)

    # Button to make predictions and display plots
    if st.sidebar.button('Predict and Analyze'):
        scaled_input = scale_input(input_data, scaler, FEATURE_NAMES)
        prediction = model.predict(scaled_input)

        st.subheader('Predicted Penetration Rate (ROP):')
        st.write(prediction[0][0])

        explainer = shap.KernelExplainer(model.predict, shap.sample(scaled_input, 100))
        shap_values = explainer.shap_values(scaled_input)
        shap.summary_plot(shap_values, scaled_input, feature_names=FEATURE_NAMES)

    # Line chart for UCS (MPa) over the tunnel stations
    if 'UCS (MPa)' in df.columns:
        st.sidebar.subheader("UCS Trend Over Tunnel Stations")
        fig_uc = px.line(df, x='UCS (MPa)', y='Tunnel stations (m)', title='UCS (MPa) over Tunnel Stations')
        st.sidebar.plotly_chart(fig_uc)

    # Clean up the temporary files
    os.remove(model_path)
    os.remove(dataset_path)

if __name__ == '__main__':
    main()
