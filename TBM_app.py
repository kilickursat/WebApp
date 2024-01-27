import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import joblib
import requests
from tensorflow.keras.models import load_model
from PIL import Image
import os
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt

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

# Function to load local images
def load_images():
    image_scientist = Image.open('Leonardo_Diffusion_XL_An_AI_scientist_with_his_cuttingedge_tec_1.jpg')
    image_tunnel = Image.open('A__high_tech_tunnel_boring_machine_excavates_under_a_city_with_cross_sectional_view_GIF_format__Style-_Anime_seed-0ts-1705818285_idx-0.png')
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

    # Load the dataset
    df = pd.read_excel(dataset_path)

    # Calculate descriptive statistics
    descriptive_stats = df.describe()

    # Define feature names
    FEATURE_NAMES = ['UCS (MPa)', 'BTS (MPa)', 'PSI (kN/mm)', 'DPW (m)', 'Alpha angle (degrees)']

    # Initialize input_data dictionary
    input_data = {}

    # Set min and max values for each feature based on descriptive statistics
    for feature in FEATURE_NAMES:
        if feature in df.columns:
            min_value = descriptive_stats.at['min', feature]
            max_value = descriptive_stats.at['max', feature]
            default_value = (min_value + max_value) / 2
            input_data[feature] = st.sidebar.number_input(feature, min_value=min_value, max_value=max_value, value=default_value)
        else:
            st.sidebar.write(f"Feature {feature} not found in dataset.")
            input_data[feature] = 0.0

    # Button to make predictions and display plots
    if st.sidebar.button('Predict and Analyze'):
        scaled_input = scale_input(input_data, scaler, FEATURE_NAMES)
        prediction = model.predict(scaled_input)
        st.subheader('Predicted Penetration Rate (ROP):')
        st.write(prediction[0][0])

        # Calculate SHAP values
        explainer = shap.KernelExplainer(model.predict, shap.sample(scaled_input, 100))
        shap_values = explainer.shap_values(scaled_input)
        
        # Create a SHAP bar plot
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap.summary_plot(shap_values, FEATURE_NAMES, plot_type="bar")
        st.pyplot(plt)  # Display the plot in Streamlit

        # Add Actual vs Predicted Plot
        actual = df['Measured ROP (m/h)']
        predicted = model.predict(scaler.transform(df[FEATURE_NAMES]))
        fig_act_vs_pred = px.scatter(x=actual, y=predicted.flatten(), labels={'x': 'Actual ROP', 'y': 'Predicted ROP'})
        st.plotly_chart(fig_act_vs_pred)

    # Clean up the temporary files
    os.remove(model_path)
    os.remove(dataset_path)

if __name__ == '__main__':
    main()
