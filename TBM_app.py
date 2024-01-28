import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import shap
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import os
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import numpy as np

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

# Function to create the sidebar
def create_sidebar(FEATURE_NAMES, df):
    st.sidebar.header("User Input Features")
    input_data = {}
    for feature in FEATURE_NAMES:
        min_value = float(df[feature].min())
        max_value = float(df[feature].max())
        default_value = (min_value + max_value) / 2.0
        input_data[feature] = st.sidebar.number_input(f"{feature} (Min: {min_value}, Max: {max_value})", min_value=min_value, max_value=max_value, value=default_value)
    return input_data

# Function to display the header
def display_header():
    st.title("Tunnel Boring Machine Performance Predictor")
    st.markdown("## Descriptive Analysis, Predictions & Feature Trends")

# Function to display dataset statistics
def display_dataset_statistics(df):
    st.write("Dataset Descriptive Statistics:")
    st.dataframe(df.describe())

# Function to plot feature importance
def plot_feature_importance(model, scaler, df, FEATURE_NAMES):
    explainer = shap.Explainer(model, scaler.transform(df[FEATURE_NAMES]))
    shap_values = explainer(scaler.transform(df[FEATURE_NAMES]))
    st.subheader('Feature Importance:')
    shap.summary_plot(shap_values.values, df[FEATURE_NAMES], plot_type="bar", show=False)
    st.pyplot(plt.gcf())

# Function to plot actual vs predicted
def plot_actual_vs_predicted(model, scaler, df, FEATURE_NAMES):
    actual = df['Measured ROP (m/h)']
    predicted = model.predict(scaler.transform(df[FEATURE_NAMES])).flatten()
    fig_act_vs_pred = px.scatter(x=actual, y=predicted, labels={'x': 'Actual ROP', 'y': 'Predicted ROP'})
    best_fit = np.polyfit(actual, predicted, 1)
    best_fit_line = go.Scatter(x=actual, y=np.polyval(best_fit, actual), mode='lines', name='Best Fit Line')
    fig_act_vs_pred.add_trace(best_fit_line)
    st.plotly_chart(fig_act_vs_pred)

# Main function
def main():
    st.set_page_config(layout="wide")  # Set to wide layout
    display_header()
    image_scientist, image_tunnel = load_images()
    display_images(image_scientist, image_tunnel)

    with st.spinner('Loading Model and Data...'):
        model_url = 'https://github.com/kilickursat/WebApp/raw/main/ann_model.h5'
        scaler_url = 'https://github.com/kilickursat/WebApp/raw/main/scaler.pkl'
        dataset_url = 'https://github.com/kilickursat/WebApp/raw/main/TBM_Performance.xlsx'

        model_path = download_file(model_url, is_model=True)
        model = load_model(model_path)
        scaler = joblib.load(download_file(scaler_url))
        dataset_path = download_file(dataset_url, is_excel=True)

        df = pd.read_excel(dataset_path)

    FEATURE_NAMES = ['UCS (MPa)', 'BTS (MPa)', 'PSI (kN/mm)', 'DPW (m)', 'Alpha angle (degrees)']
    input_data = create_sidebar(FEATURE_NAMES, df)

    # Layout for descriptive statistics
    col1, col2 = st.columns(2)
    with col1:
        display_dataset_statistics(df)

    # Layout for feature trends
    col3, col4 = st.columns(2)
    with col3:
        for feature in FEATURE_NAMES[:len(FEATURE_NAMES)//2]:  # First half of features
            if feature in df.columns:
                fig = px.line(df, x='Tunnel stations (m)', y=feature, title=f'{feature} over Tunnel Stations')
                st.plotly_chart(fig)

    with col4:
        for feature in FEATURE_NAMES[len(FEATURE_NAMES)//2:]:  # Second half of features
            if feature in df.columns:
                fig = px.line(df, x='Tunnel stations (m)', y=feature, title=f'{feature} over Tunnel Stations')
                st.plotly_chart(fig)

    if st.sidebar.button('Predict and Analyze'):
        with st.spinner('Calculating Predictions...'):
            scaled_input = scale_input(input_data, scaler, FEATURE_NAMES)
            prediction = model.predict(scaled_input)
            st.subheader('Predicted Penetration Rate (ROP):')
            st.write(prediction[0][0])

            # Layout for SHAP and Actual vs Predicted plots
            col5, col6 = st.columns(2)
            with col5:
                plot_feature_importance(model, scaler, df, FEATURE_NAMES)
            with col6:
                plot_actual_vs_predicted(model, scaler, df, FEATURE_NAMES)

    # Add dataset link
    st.markdown("### Dataset Reference:")
    st.markdown("[Tunnel Boring Machine Performance Analysis](https://www.sciencedirect.com/science/article/pii/S0886779807000508)")

    os.remove(model_path)
    os.remove(dataset_path)

if __name__ == '__main__':
    main()
