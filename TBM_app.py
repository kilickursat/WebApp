import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import joblib
import requests
from tensorflow.keras.models import load_model
from io import BytesIO

# Function to download a file from a URL
def download_file(url):
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)

# Function to load images (adjust the path to where you've saved the images)
def load_images():
    image_scientist = 'https://github.com/kilickursat/WebApp/blob/main/Leonardo_Diffusion_XL_An_AI_scientist_with_his_cuttingedge_tec_1.jpg'
    image_tunnel = 'https://github.com/kilickursat/WebApp/blob/main/A__high_tech_tunnel_boring_machine_excavates_under_a_city_with_cross_sectional_view_GIF_format__Style-_Anime_seed-0ts-1705818285_idx-0.png'
    return image_scientist, image_tunnel

# Display images at the top of the main page
def display_images(image_scientist, image_tunnel):
    col1, col2 = st.columns([1, 1])  # Updated from st.beta_columns to st.columns
    with col1:
        st.image(image_scientist, width=300, caption='AI Scientist')
    with col2:
        st.image(image_tunnel, width=300, caption='Tunnel Boring Machine')

# Main function
def main():
    # Display images
    image_scientist, image_tunnel = load_images()
    display_images(image_scientist, image_tunnel)

    # Download and load the trained model and scaler
    model_url = 'https://github.com/kilickursat/WebApp/raw/main/ann_model.h5'
    scaler_url = 'https://github.com/kilickursat/WebApp/raw/main/scaler.pkl'

    model = load_model(download_file(model_url))
    scaler = joblib.load(download_file(scaler_url))

    # Function to scale input features
    def scale_input(input_data, scaler):
        return scaler.transform(pd.DataFrame([input_data]))

    # Streamlit app layout
    st.title("TBM Penetration Rate Prediction")

    # User input fields
    st.sidebar.header("Input Features")
    # Example feature, replace with your actual features
    UCS = st.sidebar.number_input('UCS (MPa)', min_value=0.0, max_value=100.0, value=50.0)
    BTS = st.sidebar.number_input('BTS (MPa)', min_value=0.0, max_value=100.0, value=50.0)
    # Add more input fields as per your features

    # Collect inputs
    input_data = {
        'UCS': UCS,
        'BTS': BTS,
        # Add other features here
    }

    # Load your dataset (replace with your actual dataset)
    df = pd.read_excel('path_to_your_dataset/TBM_Performance.xlsx')

    # Split the main screen into left and right
    left_column, right_column = st.columns(2)

    # Button to make predictions and display plots
    if st.sidebar.button('Predict and Analyze'):
        # Scale inputs
        scaled_input = scale_input(input_data, scaler)

        # Generate predictions
        prediction = model.predict(scaled_input)

        # Display the prediction in the left column
        left_column.subheader('Predicted Penetration Rate (ROP):')
        left_column.write(prediction[0][0])

        # SHAP Feature Importance
        explainer = shap.KernelExplainer(model.predict, shap.sample(scaled_input, 100))
        shap_values = explainer.shap_values(scaled_input)
        left_column.subheader('SHAP Feature Importance:')
        left_column.pyplot(shap.force_plot(explainer.expected_value[0], shap_values[0], feature_names=list(input_data.keys())))

        # Actual vs Predicted Plot
        actual = df['Measured ROP (m/h)']
        predicted = model.predict(scaler.transform(df.drop(columns=['Measured ROP (m/h)', 'Type of rock and descriptions', 'Tunnel stations (m)'])))
        fig = px.scatter(x=actual, y=predicted.flatten(), labels={'x': 'Actual ROP', 'y': 'Predicted ROP'})
        left_column.subheader('Actual vs Predicted Plot:')
        left_column.plotly_chart(fig)

    # Right column: Show dataframe and example line chart
    right_column.subheader("Dataset Overview")
    right_column.dataframe(df.head())  # Show the first few rows of the dataset

    # Example line chart
    if 'some_column' in df.columns:
        right_column.subheader("Line Chart Example")
        right_column.line_chart(df['some_column'])

    # Make sure to replace placeholder paths and feature names with actual values

if __name__ == '__main__':
    main()
