
# Tunnel Boring Machine Performance Predictor

Welcome to the Tunnel Boring Machine Performance Predictor! This repository, maintained by Kursat Kilic, contains a Streamlit web app designed to predict and analyze the performance of tunnel boring machines using various operational parameters.

## Dataset

The dataset used in this application provides insights into different aspects of tunnel boring machines and can be found here: [Tunnel Boring Machine Performance Analysis](https://www.sciencedirect.com/science/article/pii/S0886779807000508).

### Features Description

- **UCS (MPa)**: Uniaxial Compressive Strength, indicating the maximum axial compressive stress that a material can withstand.
- **BTS (MPa)**: Brazilian Tensile Strength, measuring the tensile strength of rocks and concrete.
- **PSI (kN/mm)**: Penetration Strength Index, representing the penetration capabilities of a tunnel boring machine.
- **DPW (m)**: Diameter of the Penetrating Wheel, which is crucial for the machine's ability to tunnel through various geologies.
- **Alpha Angle (degrees)**: The angle of attack of the boring disc, impacting the cutting efficiency and wear.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- Required Python libraries: `streamlit`, `pandas`, `plotly`, `matplotlib`, `shap`, `joblib`, `tensorflow`, `PIL`, `requests`

### Installing

1. **Clone the Repository**: 

   ```
   git clone https://github.com/kilickursat/WebApp.git
   cd WebApp
   ```

2. **Install Dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**:

   ```
   streamlit run app.py
   ```

   Access the application in your web browser at: `http://localhost:8501`.

## Using the Application

- **Sidebar for User Input**: Input parameters for the tunnel boring machine.
- **View Descriptive Statistics**: Observe descriptive statistics of the dataset on the main page.
- **Visualizations**: Interact with plots for data understanding.
- **Predict Performance**: Enter parameters and use the 'Predict and Analyze' button for predictions and analysis.

## Contributing

Interested in contributing? Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## Author

- **[Kursat Kilic](https://github.com/kilickursat)** - *Initial work*

See [contributors](https://github.com/kilickursat/WebApp/contributors) for more project participants.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- etc.
