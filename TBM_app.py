# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:08:37 2021
@author: KURSAT
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import streamlit as st
from PIL import Image

# Create a title
st.write("""
# Prediction of TBM Penetration Rate 
  
Decide penetration rate based on rock properties using machine learning and python !!!
""")

# Web Application Logo
image = Image.open('Tunel-Acma-Makineleri.jpg')
st.image(image, width=500)

# Read the data
dt2 = pd.read_excel("TBM_Performance.xlsx", engine='openpyxl')

dt2["ROCK_PRO"] = dt2["UCS"] / dt2["BTS"]

X = dt2[["UCS", "Fs", "Orientation", "BTS", "PI", "ROCK_PRO"]]
y = dt2[["ROP"]]

# Set a subheader
st.subheader("Data Information")

# Show the data as a table
st.dataframe(dt2)

# Show statistics on the data
st.write(dt2.describe())

dt2["ROCK_PRO"] = dt2["UCS"] / dt2["BTS"]
X = pd.DataFrame(np.c_[dt2["ROCK_PRO"], dt2["PI"], dt2["Orientation"]], columns=["ROCK_PRO", "PI", "Orientation"])
y = dt2[['ROP']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')
st.sidebar.subheader('Please play with the sidebars to create new prediction')


def user_input_features():
    ROCK_PRO = st.sidebar.slider('ROCK_PRO', float(X.ROCK_PRO.min()), float(X.ROCK_PRO.max()), float(X.ROCK_PRO.mean()))
    PI = st.sidebar.slider('PI', float(X.PI.min()), float(X.PI.max()), float(X.PI.mean()))
    Orientation = st.sidebar.slider('Orientation', float(X.Orientation.min()), float(X.Orientation.max()),
                                    float(X.Orientation.mean()))

    data = {'ROCK_PRO': ROCK_PRO,
            'PI': PI,
            'Orientation': Orientation
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

SEED = 1
params = {'loss_function': 'MAPE',  # objective function
          'eval_metric': 'RMSE',  # metric
          'learning_rate': 0.02,
          'verbose': 50,
          'random_seed': SEED
          }

# Print specified input parameters
st.header('Specify Input Parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = CatBoostRegressor(**params).fit(X_train, y_train)

# Apply Model to Make Prediction
st.subheader("Model Test Accuracy Score:")
predictions = model.predict(df)
# predictions = model.predict(X_test)
# r2_score(predictions,y_test).round(2)

st.write(str(r2_score(y_test, model.predict(X_test)) * 100) + "%")

# PREDICTION
st.write("""
# PREDICTED PENETRATION RATE
""")

st.header('Prediction of Penetration Rate')
st.write(predictions)
st.write('---')

st.set_option('deprecation.showPyplotGlobalUse', False)

fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('CatBoost - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance')
st.pyplot()
