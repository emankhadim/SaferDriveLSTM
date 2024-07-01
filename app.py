import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data
from feature_extraction import extract_features
from train_model import train_model

# Load and preprocess data
file_path = 'data/vehicle_crashes.csv'
data = preprocess_data(file_path)

# Feature extraction
data_features_df, feature_names = extract_features(data)

# Streamlit application
st.title('Vehicle Safety Time Series Prediction')
st.write('This application predicts various aspects of vehicle safety incidents.')

# Show data
if st.checkbox('Show raw data'):
    st.write(data.head())

# Feature selection
selected_features = st.multiselect('Select features to predict', feature_names, default=feature_names[:2])

# Prepare and train the model
if st.button('Train Model'):
    model, predictions, y_test = train_model(data_features_df, selected_features)
    st.success('Model trained successfully!')

    # Plot the predictions
    for i, feature in enumerate(selected_features):
        plt.figure(figsize=(14, 7))
        plt.plot(y_test[:, i], label='Actual')
        plt.plot(predictions[:, i], label='Predicted', color='red')
        plt.title(f'LSTM Model Predictions for {feature}')
        plt.xlabel('Time')
        plt.ylabel(feature)
        plt.legend()
        st.pyplot(plt)

# Streamlit command to run the app
# Command: `streamlit run app.py`
