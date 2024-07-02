# SafeDriveLSTM: Vehicle Safety Time Series Prediction using LSTM

SafeDriveLSTM is a project focused on predicting vehicle safety incidents using Long Short-Term Memory (LSTM) neural networks. This project involves developing a Streamlit web application to facilitate the prediction process and visualize results interactively.

## Features
- Literature research on supervised learning and time series prediction.
- Data preprocessing and cleaning of large vehicle collision datasets.
- Feature extraction using DictVectorizer.
- Building and training LSTM models for predicting injuries and fatalities.
- Evaluating model performance and visualizing results.
- Deploying an interactive web application for feature selection, model training, and prediction visualization.

## Dataset
- **Source**: The dataset used in this project is sourced from data.gov.
- **Description**: The dataset contains large-scale vehicle collision data, including features like date, time, location, and the severity of the incident.

## Model
- **Architecture**: Long Short-Term Memory (LSTM) neural networks.
- **Frameworks**: Python, pandas, numpy, scikit-learn, keras, matplotlib, Streamlit.
- **Parameters**:
  - Input features: Various features related to vehicle collisions.
  - Sequence length: Configured based on the time series nature of the data.
  - LSTM layers: One or more LSTM layers with a configurable number of units.
  - Dropout: Dropout layers to prevent overfitting.
  - Optimizer: Adam optimizer.
  - Loss function: Mean Squared Error (MSE).

## Installation
1. Clone the repository: `git clone https://github.com/emankhadim/SaferDriveLSTM`, then navigate to the directory: `cd SafeDriveLSTM`.
2. Install the required packages with `pip install -r requirements.txt`.

## Usage
1. **Run the training script**: Preprocess the data, extract features, and train the LSTM model. Use the command `python train.py`.
2. **Run the testing script**: Evaluate the model performance on the test dataset. Use the command `python test.py`.
3. **Run the Streamlit app**: Launch the interactive interface to visualize the predictions. Use the command `streamlit run app.py`.

## Training
- The training script preprocesses the data, extracts features using DictVectorizer, and trains the LSTM model.
- The model is trained on sequences of data points, learning to predict the severity of vehicle collisions.
- Training involves optimizing the model parameters using the Adam optimizer and minimizing the Mean Squared Error (MSE) loss function.

## Testing
- The testing script evaluates the model's performance on a separate test dataset.
- Performance metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used to assess the model.

## Acknowledgements
Special thanks to the data.gov providers for the dataset and the open-source community for the tools and libraries used in this project.
