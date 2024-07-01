import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length]
        sequences.append((seq, label))
    return sequences

def train_model(data_features_df, selected_features, seq_length=4):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_features_df[selected_features])

    # Create sequences
    sequences = create_sequences(scaled_data, seq_length)
    train_size = int(len(sequences) * 0.8)
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    X_train = np.array([seq[0] for seq in train_sequences])
    y_train = np.array([seq[1] for seq in train_sequences])
    X_test = np.array([seq[0] for seq in test_sequences])
    y_test = np.array([seq[1] for seq in test_sequences])

    # Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(len(selected_features)))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    return model, predictions, y_test

# Example usage:
# from data_preprocessing import preprocess_data
# from feature_extraction import extract_features
# data = preprocess_data('path_to_your_cleaned_vehicle_crash.csv')
# data_features_df, feature_names = extract_features(data)
# selected_features = feature_names[:2]  # Example feature selection
# model, predictions, y_test = train_model(data_features_df, selected_features)
