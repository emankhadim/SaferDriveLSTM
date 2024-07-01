import pandas as pd

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Data cleaning steps
    data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'], errors='coerce')
    data['CRASH TIME'] = pd.to_datetime(data['CRASH TIME'], format='%H:%M', errors='coerce').dt.time
    data['DATETIME'] = pd.to_datetime(data['CRASH DATE'].astype(str) + ' ' + data['CRASH TIME'].astype(str), errors='coerce')
    data.set_index('DATETIME', inplace=True)
    data.drop(columns=['CRASH DATE', 'CRASH TIME'], inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    return data

