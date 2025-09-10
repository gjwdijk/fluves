import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import os
import itertools
import joblib

from sklearn.preprocessing import MinMaxScaler

FILE_PATH_DAY_1 = 'data_for_model/25_event_day.npy'
NORMAL_DAY_PATH = 'data_for_model/sound_levels_data.npy'



def get_min_max_test():

    X_test = np.load(FILE_PATH_DAY_1, allow_pickle=True)
    X_test = np.stack(X_test['data'], axis=0) 
    shape = X_test.shape

    X_train = np.load(NORMAL_DAY_PATH)
    position_indices = np.arange(689)
    values = 60 + 10 * position_indices  
    valid_positions = np.where(values >= 1260)[0]
    X_train = X_train[:, valid_positions, :, :]  

    scalers = joblib.load('scalers.pkl')

    for i, scaler in enumerate(scalers):
        test_flat_values = X_test[:, i, :, 0].reshape(-1, 1)
        train_flat_values = X_train[:, i, :, 0].reshape(-1, 1)

        if np.max(test_flat_values) > np.max(train_flat_values):
            print(f"Location {1260 + 10 * i}: normal max={np.max(train_flat_values)}; event max={np.max(test_flat_values)}")
            print(f"normal min={np.min(train_flat_values)}; event min={np.min(test_flat_values)}")


        # print(f"Normal Location {1260 + 10 * i}: min={np.min(train_flat_values)}; max={np.max(train_flat_values)}")
        # print(f"Event Location {1260 + 10 * i}: min={np.min(test_flat_values)}; max={np.max(test_flat_values)}")

def load_normalise_test_data(filepath):

    scalers = joblib.load('scalers.pkl')

    data = np.load(filepath, allow_pickle=True)
    X_data = np.stack(data['data'], axis=0) 
    X_file_names = np.stack(data['filename'], axis=0) 

    shape = X_data.shape

    num_positions = shape[1] # shape (1440, 569, 59, 1)

    for i in range(num_positions):
        flat_values = X_data[:, i, :, 0].reshape(-1, 1)
        scaled = scalers[i].transform(flat_values)
        X_data[:, i, :, 0] = scaled.reshape(shape[0], shape[2])

    X_test_normalised = np.transpose(X_data.squeeze(), (0, 2, 1))

    return X_test_normalised, X_file_names


def get_anomalies():

    X_test, X_file_names = load_normalise_test_data(FILE_PATH_DAY_1)

    print(np.max(X_test[:, :, 296].reshape(-1, 1)))


    # anomalies_idx = np.where(X_test > 1)

    # for sample_idx, timestep_idx, location_idx in zip(anomalies_idx[0], anomalies_idx[1], anomalies_idx[2]):
    #     location = 1260 + 10 * location_idx
    #     print(f"Sample {X_file_names[sample_idx]}, Timestep {timestep_idx}, Location {location}, Value: {X_test[sample_idx, timestep_idx, location_idx]}")

    

if __name__ == "__main__":
    get_anomalies()



