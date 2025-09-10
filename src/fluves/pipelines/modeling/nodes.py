"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.12
"""

from typing import Tuple, Dict
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Attention, Conv1D, Conv1DTranspose
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.losses import Huber

import matplotlib.pyplot as plt

def slice_and_dice(xr_data: xr.Dataset, poi:Dict):
    print(f'slice_and_dice {xr_data}')
    poi = np.arange(poi['start'], poi['end'], poi['increment'])
    ds_raw = xr_data.sel(position=poi)
    l = int(len(ds_raw['time']) / 59)
    x = np.stack([ds_raw.sel(time=ds_raw['time'][0+i:59+i]).Pxx_dB.values for i in range(l)])
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    X_train = x
    print(X_train.shape)
    pos = 430
    x1 = ds_raw.sel(time=ds_raw['time'], position = pos*10)
    print(x1)
    df = pd.DataFrame({'timestamp': np.stack(ds_raw['time']), 'value': np.stack(x1.Pxx_dB.values)})
    print(df.head())
    return X_train

def scale(data: np.ndarray):
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape((data.shape[0], data.shape[2], data.shape[1]))
    print(f'scale: {X_train_normalized.shape}')
    return X_train_normalized

def reshape_and_normalise(data: np.ndarray):
    position_indices = np.arange(689)
    values = 60 + 10 * position_indices  

    valid_positions = np.where(values >= 1260)[0]
    reshaped_data = data[:, valid_positions, :, :]

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(reshaped_data.reshape(-1, reshaped_data.shape[-1])).reshape((reshaped_data.shape[0], reshaped_data.shape[2], reshaped_data.shape[1]))

    return X_normalized

def build_old_model() -> keras.Model:

    input_shape = (59, 569)  # (time_steps, features)

    # Encoder
    input_seq = Input(shape=input_shape)
    encoded = LSTM(128, activation='relu', return_sequences=True)(input_seq) 
    encoded = LSTM(64, activation='relu', return_sequences=False)(encoded)  
    encoded = Dropout(0.2)(encoded)

    # Latent space
    latent_space = Dense(128, activation='relu')(encoded)

    # Decoder
    decoded = RepeatVector(input_shape[0])(latent_space)  
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(128, activation='relu', return_sequences=True)(decoded)
    decoded = Dropout(0.2)(decoded)

    decoded = TimeDistributed(Dense(input_shape[1]))(decoded)

    model = Model(input_seq, decoded)

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

    # GJ - Altijd eerst een keer een fit doen alvorens het model te saven
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # history = model.fit(X_train_normalized, X_train_normalized, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

    return model

def train_and_validation_loss(model: keras.Model, X_train_normalized: np.ndarray) -> plt.figure:
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_normalized, X_train_normalized, epochs=200, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    return model, plt

def predict(model: keras.Model, X_normalized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reconstructed_data = model.predict(X_normalized)
    reconstruction_error = np.abs(X_normalized - reconstructed_data)
    return reconstructed_data, reconstruction_error

def thresholds(reconstruction_error: np.ndarray) -> np.ndarray:
    reconstruction_error_99p = np.percentile(reconstruction_error, q=99, axis=1)
    return np.percentile(reconstruction_error_99p, q=99, axis=0)

def get_anomalies(reconstruction_error: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # print(thresholds.shape)
    reconstruction_error_99p = np.percentile(reconstruction_error, q=99, axis=1)
    anomalies = reconstruction_error_99p[:, :] > thresholds
    # [print(f'get_anomalies: {anomalies[i]}') for i in range(5)]
    return anomalies

def plot_original_reconstructed(sample_idx, location, original_series: np.ndarray, reconstructed_series: np.ndarray):

    feature_idx = int((location - 1260)/10)

    original_series = original_series[sample_idx, :, feature_idx]
    reconstructed_series = reconstructed_series[sample_idx, :, feature_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(original_series, label="Original", linestyle="--", marker="o", color="royalblue")
    plt.plot(reconstructed_series, label="Reconstructed", linestyle="-", marker="s", color="darkorange")

    plt.xlabel("Time Step")
    plt.ylabel("Feature Value")
    plt.title(f"Original vs. Reconstructed Series (Location {location})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    return plt

def slice_and_dice2(xr_data: xr.Dataset, poi:Dict):
    print(f'slice_and_dice {xr_data}')
    # override pos ...
    # 26: pos = 510
    poi = np.arange(poi['start'], poi['end'], poi['increment'])
    df = pd.DataFrame([])
    for pos in poi:
        x1 = xr_data.sel(time=xr_data['time'], position = pos)
        print(x1)
        # df = pd.DataFrame({'timestamp': np.stack(x1['time']), 'position': pos, 'value': np.stack(x1.Pxx_dB.values)})
        dfp = pd.DataFrame({'timestamp': np.stack(x1['time']), 'position': [pos for i in range(len(x1['time']))], 'value': np.stack(x1.Pxx_dB.values)})
        df = pd.concat([df, dfp])
    print(df)
    return df

def slice_and_dice3(xr_data: xr.Dataset, poi:Dict):
    poi = np.arange(poi['start'], poi['end'], poi['increment'])
    xr_data = xr_data.sel(position=poi)
    print(f'slice_and_dice3 {xr_data}')
    df = pd.DataFrame({'timestamp': np.stack(xr_data['time'])})
    for pos in xr_data['position'].values:
        print(f'pos:{pos}')
        df[f'pos_{int(pos)}'] = pd.Series(np.stack(xr_data.sel(position=pos).Pxx_dB.values)).values
    print(df)
    # df.set_index('timestamp', inplace=True)
    return df

from datetime import datetime
from matplotlib import dates as md

def plot_dates_values(data):
    dates = data["timestamp"].to_list()
    values = data["value"].to_list()

    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(dates, values)
    plt.show()
    return plt

def plot_dates_values3(data):
    data2 = normalize3(data)
    # print(data.columns.to_list())
    print(data2.head())
    dates = data["timestamp"].to_list()

    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(xfmt)
    for col in data2.columns[2:10]:
        print(f'add column {col}')
        ax.plot(dates, data2[col], label=f"Position {col}")
    ax.legend()
    plt.show()
    return plt

def normalize3(df):
    # Get the `value` column from the training dataframe.
    dfr = df.copy()
    for col in df.columns[2:]:
        training_value = df[col].to_list()

        # Normalize `value` and save the mean and std we get,
        # for normalizing test data.
        training_value, training_mean, training_std = normalize(training_value)
        print(len(training_value))
        dfr[col] = training_value
    return dfr


def get_value_from_df(df):
    return df.value.to_list()


def normalize(values):
    print(f'normalize: {type(values)}')
    mean = np.mean(values)
    print(f'normalize: {type(mean)}')
    values -= mean
    std = np.std(values)
    values /= std
    return values, mean, std

def prepare(df):
    # Get the `value` column from the training dataframe.
    training_value = get_value_from_df(df)

    # Normalize `value` and save the mean and std we get,
    # for normalizing test data.
    training_value, training_mean, training_std = normalize(training_value)
    print(len(training_value))
    return training_value, training_mean, training_std, {'training_mean': training_mean, 'training_std': training_std}


TIME_STEPS = 32

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    # Convert 2D sequences into 3D as we will be feeding this into
    # a convolutional layer.
    res = np.expand_dims(output, axis=2)
    print(res.shape)
    return res

def create_sequences3(df, time_steps=TIME_STEPS):
    output = []
    for col in df.columns[2:10]:
        print(f'create_sequences3 for {col}')
        values = df[col].to_list()
        for i in range(len(values) - time_steps):
            output.append(values[i : (i + time_steps)])
    # Convert 2D sequences into 3D as we will be feeding this into
    # a convolutional layer.
    res = np.expand_dims(output, axis=2)
    print(res.shape)
    return res

def build_model2(x_train) -> keras.Model:
    n_steps = x_train.shape[1]
    n_features = x_train.shape[2]
    keras.backend.clear_session()
    model = keras.Sequential(
        [
            Input(shape=(x_train.shape[1], x_train.shape[2])),
            Conv1D(
                filters=32,
                kernel_size=7,
                padding="same",
                strides=2,
                activation="relu",
            ),
            Dropout(rate=0.2),
            Conv1D(
                filters=16,
                kernel_size=7,
                padding="same",
                strides=2,
                activation="relu",
            ),
            Conv1DTranspose(
                filters=16,
                kernel_size=7,
                padding="same",
                strides=2,
                activation="relu",
            ),
            Dropout(rate=0.2),
            Conv1DTranspose(
                filters=32,
                kernel_size=7,
                padding="same",
                strides=2,
                activation="relu",
            ),
            Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    history = model.fit(
        x_train,
        x_train,
        epochs=200,
        batch_size=60,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min", restore_best_weights=True)
        ],
    )       

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    return model

def thresholds2(model, x_train):
    # Get train MAE loss.
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    plt.hist(train_mae_loss, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)

    # Checking how the first sequence is learnt
    #plt.plot(x_train[0])
    #plt.show()
    #plt.plot(x_train_pred[0])
    #plt.show()
    return threshold


def normalize_test(values, mean, std):
    print(f'normalize: {type(values)}')
    print(f'normalize: {type(mean)}')
    #values1 = [values[i] - mean for i in range(len(values))]
    #values2 = [values1[i] / std for i in range(len(values1))]
    values -= mean
    values /= std
    return values


def anomalies(df_test, model, training_parameters: Dict, threshold):
    training_mean = np.float64(training_parameters['training_mean'])
    training_std = np.float64(training_parameters['training_std'])
    #test_value = get_value_from_df(df_test)
    #print(test_value)
    #print(df_test)
    npos = len(list(set(df_test['position'].tolist())))
    fig, axs = plt.subplots(npos)
    fig.suptitle('Soundlevel data per position')
    n = 0
    test_value = {}
    for pos, dfp in df_test.groupby('position'):
        print(f'anomalies position {pos}')
        test_value[pos] = normalize_test(get_value_from_df(dfp), training_mean, training_std)
        dates = dfp["timestamp"].to_list()
        xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
        # axs[n].xticks(rotation=25)
        axs[n].plot(dates, test_value[pos], label=f'pos {pos}')
        axs[n].legend()
        n += 1
    plt.show()

    fig, axs = plt.subplots(npos)
    fig.suptitle('Anomalies per position')
    n = 0
    anomalies = {}
    for pos, dfp in df_test.groupby('position'):
        # Create sequences from test values.
        x_test = create_sequences(test_value[pos])
        print("Test input shape: ", x_test.shape)

        # Get test MAE loss.
        x_test_pred = model.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))

        #plt.hist(test_mae_loss, bins=50)
        #plt.xlabel("test MAE loss")
        #plt.ylabel("No of samples")
        #plt.show()

        # Detect all the samples which are anomalies.
        anomalies[pos] = (test_mae_loss > threshold).tolist()
        print("Number of anomaly samples: ", np.sum(anomalies[pos]))
        print("Indices of anomaly samples: ", np.where(anomalies[pos]))

        # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
        anomalous_data_indices = []
        for data_idx in range(TIME_STEPS - 1, len(test_value[pos]) - TIME_STEPS + 1):
            time_series = range(data_idx - TIME_STEPS + 1, data_idx)
            if all([anomalies[pos][j] for j in time_series]):
                anomalous_data_indices.append(data_idx)
        print(anomalous_data_indices)

        df_subset = dfp.iloc[anomalous_data_indices, :]
        print(df_subset)
        plt.subplots_adjust(bottom=0.2)
        plt.xticks(rotation=25)
        ax = plt.gca()
        xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
        ax.xaxis.set_major_formatter(xfmt)

        dates = dfp["timestamp"].to_list()
        values = dfp["value"].to_list()
        axs[n].plot(dates, values, label=f"test data for postion {pos}")

        dates = df_subset["timestamp"].to_list()
        # dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
        values = df_subset["value"].to_list()
        axs[n].plot(dates, values, label="anomalies", color="r")

        axs[n].legend()
        n += 1
    plt.show()

    return
