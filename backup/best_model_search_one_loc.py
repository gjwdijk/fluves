import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import os
import itertools
import joblib

from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from pathlib import Path
import seaborn as sns
from sklearn.metrics import auc

TRAIN_FILEPATH = 'data_for_model/sound_levels_data.npy'
SCALERS_PATH = 'scalers'
MODELS_PATH = 'models'
TRAIN_VAL_LOSS = 'train_val_loss/'
ORIGIN_RECONSTRUCT = 'original_reconstruct/'
POSITION_TABLE_PATH = 'position_table.csv'
LOCATIONS_TO_RETRAIN = [3640, 4440, 4640, 4740, 4940]


def get_locations(position_table_path):

    # load the table
    position_table = pd.read_csv(position_table_path)

    # get only quiet locations
    quiet_positions = position_table[position_table['location_name'] == 'quiet']['position_fiber'].values
    quiet_set = set(quiet_positions)

    # Define the starting point, step, and ending point
    start = 6240
    end = 6840
    step = 100
    search_radius = 50  # +/-50

    selected_positions = []

    for target in range(start, end + 1, step):
        if target in quiet_set:
            selected_positions.append(target)
        else:
            # Search in range [target - 50, target + 50]
            nearby_range = range(target - search_radius, target + search_radius + 1)
            quiet_nearby = [pos for pos in nearby_range if pos in quiet_set]

            if quiet_nearby:
                # Pick the one closest to target
                closest = min(quiet_nearby, key=lambda x: abs(x - target))
                selected_positions.append(closest)
            else:
                print(f"No quiet location found near {target}, skipping...")
    
    return selected_positions


def build_model(X_train_shape, hidden_units, dropout, lr, loss, activation_func):
    """Building of the model. 

    This function builds an LSTM-Autoencoder model based on the specified hyperparemeters.

    Args:
        X_train_shape (tuple): Shape of the input training data (batch_size, timesteps, features).
        hidden_units (int): Number of LSTM units in the encoder and decoder.
        dropout (float): Dropout rate to apply after LSTM layers.
        lr (float): Learning rate for the optimizer.
    loss (str): Loss function to be used for training.

    Returns:
        tensorflow.keras.Model: Compiled LSTM-Autoencoder model.
    
    """

    input_shape = (X_train_shape[1],X_train_shape[2])  # (time_steps, features)

    # Encoder
    input_seq = Input(shape=input_shape)
    encoded = LSTM(hidden_units, activation=activation_func, return_sequences=True)(input_seq) 
    encoded = LSTM(hidden_units//2, activation=activation_func, return_sequences=False)(encoded)
    encoded = Dropout(dropout)(encoded)

    # Latent space
    latent_space = Dense(hidden_units, activation='relu')(encoded)

    # Decoder
    decoded = RepeatVector(input_shape[0])(latent_space)  
    decoded = LSTM(hidden_units//2, activation=activation_func, return_sequences=True)(decoded)
    decoded = LSTM(hidden_units, activation=activation_func, return_sequences=True)(decoded)
    decoded = Dropout(dropout)(decoded)

    decoded = TimeDistributed(Dense(input_shape[1]))(decoded)

    model = Model(input_seq, decoded)

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)

    model.summary()

    return model


def load_normalise_one_loc_split_train_test(train_path, location):

    data = np.load(train_path)

    loc = int((location - 1260)/10)
    data = data[:, loc, :]

    X_train, X_test = train_test_split(data, test_size=0.2, random_state=12)

    print(f"MIN:{np.min(X_train)}, MAX:{np.max(X_train)}")

    X_train_shape = X_train.shape
    X_test_shape = X_test.shape

    scaler = MinMaxScaler()

    X_train_flat_values = X_train.reshape(-1, 1)
    X_train_normalised = scaler.fit_transform(X_train_flat_values)
    X_train_normalised = X_train_normalised.reshape(X_train_shape[0], X_train_shape[1], 1)

    X_test_flat_values = X_test.reshape(-1, 1)
    X_test_normalised = scaler.transform(X_test_flat_values)
    X_test_normalised = X_test_normalised.reshape(X_test_shape[0], X_test_shape[1], 1)

    joblib.dump(scaler, f'{SCALERS_PATH}/scaler_{location}.pkl')

    return X_train_normalised, X_test_normalised, scaler



def plot_original_reconstructed_one_loc(location, original_series, reconstructed_series, save_path=None):

    plt.figure(figsize=(12, 5))
    plt.plot(original_series, label="Original", linestyle="--", marker="o", color="royalblue")
    plt.plot(reconstructed_series, label="Reconstructed", linestyle="-", marker="s", color="darkorange")

    plt.xlabel("Time Step")
    plt.ylabel("Feature Value")
    plt.ylim((0, 1))
    plt.title(f"Original vs. Reconstructed Series (Location {location})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  
    else:
        plt.show()


def train_and_evaluate_model(X_train, hidden_units, dropout, lr, loss_fn, activation_function, location):
    """Train and evaluate a model with given hyperparameters."""

    os.makedirs(MODELS_PATH, exist_ok=True)

    model = build_model(X_train.shape, hidden_units, dropout, lr, loss_fn, activation_function)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, X_train, epochs=200, batch_size=32,
        validation_split=0.2, callbacks=[early_stopping], verbose=1
    )

    epochs_trained = len(history.history['loss'])

    if epochs_trained < 100:
        print(f"Model for location {location} trained only {epochs_trained} epochs. Retrying with lower LR (0.0005).")

        model = build_model(X_train.shape, hidden_units, dropout, 0.0005, loss_fn, activation_function)
        history = model.fit(
            X_train, X_train, epochs=200, batch_size=32,
            validation_split=0.2, callbacks=[early_stopping], verbose=1
        )

    # file_name = f"LSTM_loc_{location}_{hidden_units}hu1_{int(hidden_units/2)}hu2_{dropout}do_{lr}lr_{loss_fn}_{activation_function}"
    model_filename = f"{MODELS_PATH}/location_{location}.keras"
    model.save(model_filename)

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f"Loss Curve ({hidden_units}hu, {dropout}do, {lr}lr, {loss_fn})")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"{TRAIN_VAL_LOSS}/location_{location}.png", dpi=300, bbox_inches='tight')
    plt.close()

    reconstructed_data = model.predict(X_train, batch_size=32)
    reconstruction_error = np.abs(X_train - reconstructed_data)

    print(f"Mean: {np.mean(reconstruction_error)}, Min: {np.min(reconstruction_error)}, Max: {np.max(reconstruction_error)}, 99th Percentile: {np.percentile(reconstruction_error, 99)}")


    sample = 500
    plot_original_reconstructed_one_loc(location, X_train[sample], reconstructed_data[sample], f"{ORIGIN_RECONSTRUCT}/location_{location}.png")

    return np.mean(reconstruction_error), model_filename


def best_model_search(location):
    """Perform hyperparameter search and save the best model."""

    # Load data
    X_train, X_test, scaler = load_normalise_one_loc_split_train_test(TRAIN_FILEPATH, location)


    # Hyperparameters grid search
    hidden_units_list = [512]
    dropouts = [0.2]
    learning_rates = [0.001]
    loss_functions = ['mae']
    activation_functions = ['tanh']

    # best_model = None
    # best_error = float('inf')

    # Iterate over all hyperparameter combinations
    for hidden_units, dropout, lr, loss_fn, activation_function in itertools.product(hidden_units_list, dropouts, learning_rates, loss_functions, activation_functions):

        print(f"Training model on location {location}")

        error, model_filename = train_and_evaluate_model(X_train, hidden_units, dropout, lr, loss_fn, activation_function, location)

    #     # Track best model
    #     if error < best_error:
    #         best_error = error
    #         best_model = model_filename

    # print(f"\nBest Model: {best_model} with error {best_error:.3f}")


def train_on_locations():

    locations = get_locations(POSITION_TABLE_PATH)

    for loc in locations:
        best_model_search(loc)

    
if __name__ == "__main__":
    train_on_locations()