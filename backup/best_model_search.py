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


MODEL_DIR = "models/"
TRAIN_VAL_LOSS = "train_val_loss/"
ORIGIN_RECONSTRUCT = "original_reconstruct/"
TRAIN_DATA_PATH = "data_for_model/sound_levels_data.npy"

def load_the_data(filepath):

    data = np.load(filepath)
    data = data.transpose(0, 2, 1)
    return data


def load_normal_sound_normalise_per_loc(filepath):
    """Load and preprocess data of the normal day.

    This function loads the sound levels of the normal day, selects the location of interest,
    applies MinMax scaler to normalise  values per location (not global normalisation) to range of (0, 1).

    Returns: 
        numpy.ndarray: The normalized data with shape (1388, 59, 569).
        MinMaxScaler: The fitted MinMaxScaler instances per location.

    """
    # loading sound level data; shape (1388, 569, 59)
    data = np.load(filepath)

    num_positions = data.shape[1] 
    num_timesteps = data.shape[2] 
    num_samples = data.shape[0]  

    scalers = [MinMaxScaler() for _ in range(num_positions)]

    # Apply scaling separately for each position
    for i in range(num_positions):
        flat_values = data[:, i, :].reshape(-1, 1)
        scaled = scalers[i].fit_transform(flat_values)
        data[:, i, :] = scaled.reshape(num_samples, num_timesteps)

    X_train_normalized = data.transpose(0, 2, 1)

    # joblib.dump(scalers, 'scalers.pkl')

    return X_train_normalized


def build_model(X_train_shape, hidden_units, dropout, lr, loss, activation_function):
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
    encoded = LSTM(hidden_units, activation=activation_function, return_sequences=True)(input_seq) 
    encoded = LSTM(hidden_units//2, activation='relu', return_sequences=False)(encoded)
    encoded = Dropout(dropout)(encoded)

    # Latent space
    latent_space = Dense(int(hidden_units), activation='relu')(encoded)

    # Decoder
    decoded = RepeatVector(input_shape[0])(latent_space)  
    decoded = LSTM(hidden_units//2, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(hidden_units, activation=activation_function, return_sequences=True)(decoded)
    decoded = Dropout(dropout)(decoded)

    decoded = TimeDistributed(Dense(input_shape[1]))(decoded)

    model = Model(input_seq, decoded)

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)

    model.summary()

    return model



def plot_original_reconstructed(sample_idx, location, original_series, reconstructed_series, save_path=None):

    feature_idx = int((location - 1260)/10)

    original_series = original_series[sample_idx, :, feature_idx]
    reconstructed_series = reconstructed_series[sample_idx, :, feature_idx]

    plt.figure(figsize=(8, 5))
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



def train_and_evaluate_model(X_train, hidden_units, dropout, lr, loss_fn, activation_function):
    """Train and evaluate a model with given hyperparameters."""

    os.makedirs(MODEL_DIR, exist_ok=True)

    model = build_model(X_train.shape, hidden_units, dropout, lr, loss_fn, activation_function)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, X_train, epochs=300, batch_size=32,
        validation_split=0.2, callbacks=[early_stopping], verbose=1
    )
    file_name = f"LSTM_{hidden_units}hu1_{int(hidden_units/2)}hu2_{dropout}do_{lr}lr_{loss_fn}_{activation_function}"
    model_filename = f"{MODEL_DIR}/{file_name}.keras"
    model.save(model_filename)

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f"Loss Curve ({hidden_units}hu, {dropout}do, {lr}lr, {loss_fn})")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"{TRAIN_VAL_LOSS}/{file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    reconstructed_data = model.predict(X_train, batch_size=32)
    reconstruction_error = np.abs(X_train - reconstructed_data).mean()

    print(f"Reconstruction error: {reconstruction_error}")

    plot_original_reconstructed(500, 4220, X_train, reconstructed_data, f"{ORIGIN_RECONSTRUCT}/{file_name}.png")

    return reconstruction_error, model_filename


def best_model_search():
    """Perform hyperparameter search and save the best model."""

    # Load data
    X_train = load_normal_sound_normalise_per_loc(TRAIN_DATA_PATH)


    # Hyperparameters grid search
    hidden_units_list = [512]
    dropouts = [0.1]
    learning_rates = [0.0001, 0.0005, 0.001]
    loss_functions = ['mae']
    activation_functions = ['tanh']

    best_model = None
    best_error = float('inf')

    # Iterate over all hyperparameter combinations
    for hidden_units, dropout, lr, loss_fn, activation_function in itertools.product(hidden_units_list, dropouts, learning_rates, loss_functions, activation_functions):

        print(f"Training model with {hidden_units}hu, {dropout}do, {lr}lr, {loss_fn}, {activation_function}")

        error, model_filename = train_and_evaluate_model(X_train, hidden_units, dropout, lr, loss_fn, activation_function)


        # Track best model
        if error < best_error:
            best_error = error
            best_model = model_filename

    print(f"\nBest Model: {best_model} with error {best_error:.3f}")

    
if __name__ == "__main__":
    best_model_search()