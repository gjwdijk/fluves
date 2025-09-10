from pathlib import PurePosixPath
from typing import Any, Dict, Union, Callable, List
from pathlib import Path
from datetime import timezone, datetime

import fsspec
import numpy as np
import xarray as xr
import pandas as pd

from pyhdas.frequency import spectrogram, add_db, energy
from pyhdas.aggregate import quantile
from pyhdas.aragon import concat_raw_data, aragon_select_files, aragon_load_data

import matplotlib.pyplot as plt

from kedro_datasets.partitions import PartitionedDataset

def datetime_file_in_range(filename: str, 
                           start: Union[pd.Timestamp, datetime, str] = None,
                           end: Union[pd.Timestamp, datetime, str]= None) -> bool:
    datetime_file = pd.to_datetime(filename[:20], format="%Y_%m_%d_%Hh%Mm%Ss").tz_localize("UTC")
    start = pd.Timestamp(start) if isinstance(start, str) else start
    end = pd.Timestamp(end) if isinstance(end, str) else end
    if start and end:
        if start - pd.Timedelta(61, unit="s") <= datetime_file <= end:
            return True
        else:
            return False
    return True

def get_file_list(partitioned_input: Dict[str, Callable[[], Any]], 
                  start: Union[pd.Timestamp, datetime, str] = None,
                  end: Union[pd.Timestamp, datetime, str]= None) -> List[str]:
    print(f'get_file_list: {start} - {end}')
    file_list = []
    for partition_key, partition_load_func in sorted(partitioned_input.items()):
        if datetime_file_in_range(partition_key, start=start, end=end):
            _, filename = partition_load_func()
            print(f'get_file_list: {partition_key} - {filename}')
            file_list.append(filename)

    if not file_list:
        print(f"The list is empty: {start}, {end}")

    return file_list

def get_raw_data(file_list: List[str], 
                 start: Union[pd.Timestamp, datetime, str],
                 end: Union[pd.Timestamp, datetime, str],
                 poi: Dict):

    print(f'get_raw_data {file_list} {start}-{end}')
    start = pd.Timestamp(start) if isinstance(start, str) else start
    end = pd.Timestamp(end) if isinstance(end, str) else end

    print(f'BEFORE concat_raw_data {poi} {type(poi)}')
    ds_raw = concat_raw_data(file_list)
    print(f'AFTER concat_raw_data')
    start = start.tz_localize(None)
    end = end.tz_localize(None)
    print(f'BEFORE ds_raw.sel')
    # slice
    # ds_raw = ds_raw.sel(time=slice(start, end))
    poi = np.arange(poi['start'], poi['end'], poi['increment'])
    ds_raw = ds_raw.sel(time=slice(start, end), position=poi)
    print(f'AFTER ds_raw.sel {ds_raw}')
    return ds_raw

def get_raw_data_file_by_file(file_list: List[str], 
                 start: Union[pd.Timestamp, datetime, str],
                 end: Union[pd.Timestamp, datetime, str],
                 poi: Dict):

    print(f'get_raw_data {file_list} {start}-{end}')
    start = pd.Timestamp(start) if isinstance(start, str) else start
    end = pd.Timestamp(end) if isinstance(end, str) else end
    start = start.tz_localize(None)
    end = end.tz_localize(None)
    poi = np.arange(poi['start'], poi['end'], poi['increment'])
    print(f'BEFORE concat_raw_data {poi} {type(poi)}')

    data = []
    for _, file in enumerate(file_list):
        print(f'get_raw_data2 {file}') 
        ds_raw = concat_raw_data([file])
        ds_raw = ds_raw.sel(time=slice(start, end), position=poi)
        data.append(ds_raw)
    xr_data = xr.concat(data, dim="time").sortby("time")
    print(f'AFTER concat_raw_data {xr_data}')   

    return xr_data

def get_raw_soundlevel_data(file_list: List[str]):
    print(f'get_raw_soundlevel_data {len(file_list)}')
    data = []
    for _, file in enumerate(file_list):
        print(f'get_raw_soundlevel_data {file}') 
        ds_raw = concat_raw_data([file])
        ds_soundlevel, X_train = get_soundlevel_data(ds_raw=ds_raw)
        data.append(ds_soundlevel)
    xr_data = xr.concat(data, dim="time").sortby("time")
    print(f'AFTER concat_raw_data {xr_data}')   
    print(f'AFTER get_soundlevel_data {ds_soundlevel.head()}')
    return xr_data

def get_raw_spectogram_data(partitioned_input: Dict[str, Callable[[], Any]], variable: str = 'strain'):
    # GJ - This generates full dataset and will result in memory failures when concat
    data = []
    for partition_key, partition_load_func in sorted(partitioned_input.items())[:5]:
        _, filename = partition_load_func()
        print(f'get_raw_spectogram_data {filename}') 
        ds_raw = concat_raw_data([filename])
        ds_spect = spectrogram(ds_raw, variable=variable)
        print(ds_spect)
        data.append(ds_spect)
    xr_data = xr.concat(data, dim="time").sortby("time")
    print(f'AFTER get_raw_spectogram_data')
    return xr_data

def get_soundlevel_data(ds_raw):
    # excluding the end of the pipe
    #poi = np.arange(60, 6950, 10)
    #poi = np.arange(1260, 6950, 10)
    #ds_raw = ds_raw.sel(position=poi)

    ds_spect = spectrogram(ds_raw, variable='strain')
    ds_soundlevel = ds_spect[["Pxx"]].sum(dim="freq")
    ds_soundlevel = add_db(ds_soundlevel)   
    print(f'get_soundlevel_data {type(ds_soundlevel)}')
    print(ds_soundlevel)

    X_train = ds_soundlevel.Pxx_dB.values  # Extracting the numpy array
    print(X_train.shape)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    print(X_train.shape)
    print(f'get_soundlevel_data {type(X_train)}')
    if X_train.shape[1] != 59:
        return ds_soundlevel, None

    return ds_soundlevel, X_train

def get_soundlevel_from_file(filename: str, poi: Dict = {}):
    print(f'get_soundlevel_from_file {filename} - {poi}')
    ds_raw = concat_raw_data([filename])
    if poi:
        ds_raw = ds_raw.sel(position=np.arange(poi['start'], poi['end'], poi['increment']))
    _, soundlevel_data = get_soundlevel_data(ds_raw=ds_raw)
    return soundlevel_data

def get_timestamp_from_filename(filename: str) -> str:
    fn = filename.split('/')[-1]  # Timestamp is in the .bin filename at the end of full filename path
    file_timestamp = "_".join(fn.split('_')[:4])  
    print(f'get_timestamp_from_filename for {file_timestamp}')
    return file_timestamp

def concat_soundlevel_data(partitioned_input: Dict[str, Callable[[], Any]],
                           start: Union[pd.Timestamp, datetime, str] = None,
                           end: Union[pd.Timestamp, datetime, str] = None,
                           poi: Dict = {}):
    data = []
    for partition_key, partition_load_func in sorted(partitioned_input.items()):
        if datetime_file_in_range(partition_key, start=start, end=end):
            _, filename = partition_load_func()
            sl = get_soundlevel_from_file(filename, poi=poi)
            if sl is not None:
                file_timestamp = get_timestamp_from_filename(filename)
                data.append((file_timestamp, sl))
    dtype = [('filename', 'U256'), ('data', 'O')]
    return np.array(data, dtype=dtype)

def plot_soundlevel(ds_soundlevel):
    # Plot soundlevel over time and all positions
    fig,ax = plt.subplots(figsize=(15,6))
    ds_soundlevel.Pxx_dB.plot(x="position", y="time", ax=ax, vmax = 40, cmap="magma")
    ax.set_title(f"soundlevel [dB] over all frequencies")

    # save in the normal data directory
    # output_folder = Path(f"normal_soundlevel_plots_all_locs")
    # output_folder.mkdir(parents=True, exist_ok=True)

    file = "FLUVES12345678"
    plot_filename = f"{file[:-4]}.png"
    print(f'plot_soundlevel: {plot_filename}')
    
    # plt.savefig(plot_filename,  bbox_inches='tight')

    # plt.close(fig)
    return plt


def compute_spectral_centroid(spectrogram_data):
    
    freq = spectrogram_data['freq'].values 
    Pxx = spectrogram_data['Pxx'].values  

    # Calculate spectral centroid for each time step
    numerator = np.sum(Pxx * freq[:, np.newaxis], axis=0)  
    denominator = np.sum(Pxx, axis=0)  
    spectral_centroid = numerator / denominator  

    return spectral_centroid

def compute_spectral_bandwidth(spectrogram_data):
    
    freq = spectrogram_data['freq'].values  # Frequencies (Hz)
    Pxx = spectrogram_data['Pxx'].values   # Power spectral density

    spectral_centroid = compute_spectral_centroid(spectrogram_data)

    # Calculate Spectral Bandwidth
    numerator_bandwidth = np.sum(Pxx * ((freq[:, np.newaxis] - spectral_centroid)**2), axis=0)
    denominator_bandwidth = np.sum(Pxx, axis=0)
    spectral_bandwidth = np.sqrt(numerator_bandwidth / denominator_bandwidth)

    return spectral_bandwidth

def compute_spectral_flatness(spectrogram_data):

    Pxx = spectrogram_data['Pxx'].values  # Power spectral density
    
    geometric_mean = np.exp(np.mean(np.log(Pxx), axis=0))  # Geometric mean
    arithmetic_mean = np.mean(Pxx, axis=0)  # Arithmetic mean
    
    spectral_flatness = geometric_mean / arithmetic_mean
    return spectral_flatness


def create_spectral_features_plot(ds_spect, poi):
    n_positions = len(poi)

    max_columns = 2
    n_rows = math.ceil(n_positions / max_columns)
    fig, axes = plt.subplots(n_rows, max_columns, figsize=(15, 5 * n_rows), constrained_layout=True)

    axes = axes.flatten()
    flag = True
    
    if n_positions == 1:  
        fig, axes = plt.subplots(n_positions, 1, figsize=(10, 5 * n_positions), constrained_layout=True)
        axes = [axes]
        flag = False

    # For every position, calculate and plot spectral centroid and bandwidth
    for idx, pos in enumerate(ds_spect.position):
        ax = axes[idx]
        spectral_centroid = compute_spectral_centroid(ds_spect.sel(position=pos))
        spectral_bandwidth = compute_spectral_bandwidth(ds_spect.sel(position=pos), spectral_centroid)

        ax.plot(ds_spect.time, spectral_centroid, label="Spectral Centroid", color='r')
        ax.plot(ds_spect.time, spectral_bandwidth, label="Spectral Bandwidth", color='b')
        ax.set_title(f"Spectral Features at {pos.values:.1f}m")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    if flag:
        for ax in axes[n_positions:]:
            fig.delaxes(ax)

    output_folder = Path(f"spectral_features_plots/{label}")
    output_folder.mkdir(parents=True, exist_ok=True)

    plot_filename = f"{start.strftime('%Y-%m-%d_%H:%M:%S')}_{end.strftime('%H:%M:%S')}_bdw_and_sc.png"
    plt.savefig(output_folder / plot_filename)

    plt.close(fig)
    return plt

def read_sl_data(ds_raw, poi: Dict):
    print(ds_raw)
    ds_raw = ds_raw.sel(position=np.arange(poi['start'], poi['end'], poi['increment']))
    print(ds_raw)


