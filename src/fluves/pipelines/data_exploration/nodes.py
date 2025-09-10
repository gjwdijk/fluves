import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhdas.frequency import spectrogram, add_db, energy

def get_event_data(ds_events):
    print(f'get_event_data {type(ds_events)}')
    print(f'get_event_data {ds_events.head()}')
    return ds_events

def process_event_data(ds_events, ds_raw):
    for _, event in ds_events.iterrows():
        # load the event data
        print(event)
        start, end, poi, label = event["start"], event["end"], event["poi"], event["label_anon"]
        poi = np.array([int(x.strip()) for x in poi.split(",")])
        print(poi)
        # select the locations 50m here, 50m there 
        poi = np.arange(4150, 4350, 10)
        print(poi)
        # load the strain data based on the start and the end of the event
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        start = start.tz_localize("UTC")
        end = end.tz_localize("UTC")
        day = start.day
        # if the duration of the event is more than 2 minutes, print the event_label, date and start and end time, and move on to the next row
        event_duration = (end - start).total_seconds()
        if event_duration > 120 or event_duration <= 1:
            continue

        #dir_data = Path(fr"data/{start.day}")
        #file_list = list(aragon_select_files(dir_data, start, end, extension="bin"))
        #if not file_list:
        #    print(f"The list is empty: {start}, {end}, {label}")
        #    continue

        #ds_raw = concat_raw_data(file_list)
        #ds_raw = ds_raw.sel(time=slice(start.tz_localize(None), end.tz_localize(None)), position=poi)
    

    return None


def get_color(label):
    color_map = {
        "label_0": "gray",
        "label_1": "red",
        "label_2": "blue",
        "label_3": "green",
        "label_4": "yellow",
        "label_5": "orange",
        "label_6": "purple",
        "label_7": "magenta",
    }
    return color_map.get(label, "black")  

def plot_with_events(ds_raw, events, start_dt, end_dt, loc):
    print(f'plot_with_events: {events} - pos: {loc}')
    ds_raw = ds_raw.sel(time=slice(pd.Timestamp(start_dt).tz_localize(None), pd.Timestamp(end_dt).tz_localize(None)))
    print(ds_raw)
    ds_spect = spectrogram(ds_raw, variable='strain')
    print(ds_spect)

    loc = np.arange(loc['start'], loc['end'], loc['increment'])
    print(loc)

    events = events[events['poi'].apply(lambda x: any([str(l) in x for l in loc]))]
    loc_events = events[((events["start"] < pd.Timestamp(end_dt).tz_localize(None)) & (events["end"] > pd.Timestamp(start_dt).tz_localize(None)))]
    print(loc_events)
    fig, ax = plt.subplots(figsize=(12,6))
    [ax.plot(ds_raw.time, ds_raw.sel(position=l).strain, label=f"Strain at {l}m") for l in loc]
    
    seen_labels = set()
    
    for _, event in loc_events.iterrows():
        start, end, label = event["start"], event["end"], event["label_anon"]
        color = get_color(label)

        if label not in seen_labels:
            ax.axvspan(start, end, alpha=0.3, label=f"{label}", color=color)
            seen_labels.add(label)  
        else:
            ax.axvspan(start, end, alpha=0.3, color=color) 
            
    ax.set_ylabel("strain [m/m]")
    ax.set_title(f"Strain timeseries at {loc}")
    ax.legend(loc="best", fancybox=True, shadow=True)
    plt.show()

    return plt

