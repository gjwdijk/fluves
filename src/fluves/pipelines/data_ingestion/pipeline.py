from typing import Any

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import read_sl_data, get_raw_spectogram_data, get_raw_data, get_soundlevel_data, plot_soundlevel, concat_soundlevel_data, get_file_list, get_raw_soundlevel_data


def create_pipeline(**kwargs: Any) -> Pipeline:
    p1 = pipeline(
        [
           node(
                func=concat_soundlevel_data,
                inputs=['fluves_25', 'params:start_date_25', 'params:end_date_25', 'params:poi'],
                outputs='sound_level_dataset_25',
                name='normal_soundlevel_data'
            ),
        ]
    )
    p2 = pipeline(
        [
           node(
                func=get_file_list,
                inputs=['fluves_19',],
                outputs='file_list',
            ),
           node(
                func=get_raw_soundlevel_data,
                inputs='file_list',
                outputs='sound_level_xdataset_19',
            ),
           node(
                func=get_file_list,
                inputs=['fluves_26',],
                outputs='file_list_26',
            ),
           node(
                func=get_raw_soundlevel_data,
                inputs='file_list_26',
                outputs='sound_level_xdataset_26',
            ),
        ]
    )
    p3 = pipeline(
        [
          node(
                func=read_sl_data,
                inputs=['sound_level_xdataset_25', 'params:poi'],
                outputs=None,
            ),
        ]
    )
    p4 = pipeline(
        [
          node(
                func=get_raw_spectogram_data,
                inputs=['fluves_25', 'params:variable'],
                outputs='spectogram_strain_xdataset_25',
            ),
        ]
    )

    p5 = pipeline(
        [
           node(
                func=get_file_list,
                inputs=['fluves_19', 'params:start_date_19', 'params:end_date_19'],
                outputs='file_list',
            ),
           node(
                func=get_raw_data,
                inputs=['file_list', 'params:start_date_19', 'params:end_date_19', 'params:poi2'],
                outputs=None,
            ),
        ]
    )


    return p5
