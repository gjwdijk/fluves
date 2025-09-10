"""
This is a boilerplate pipeline 'data_exploration'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from fluves.pipelines.data_ingestion.nodes import concat_soundlevel_data
from .nodes import get_event_data, process_event_data, plot_with_events
from fluves.pipelines.data_ingestion.nodes import get_raw_data_file_by_file, get_file_list

def create_pipeline(**kwargs) -> Pipeline:

    p1 = pipeline(
        [
            node(
                func=get_event_data,
                inputs='events',
                outputs='iter',
                name='get_event_data'
            ),
            node(
                func=process_event_data,
                inputs=['iter', 'sound_level_dataset_19'],
                outputs=None,
            ),
        ]
    )

    p2 = pipeline(
        [    
           node(
                func=get_file_list,
                inputs=['fluves_25', 'params:start_date_25', 'params:end_date_25'],
                outputs='files',
            ),
           node(
                func=get_raw_data_file_by_file,
                inputs=['files', 'params:start_date_25', 'params:end_date_25', 'params:poi2'],
                outputs='raw',
            ),
            node(
                func=plot_with_events,
                inputs=['raw', 'events', 'params:start_date_25', 'params:end_date_25', 'params:poi2'],
                outputs=None,
            ),
        ]
    )

    return p2
