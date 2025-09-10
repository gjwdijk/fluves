"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from .nodes import reshape_and_normalise, build_old_model, train_and_validation_loss, \
    plot_original_reconstructed, predict, slice_and_dice, scale, thresholds, get_anomalies, \
    slice_and_dice2, plot_dates_values, prepare, create_sequences, build_model2, \
    anomalies, thresholds2, slice_and_dice3, plot_dates_values3, create_sequences3, normalize3


def create_pipeline(**kwargs) -> Pipeline:
    p1 = pipeline(
        [
            node(
                func=reshape_and_normalise,
                inputs='sound_level_dataset',
                outputs='sl_model_input',
                name='reshape_and_normalise'
            ),
            node(
                func=build_old_model,
                inputs=[],
                outputs='temp_model',
                name='build_old_model'
            ),
            node(
                func=train_and_validation_loss,
                inputs=['temp_model', 'sl_model_input'],
                outputs=['keras_model', 'loss'],
                name='train_and_validation_loss'
            ),
            node(
                func=predict,
                inputs=['keras_model', 'sl_model_input'],
                outputs=['reconstructed_data', 'reconstruction_error'],
                name='predict'
            ),
            node(
                func=plot_original_reconstructed,
                inputs=['params:sample_idx', 'params:location', 'sl_model_input', 'reconstructed_data'],
                outputs='org_reconstr',
                name='plot_original_reconstructed'
            ),
        ]
    )
    p2 = pipeline(
        [
            node(
                func=slice_and_dice,
                inputs=['sound_level_xdataset_19', 'params:poi'],
                outputs='train',
            ),
             node(
                func=scale,
                inputs='train',
                outputs='normalized_train',
            ),
            node(
                func=predict,
                inputs=['keras_model', 'normalized_train'],
                outputs=['reconstructed_data', 'reconstruction_error'],
            ),
            node(
                func=thresholds,
                inputs='reconstruction_error',
                outputs='thresholds',
            ),
            node(
                func=plot_original_reconstructed,
                inputs=['params:sample_idx', 'params:location', 'normalized_train', 'reconstructed_data'],
                outputs='org_reconstr',
            ),
        ]
    )

    p3 = pipeline(
        [
            node(
                func=slice_and_dice,
                inputs=['sound_level_xdataset_19', 'params:poi'],
                outputs='train',
            ),
             node(
                func=scale,
                inputs='train',
                outputs='normalized_train',
            ),
            node(
                func=build_old_model,
                inputs=[],
                outputs='temp_model',
            ),
            node(
                func=train_and_validation_loss,
                inputs=['temp_model', 'normalized_train'],
                outputs=['keras_model', 'loss'],
            ),
        ]
    )
    p5 = pipeline(
        [
            node(
                func=slice_and_dice,
                inputs=['sound_level_xdataset_25', 'params:poi'],
                outputs='train',
            ),
             node(
                func=scale,
                inputs='train',
                outputs='normalized_test',
            ),
            node(
                func=predict,
                inputs=['keras_model', 'normalized_test'],
                outputs=['reconstructed_data', 'reconstruction_error'],
            ),
            node(
                func=plot_original_reconstructed,
                inputs=['params:sample_idx', 'params:location', 'normalized_test', 'reconstructed_data'],
                outputs='test_reconstr',
            ),
            node(
                func=get_anomalies,
                inputs=['reconstruction_error', 'thresholds'],
                outputs='test_recoanomaliesnstr',
            ),
        ]
    )
        
    p6 = pipeline(
        [
            node(
                func=slice_and_dice2,
                inputs=['sound_level_xdataset_19', 'params:poi'],
                outputs='pdpoi',
            ),
            node(
                func=plot_dates_values,
                inputs='pdpoi',
                outputs=None,
            ),
            node(
                func=prepare,
                inputs='pdpoi',
                outputs=['training_values', 'training_mean', 'training_std', 'train_parameters'],
            ),
            node(
                func=create_sequences,
                inputs='training_values',
                outputs='train_data',
            ),
            node(
                func=build_model2,
                inputs='train_data',
                outputs='keras_model2',
            ),
       ]
    )

    p7 = pipeline(
        [
            node(
                func=thresholds2,
                inputs=['keras_model2', 'train_data'],
                outputs='threshold',
            ),
            node(
                func=slice_and_dice2,
                inputs=['sound_level_xdataset_25', 'params:poi2'],
                outputs='pdpoi_test',
            ),
            node(
                func=anomalies,
                inputs=['pdpoi_test', 'keras_model2', 'train_parameters', 'threshold'],
                outputs=None,
            ),
       ]
    )

    p8 = pipeline(
        [
            node(
                func=normalize3,
                inputs='sound_level_pd_dataset_19',
                outputs='df19',
            ),
            node(
                func=create_sequences3,
                inputs='df19',
                outputs='xtrain',
            ),
            node(
                func=build_model2,
                inputs='xtrain',
                outputs='keras_model3',
            ),
        ]
    )

    p9 = pipeline(
        [
            node(
                func=slice_and_dice3,
                inputs=['sound_level_xdataset_26', 'params:poi2'],
                outputs='sound_level_pd_dataset_26',
            ),
            node(
                func=plot_dates_values3,
                inputs='sound_level_pd_dataset_26',
                outputs=None,
            ),
        ]
    )

    p10 = pipeline(
        [
            node(
                func=thresholds2,
                inputs=['keras_model3', 'train_data'],
                outputs='threshold',
            ),
            node(
                func=slice_and_dice2,
                inputs=['sound_level_xdataset_25', 'params:poi'],
                outputs='pdpoi_test',
            ),
            node(
                func=anomalies,
                inputs=['pdpoi_test', 'keras_model3', 'train_parameters', 'threshold'],
                outputs=None,
            ),
       ]
    )


    return p7
