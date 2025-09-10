from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path

import numpy as np
import keras
from keras import Model


class KerasModelDataset(AbstractDataset[np.ndarray, np.ndarray]):
    """``KerasModelDataset`` loads / save keras model in keras V3 format

    Example:
    ::

        >>> KerasModelDataset(filepath='/keras/model/model.keras')
    """

    def __init__(self, filepath: str,  metadata: dict[str, Any] | None = None):
        """Creates a new instance of KerasModelDataset to load / save image data for given filepath.

        Args:
            filepath: The location of the keras model file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self.metadata = metadata

    def load(self) -> keras.Model:
        """Loads data from the keras file.

        Returns:
            Data from the file as a keras v3 model
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        return keras.models.load_model(load_path)
   
    def save(self, model: keras.Model) -> None:
        """Saves keras model data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        model.save(save_path)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)