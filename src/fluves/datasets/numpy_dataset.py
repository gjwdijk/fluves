from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import numpy as np

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class NumpyDataset(AbstractDataset[np.ndarray, np.ndarray]):
    """``NumpyDataset`` loads / save Numpy data 

    Example:
    ::

        >>> NumpyDataset(filepath='/npy/file/path.npy')
    """

    def __init__(self, filepath: str, metadata: dict[str, Any] | None = None):
        """Creates a new instance of NumpyDataset to load / save image data for given filepath.

        Args:
            filepath: The location of the numpy file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self.metadata = metadata

    def load(self) -> np.ndarray:
        """Loads data from the numpy file.

        Returns:
            Data from the file as a numpy array
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        return np.load(self._fs.open(load_path, mode="rb"))
   
    def save(self, data: np.ndarray) -> None:
        """Saves numpy data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, mode="wb") as f:
            np.save(f, data)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)