from pathlib import PurePosixPath
from typing import Any, Dict, Tuple
from pathlib import Path
from datetime import timezone, datetime

import fsspec
import numpy as np
import xarray as xr
import pandas as pd


from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path

from pyhdas.aragon import concat_raw_data, aragon_select_files, aragon_load_data


class FluvesDataset(AbstractDataset[np.ndarray, np.ndarray]):
    """``FluvesDataset`` loads / save fluves data from a given filepath as `numpy` array

    Example:
    ::
        >>> FluvesDataset(filepath='${FLUVES_DATA_DIR}/2021_02_19_23h55m41s_HDAS_2DRawData_Strain.bin')
    """

    def __init__(self, filepath: str, metadata: dict[str, Any] | None = None):
        """Creates a new instance of FluvesDataset to load / save data for given filepath.

        Args:
            filepath: The location of the file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self.metadata = metadata

    def load(self) -> Tuple[xr.Dataset, str]:
        """Loads data from the file.

        Returns:
            Data from the file as a numpy array
            The full filename of the loaded file
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        aragon = aragon_load_data(load_path)
        return aragon, load_path

    def save(self, data: np.ndarray) -> None:
        """Save data is not supported."""
        save_path = get_filepath_str(self._filepath, self._protocol)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)

