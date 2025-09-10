from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import xarray as xr

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class XarrayDataset(AbstractDataset[xr.Dataset, xr.Dataset]):
    """``XarrayDataset`` loads / save Xarray data 

    Example:
    ::

        >>> XarrayDataset(filepath='/npy/file/path.nc')
    """

    def __init__(self, filepath: str, metadata: dict[str, Any] | None = None):
        """Creates a new instance of XarrayDataset to load / save image data for given filepath.

        Args:
            filepath: The location of the numpy file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self.metadata = metadata

    def load(self) -> xr.Dataset:
        """Loads data from the xarray file.

        Returns:
            Data from the file as a xarray
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        return xr.open_dataset(self._fs.open(load_path, mode="rb"))
   
    def save(self, data: xr.Dataset) -> None:
        """Saves xarray data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        data.to_netcdf(save_path)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)