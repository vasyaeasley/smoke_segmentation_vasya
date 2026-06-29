from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.tempo_tensor import (
    TEMPO_L3_CROP_SELECTIONS,
    load_tempo_aerosol_sequence,
    load_tempo_product,
)


class TEMPOWildfireDataset(Dataset):
    """PyTorch Dataset of TEMPO NO2 sequences for next-hour forecasting.

    Each sample is built from four consecutive hourly TEMPO Level 3 NetCDF
    scans: files (i, i+1, i+2) form the 3-channel input tensor X and file
    (i+3) is the single-channel target tensor Y representing the true NO2
    map for the 4th hour. Both X and Y are cropped to the TEMPO L3 study
    region and resampled to a 512 x 512 grid.
    """

    def __init__(
        self,
        folder_path,
        variable_name="vertical_column_troposphere",
        group="product",
        selections=None,
        fill_value=None,
        target_size=(512, 512),
    ):
        """
        Args:
            folder_path (str | Path): Directory containing hourly TEMPO .nc files.
            variable_name (str): TEMPO product variable to extract.
            group (str | None): Optional NetCDF group containing the variable.
            selections (dict | None): Extra xarray selections merged on top of
                the TEMPO L3 cropping window.
            fill_value (float | None): Replacement for NaN / inf pixels.
            target_size (tuple[int, int]): (H, W) the target Y is resampled to.
        """
        self.folder_path = Path(folder_path)
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"TEMPO folder not found: {self.folder_path}")

        # ISO-style timestamps in the filename make a lexicographic sort match
        # chronological order (e.g. TEMPO_NO2_L3_V03_20240801T135308Z_S006.nc).
        self.nc_files = sorted(self.folder_path.glob("*.nc"))

        if len(self.nc_files) < 4:
            raise ValueError(
                f"Need at least 4 TEMPO NetCDF files to build one sample, "
                f"found {len(self.nc_files)} in {self.folder_path}."
            )

        self.variable_name = variable_name
        self.group = group
        self.selections = selections
        self.fill_value = fill_value
        self.target_size = tuple(target_size)

    def __len__(self):
        return len(self.nc_files) - 3

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(f"Index {i} out of range for dataset of length {len(self)}.")

        input_files = self.nc_files[i : i + 3]
        target_file = self.nc_files[i + 3]

        input_sequence = load_tempo_aerosol_sequence(
            input_files,
            variable_name=self.variable_name,
            group=self.group,
            selections=self.selections,
            fill_value=self.fill_value,
        )
        x = input_sequence.tensor

        y = self._load_target(target_file)

        return x, y

    def _load_target(self, nc_file):
        """Load file (i+3) as a (1, H, W) tensor on the same grid as X."""
        product_selections = {**TEMPO_L3_CROP_SELECTIONS, **(self.selections or {})}

        target = load_tempo_product(
            nc_file,
            variable_name=self.variable_name,
            group=self.group,
            selections=product_selections,
            fill_value=self.fill_value,
        )

        target = target.unsqueeze(0).unsqueeze(0)
        target = F.interpolate(
            target,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return target
