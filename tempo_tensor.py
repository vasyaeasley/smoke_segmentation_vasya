import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import xarray as xr


TEMPO_L3_CROP_SELECTIONS = {
    "latitude": slice(32.5, 42.0),
    "longitude": slice(-124.5, -114.0),
}


@dataclass(frozen=True)
class TempoSequence:
    """Container for one temporal TEMPO sample."""

    tensor: torch.Tensor
    files: tuple
    variable_name: str


def get_root_coordinates(nc_file, dataset, coordinate_names=("time", "latitude", "longitude")):
    """Read root coordinate values needed by grouped TEMPO product variables."""
    with xr.open_dataset(nc_file) as root_dataset:
        return {
            name: root_dataset[name].values
            for name in coordinate_names
            if name in dataset.sizes and name in root_dataset.variables
        }


def open_tempo_dataset(nc_file, group=None):
    """Open a TEMPO NetCDF file as an xarray Dataset."""
    dataset = xr.open_dataset(nc_file, group=group)
    if group:
        coordinates = get_root_coordinates(nc_file, dataset)
        if coordinates:
            dataset = dataset.assign_coords(coordinates)

    return dataset


def get_data_array(dataset, variable_name):
    """Find a product variable in a TEMPO Dataset, allowing case-insensitive names."""
    if variable_name in dataset:
        return dataset[variable_name]

    matches = [name for name in dataset.data_vars if name.lower() == variable_name.lower()]
    if matches:
        return dataset[matches[0]]

    available = ", ".join(dataset.data_vars)
    raise KeyError(f"Variable '{variable_name}' was not found. Available variables: {available}")


def select_product_layer(data_array, selections=None):
    """Select one layer from a product variable before converting it to a 2D field.

    TEMPO products may include singleton dimensions, vertical layers, scan indices,
    or wavelength/product dimensions. Pass selections like
    {'layer': 0, 'wavelength': 354} when the variable does not squeeze to 2D.
    """
    if selections:
        data_array = data_array.sel(selections)

    return data_array.squeeze(drop=True)


def data_array_to_tensor(data_array, fill_value=None):
    """Convert one 2D TEMPO xarray DataArray to a float32 H x W tensor."""
    data_array = data_array.squeeze(drop=True)

    if data_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D geospatial field after squeezing singleton dimensions, "
            f"but got shape {data_array.shape}. Select a product layer before stacking."
        )

    values = data_array.astype("float32").values
    tensor = torch.from_numpy(values)
    if fill_value is not None:
        tensor = torch.nan_to_num(tensor, nan=fill_value, posinf=fill_value, neginf=fill_value)

    return tensor


def load_tempo_product(nc_file, variable_name="aerosol_index", group=None, selections=None, fill_value=None):
    """Load one TEMPO product variable from a NetCDF file as an H x W tensor."""
    with open_tempo_dataset(nc_file, group=group) as dataset:
        data_array = get_data_array(dataset, variable_name)
        data_array = select_product_layer(data_array, selections=selections)
        return data_array_to_tensor(data_array, fill_value=fill_value)


def stack_tempo_sequence(nc_files, variable_name="aerosol_index", group=None, selections=None, fill_value=None):
    """Load three consecutive hourly TEMPO NetCDF scans as a C x H x W tensor.

    The returned channels are ordered like the input file sequence, so pass files
    as [t-2, t-1, t0].
    """
    if len(nc_files) != 3:
        raise ValueError("Exactly three consecutive TEMPO NetCDF files are required: t-2, t-1, and t0.")

    arrays = []
    reference_shape = None

    for nc_file in nc_files:
        tensor = load_tempo_product(
            nc_file,
            variable_name=variable_name,
            group=group,
            selections=selections,
            fill_value=fill_value,
        )

        if reference_shape is None:
            reference_shape = tensor.shape
        elif tensor.shape != reference_shape:
            raise ValueError(
                f"All TEMPO scans must share the same grid before stacking. "
                f"Expected {reference_shape}, got {tensor.shape} for {nc_file}."
            )

        arrays.append(tensor)

    return torch.stack(arrays, dim=0)


def load_tempo_aerosol_sequence(
    nc_files,
    variable_name="vertical_column_troposphere",
    group="product",
    selections=None,
    fill_value=None,
):
    """Load a cropped three-hour TEMPO Level 3 tropospheric-column sequence."""
    product_selections = {**TEMPO_L3_CROP_SELECTIONS, **(selections or {})}
    tensor = stack_tempo_sequence(
        nc_files,
        variable_name=variable_name,
        group=group,
        selections=product_selections,
        fill_value=fill_value,
    )
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(512, 512),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return TempoSequence(tensor=tensor, files=tuple(nc_files), variable_name=variable_name)


def parse_selections(selection_args):
    """Parse CLI selections formatted as dim=value into an xarray selection dict."""
    if not selection_args:
        return None

    selections = {}
    for selection in selection_args:
        if "=" not in selection:
            raise ValueError(f"Selection '{selection}' must be formatted as dimension=value.")

        dimension, value = selection.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass

        selections[dimension] = value

    return selections


def save_tempo_tensor(tensor, output_path):
    """Save a stacked TEMPO tensor to disk for later Dataset integration."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Stack three hourly TEMPO Level 3 NO2 NetCDF files into a cropped C x H x W PyTorch tensor."
    )
    parser.add_argument("nc_files", nargs=3, help="Input TEMPO NetCDF files ordered as t-2 t-1 t0.")
    parser.add_argument(
        "--group",
        default="product",
        help="Optional NetCDF group containing the TEMPO product variables.",
    )
    parser.add_argument(
        "--variable",
        default="vertical_column_troposphere",
        help="TEMPO NetCDF variable to extract. Use the exact product variable name if it differs.",
    )
    parser.add_argument(
        "--select",
        action="append",
        help="Optional xarray selection formatted as dimension=value. Repeat for multiple dimensions.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        help="Optional value used to replace NaN and infinite pixels before tensor conversion.",
    )
    parser.add_argument("--out", help="Optional .pt output path for torch.save.")
    args = parser.parse_args()

    sequence = load_tempo_aerosol_sequence(
        args.nc_files,
        variable_name=args.variable,
        group=args.group,
        selections=parse_selections(args.select),
        fill_value=args.fill_value,
    )

    if args.out:
        save_tempo_tensor(sequence.tensor, args.out)

    print(f"tensor shape: {tuple(sequence.tensor.shape)}")
    print(f"dtype: {sequence.tensor.dtype}")


if __name__ == "__main__":
    main()