import tempfile
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from src.models.unet import UNetSmall
from tempo_tensor import load_tempo_aerosol_sequence


def _make_fake_nc(path, value, height=128, width=128):
    """Write a minimal TEMPO L3-shaped NetCDF file with a constant tropospheric column."""
    root = xr.Dataset(
        coords={
            "time": np.array([np.datetime64("2024-08-01T00:00:00")]),
            "latitude": np.linspace(32.5, 42.0, height, dtype="float32"),
            "longitude": np.linspace(-124.5, -114.0, width, dtype="float32"),
        }
    )
    product = xr.Dataset(
        {
            "vertical_column_troposphere": (
                ("time", "latitude", "longitude"),
                np.full((1, height, width), value, dtype="float32"),
            )
        }
    )

    root.to_netcdf(path)
    product.to_netcdf(path, group="product", mode="a")


def test_tempo_loader(tmp_dir, height=128, width=128):
    print("\n--- 1. TEMPO loader (tempo_tensor.py) ---")
    files = []
    for i, value in enumerate([0.1, 0.5, 0.9]):   # t-2, t-1, t0
        path = tmp_dir / f"tempo_t{i}.nc"
        _make_fake_nc(path, value, height, width)
        files.append(path)

    sequence = load_tempo_aerosol_sequence(files)
    t = sequence.tensor                            # (3, H, W)

    assert t.shape == (3, height, width), f"Unexpected loader shape: {t.shape}"
    assert t.dtype == torch.float32

    # Each channel should reflect the constant we wrote
    for ch, expected in enumerate([0.1, 0.5, 0.9]):
        assert abs(t[ch, 0, 0].item() - expected) < 1e-5, \
            f"Channel {ch} value mismatch: {t[ch, 0, 0].item()} != {expected}"

    print(f"  loader tensor shape : {tuple(t.shape)}   (channels, height, width)")
    print(f"  channel pixel values: t-2={t[0,0,0]:.2f}  t-1={t[1,0,0]:.2f}  t0={t[2,0,0]:.2f}")
    print("  PASSED")
    return t


def test_model_forward(single_sample_tensor):
    print("\n--- 2. UNetSmall forward pass (unet.py) ---")
    batch_size = 4
    channels, height, width = single_sample_tensor.shape

    # Stack four copies to form a batch: (batch_size, 3, H, W)
    batch = single_sample_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)
    print(f"  input batch shape   : {tuple(batch.shape)}")

    model = UNetSmall(num_channels=channels, num_classes=1)
    model.eval()
    with torch.no_grad():
        output = model(batch)

    expected_output_shape = (batch_size, 1, height, width)
    assert output.shape == expected_output_shape, \
        f"Unexpected output shape: {output.shape}, expected {expected_output_shape}"

    print(f"  output batch shape  : {tuple(output.shape)}")
    print("  PASSED")


def test_pipeline():
    print("=" * 50)
    print("Smoke-segmentation TEMPO pipeline sanity check")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sample_tensor = test_tempo_loader(tmp_dir)
        test_model_forward(sample_tensor)

    print("\nAll checks passed.")


if __name__ == "__main__":
    test_pipeline()