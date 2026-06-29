import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import torch
from utils.tempo_tensor import load_tempo_aerosol_sequence


def test_real_pipeline():
    print("=" * 50)
    print("Testing Pipeline with Real TEMPO Data")
    print("=" * 50)

    data_dir = Path("tempo_data")

    real_files = sorted([data_dir / f for f in os.listdir(data_dir) if f.endswith(".nc")])

    if len(real_files) < 3:
        print(f"Error: You need at least 3 files in 'tempo_data' to test. Found {len(real_files)}.")
        return

    test_sequence = real_files[:3]
    print("Loading sequence:")
    for f in test_sequence:
        print(f" - {f.name}")

    try:
        sequence_data = load_tempo_aerosol_sequence(test_sequence)
        tensor = sequence_data.tensor

        print("\n Extraction Success!")
        print(f"Tensor Shape: {tuple(tensor.shape)} (channels, height, width)")
        print(f"Tensor Data Type: {tensor.dtype}")

        nan_count = torch.isnan(tensor).sum().item()
        total_pixels = tensor.numel()
        print(f"NaN (missing data) pixels: {nan_count} out of {total_pixels} ({nan_count / total_pixels * 100:.1f}%)")

    except Exception as e:
        print(f"\n Pipeline Failed: {e}")


if __name__ == "__main__":
    test_real_pipeline()
