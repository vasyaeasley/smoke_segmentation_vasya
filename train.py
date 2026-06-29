"""Train UNetSmall to forecast next-hour atmospheric NO2 from TEMPO sequences.

This is a standard supervised regression setup: a 3-channel stack of consecutive
hourly TEMPO tropospheric-column scans (X) is mapped to the following hour's NO2
field (Y), and the network is optimised with mean squared error.
"""

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

# The dataset module resolves its dependencies with top-level imports such as
# `from utils.tempo_tensor import ...`, so the project's `src/` directory has to
# be importable as a package root before we import TEMPOWildfireDataset.
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.models.unet import UNetSmall
from src.utils.dataset import TEMPOWildfireDataset

DATA_DIR = ROOT_DIR / "tempo_data"
WEIGHTS_PATH = ROOT_DIR / "tempo_unet_weights.pth"
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
# TEMPO NO2 columns are ~1e15-1e16 molecules/cm^2; dividing by this constant
# brings X and Y to ~O(1) so the MSE loss and gradients stay numerically stable.
NO2_SCALE = 1e15


def get_device():
    """Return the best available accelerator: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # fill_value=0.0 replaces NaN/inf pixels from missing TEMPO retrievals so the
    # MSE loss and gradients stay finite during training.
    dataset = TEMPOWildfireDataset(DATA_DIR, fill_value=0.0)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNetSmall(num_channels=3, num_classes=1).to(device)
    criterion = nn.MSELoss()  # regression loss for continuous NO2 values
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        running_loss = 0.0

        for x, y in dataloader:
            # Scale inputs and targets to ~O(1) before the forward/backward pass.
            x = x.to(device) / NO2_SCALE
            y = y.to(device) / NO2_SCALE

            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} - average loss: {average_loss:.6f}")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved model weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
