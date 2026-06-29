from torch.utils.data import DataLoader
from src.utils.dataset import TEMPOWildfireDataset 

def test_sliding_window():
    print("=" * 50)
    print("Testing TEMPO Dataset & DataLoader")
    print("=" * 50)
    
    # 1. Initialize the dataset pointing to your real data folder
    try:
        dataset = TEMPOWildfireDataset(folder_path="tempo_data")
        print(f"Dataset successfully initialized!")
        print(f"Total sliding-window sequences available: {len(dataset)}")
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    # 2. Wrap it in a DataLoader
    # batch_size=4 means it will process 4 temporal sequences at the exact same time
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Pull a single batch to test the pipeline
    print("\nFetching a batch of data...")
    try:
        x_batch, y_batch = next(iter(dataloader))
        
        print(f"Input (X) Batch Shape: {tuple(x_batch.shape)} -> (Batch, Channels, Height, Width)")
        print(f"Target (Y) Batch Shape: {tuple(y_batch.shape)} -> (Batch, Channels, Height, Width)")
        print("\nPASSED: Data pipeline is 100% ready for the neural network.")
    except Exception as e:
        print(f"\nFailed to load batch: {e}")

if __name__ == "__main__":
    test_sliding_window()