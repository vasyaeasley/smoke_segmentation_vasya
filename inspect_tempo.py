import xarray as xr
import os

def inspect_first_file():
    # Resolve data folder from this script's location so cwd does not matter.
    folder_path = os.path.dirname(os.path.abspath(__file__))
    
    # Grab all the .nc files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
    
    if not files:
        print("No NetCDF files found in 'tempo_data'.")
        return
        
    first_file = os.path.join(folder_path, files[0])
    print(f"Opening: {first_file}\n")
    print("=" * 40)
    
    # Load the NetCDF file
    # TEMPO data sometimes uses nested groups, but xarray handles the root gracefully
    ds = xr.open_dataset(first_file, group='product')
    
    print("DATA VARIABLES:")
    for var in ds.data_vars:
        dims = ds[var].dims
        shape = ds[var].shape
        print(f" - {var}: dimensions {dims} | shape {shape}")
        
    print("-" * 40)
    print("COORDINATES:")
    for coord in ds.coords:
        print(f" - {coord}: shape {ds[coord].shape}")

if __name__ == "__main__":
    inspect_first_file()