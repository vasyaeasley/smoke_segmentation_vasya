import earthaccess
import os

def download_wildfire_data():
    print("Logging into NASA Earthdata...")
    # This will prompt you for your username/password the first time, 
    # then save it securely for future runs.
    auth = earthaccess.login(persist=True)

    # 1. Define the search parameters
    # TEMPO_NO2_L3 is Level 3 NO2 data. 
    # (You can change this to a different short_name if your digital twin requires it)
    short_name = "TEMPO_NO2_L3" 
    
    # Define a bounding box around your target wildfire [West, South, East, North]
    # For example, a box over California:
    bbox = (-124.5, 32.5, -114.0, 42.0) 
    
    # Define the time window for the fire event (YYYY-MM-DD)
    date_start = "2024-08-01" 
    date_end = "2024-08-02"

    print(f"Searching for {short_name} granules...")
    
    # 2. Query the NASA CMR (Common Metadata Repository)
    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox,
        temporal=(date_start, date_end)
    )
    
    print(f"Found {len(results)} granules matching your criteria.")

    # 3. Download the files
    if len(results) > 0:
        # Create a directory to hold the data
        os.makedirs("tempo_data", exist_ok=True)
        
        print("Downloading files...")
        # earthaccess automatically handles concurrent downloads
        downloaded_files = earthaccess.download(results, local_path="tempo_data")
        print(f"Successfully downloaded {len(downloaded_files)} files to the 'tempo_data' folder.")
    else:
        print("No data found for this time and location.")

if __name__ == "__main__":
    download_wildfire_data()