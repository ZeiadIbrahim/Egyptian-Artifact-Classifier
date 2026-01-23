import os
import requests
import time
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGE_DIR = os.path.join("dataset", "images")
CSV_DIR = os.path.join("dataset", "csv_files")
INDIVIDUAL_CSV = os.path.join(CSV_DIR, "CLE_data.csv")
MASTER_CSV = os.path.join(CSV_DIR, "master_data.csv")

BASE_URL = "https://openaccess-api.clevelandart.org/api/artworks"
TARGET_COUNT = 1000

COLUMNS = ["ID", "Source", "Title", "ObjectName", "Medium", "Period", "Classification", "ImageURL"]

def update_master_csv(new_df, source_filename):
    """Appends new data to the Master CSV with a source tracker. Handles empty files safely."""
    new_df = new_df.copy()
    new_df['original_source_file'] = source_filename
    
    # Check if file exists AND has data (size > 0)
    if os.path.exists(MASTER_CSV) and os.path.getsize(MASTER_CSV) > 0:
        try:
            master_df = pd.read_csv(MASTER_CSV)
            combined_df = pd.concat([master_df, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            # File exists but is empty -> Treat as new
            combined_df = new_df
    else:
        # File doesn't exist -> Create new
        combined_df = new_df
        
    combined_df.drop_duplicates(subset=['ID'], keep='last', inplace=True)
    combined_df.to_csv(MASTER_CSV, index=False)

def run_harvest():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    
    records = []
    print(f"--- Starting Cleveland Harvest ---")
    
    downloaded = 0
    skip = 0
    limit = 100
    pbar = tqdm(total=TARGET_COUNT, desc="Harvesting")

    while downloaded < TARGET_COUNT:
        params = {'q': 'Egypt', 'cc0': '1', 'has_image': '1', 'limit': limit, 'skip': skip}

        try:
            resp = requests.get(BASE_URL, params=params)
            if resp.status_code != 200: break
            data = resp.json().get('data', [])
            if not data: break

            for item in data:
                if downloaded >= TARGET_COUNT: break
                if 'images' not in item or 'web' not in item['images']: continue

                obj_id = item['id']
                unique_id = f"CLE_{obj_id}"
                filename = f"{unique_id}.jpg"
                save_path = os.path.join(IMAGE_DIR, filename)

                img_url = item['images']['web']['url']

                if not os.path.exists(save_path):
                    try:
                        img_resp = requests.get(img_url, timeout=10)
                        if img_resp.status_code == 200:
                            with open(save_path, 'wb') as f: f.write(img_resp.content)
                        else: continue
                    except: continue

                records.append({
                    "ID": unique_id,
                    "Source": "Cleveland Museum of Art",
                    "Title": item.get('title', 'Unknown'),
                    "ObjectName": item.get('type', 'Unknown'),
                    "Medium": item.get('technique', 'Unknown'),
                    "Period": item.get('creation_date', 'Unknown'),
                    "Classification": "Unknown",
                    "ImageURL": img_url
                })
                downloaded += 1
                pbar.update(1)
            
            skip += limit
            time.sleep(0.2)
        except Exception:
            break
            
    pbar.close()

    if records:
        df = pd.DataFrame(records)
        for col in COLUMNS:
            if col not in df.columns: df[col] = "Unknown"
        df[COLUMNS].to_csv(INDIVIDUAL_CSV, index=False)
        update_master_csv(df[COLUMNS], "CLE_data.csv")
        print(f"Success. Master CSV Updated.")

if __name__ == "__main__":
    run_harvest()