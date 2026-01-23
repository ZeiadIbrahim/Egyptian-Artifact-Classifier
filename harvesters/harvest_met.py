import os
import requests
import time
import pandas as pd
from tqdm import tqdm
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION ---
IMAGE_DIR = os.path.join("dataset", "images")
CSV_DIR = os.path.join("dataset", "csv_files")
INDIVIDUAL_CSV = os.path.join(CSV_DIR, "MET_data.csv")
MASTER_CSV = os.path.join(CSV_DIR, "master_data.csv")

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
DEPT_ID_EGYPT = 10 
TARGET_COUNT = 1020 

COLUMNS = ["ID", "Source", "Title", "ObjectName", "Medium", "Period", "Classification", "ImageURL"]

def create_session():
    """Creates a session that automatically retries failed connections."""
    session = requests.Session()
    # Retry 3 times, waiting longer between each fail (backoff_factor=1)
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def update_master_csv(new_df, source_filename):
    new_df = new_df.copy()
    new_df['original_source_file'] = source_filename
    
    if os.path.exists(MASTER_CSV) and os.path.getsize(MASTER_CSV) > 0:
        try:
            master_df = pd.read_csv(MASTER_CSV)
            combined_df = pd.concat([master_df, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            combined_df = new_df
    else:
        combined_df = new_df
        
    combined_df.drop_duplicates(subset=['ID'], keep='last', inplace=True)
    combined_df.to_csv(MASTER_CSV, index=False)

def run_harvest():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    
    session = create_session()
    records = []
    
    print(f"--- Starting Met Harvest ---")
    
    # 1. Fetch Object IDs
    try:
        resp = session.get(f"{BASE_URL}/objects?departmentIds={DEPT_ID_EGYPT}", timeout=20)
        object_ids = resp.json().get('objectIDs', [])
    except Exception as e:
        print(f"Critical Error connecting to API: {e}")
        return

    print(f"Found {len(object_ids)} objects. Downloading {TARGET_COUNT}...")
    
    downloaded = 0
    pbar = tqdm(total=TARGET_COUNT, desc="Harvesting")

    for obj_id in object_ids:
        if downloaded >= TARGET_COUNT: break
        
        # Wait 0.5 seconds to avoid bans
        time.sleep(0.5)
        
        unique_id = f"MET_{obj_id}"
        filename = f"{unique_id}.jpg"
        save_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            # Metadata Fetch
            meta_resp = session.get(f"{BASE_URL}/objects/{obj_id}", timeout=10)
            if meta_resp.status_code != 200: continue
            data = meta_resp.json()

            if not data.get('isPublicDomain'): continue
            img_url = data.get('primaryImageSmall')
            if not img_url: continue

            # Download Image (Only if missing)
            if not os.path.exists(save_path):
                img_resp = session.get(img_url, timeout=10)
                if img_resp.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(img_resp.content)
                    try:
                        with Image.open(save_path) as img: img.verify()
                    except:
                        os.remove(save_path)
                        continue
                else:
                    continue

            # Log Record
            records.append({
                "ID": unique_id,
                "Source": "Met Museum",
                "Title": data.get('title', 'Unknown'),
                "ObjectName": data.get('objectName', 'Unknown'),
                "Medium": data.get('medium', 'Unknown'),
                "Period": data.get('period', 'Unknown'),
                "Classification": "Unknown",
                "ImageURL": img_url
            })
            
            downloaded += 1
            pbar.update(1)

        except Exception as e:
            print(f" [!] Skipped {obj_id}: {e}")
            continue

    pbar.close()

    if records:
        df = pd.DataFrame(records)
        for col in COLUMNS:
            if col not in df.columns: df[col] = "Unknown"
            
        df[COLUMNS].to_csv(INDIVIDUAL_CSV, index=False)
        print(f"Saved {len(df)} records to {INDIVIDUAL_CSV}")
        update_master_csv(df[COLUMNS], "MET_data.csv")
        print(f"Master CSV updated.")

if __name__ == "__main__":
    run_harvest()