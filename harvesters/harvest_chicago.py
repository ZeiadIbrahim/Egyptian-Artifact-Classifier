import os
import requests
import time
import pandas as pd
import random
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGE_DIR = os.path.join("dataset", "images")
CSV_DIR = os.path.join("dataset", "csv_files")
INDIVIDUAL_CSV = os.path.join(CSV_DIR, "CHI_data.csv")
MASTER_CSV = os.path.join(CSV_DIR, "master_data.csv")

BASE_URL = "https://api.artic.edu/api/v1/artworks/search"
IMAGE_BASE_URL = "https://www.artic.edu/iiif/2"
IMAGES_PER_CLASS = 150 

SEARCH_MAPPING = {
    "Jewellery": "Ancient Egypt Amulet",       
    "Statuary": "Ancient Egypt Statue",
    "Pottery": "Ancient Egypt Vessel",         
    "Reliefs": "Ancient Egypt Relief"
}

COLUMNS = ["ID", "Source", "Title", "ObjectName", "Medium", "Period", "Classification", "ImageURL"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.artic.edu/"
}

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

def get_image_url(image_id):
    return f"{IMAGE_BASE_URL}/{image_id}/full/600,/0/default.jpg" if image_id else None

def run_harvest():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    
    session = requests.Session()
    session.headers.update(HEADERS)
    records = []
    
    print(f"--- Starting Chicago Harvest ---")
    
    for category, query in SEARCH_MAPPING.items():
        print(f"Processing: {category}...")
        count = 0
        page = 1
        
        # We assume about 3 pages needed to hit 150, looping safely
        while count < IMAGES_PER_CLASS:
            params = {
                "q": query,
                "query[term][is_public_domain]": "true",
                "fields": "id,title,image_id,medium_display,date_display,artwork_type_title",
                "limit": 50, "page": page
            }
            
            try:
                resp = session.get(BASE_URL, params=params)
                if resp.status_code != 200: break
                items = resp.json().get('data', [])
                if not items: break

                for item in items:
                    if count >= IMAGES_PER_CLASS: break
                    img_id = item.get('image_id')
                    obj_id = item.get('id')
                    if not img_id: continue

                    unique_id = f"CHI_{obj_id}"
                    filename = f"{unique_id}.jpg"
                    save_path = os.path.join(IMAGE_DIR, filename)

                    # Download
                    img_url = get_image_url(img_id)
                    if not os.path.exists(save_path):
                        try:
                            img_resp = session.get(img_url, timeout=10)
                            if img_resp.status_code == 200:
                                with open(save_path, 'wb') as f: f.write(img_resp.content)
                            else: continue
                        except: continue
                    
                    records.append({
                        "ID": unique_id,
                        "Source": "Art Institute of Chicago",
                        "Title": item.get('title', 'Unknown'),
                        "ObjectName": item.get('artwork_type_title', 'Unknown'),
                        "Medium": item.get('medium_display', 'Unknown'),
                        "Period": item.get('date_display', 'Unknown'),
                        "Classification": category, # Explicit Class
                        "ImageURL": img_url
                    })
                    count += 1
                    time.sleep(random.uniform(0.2, 0.5))

                page += 1
            except Exception:
                break

    if records:
        df = pd.DataFrame(records)
        for col in COLUMNS:
            if col not in df.columns: df[col] = "Unknown"
        df[COLUMNS].to_csv(INDIVIDUAL_CSV, index=False)
        update_master_csv(df[COLUMNS], "CHI_data.csv")
        print(f"Success. Master CSV Updated.")

if __name__ == "__main__":
    run_harvest()