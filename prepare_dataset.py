'''
-I created a "factory" script that automatically processes every raw image in my dataset to prepare it for AI training. 
 Instead of feeding the AI raw, messy files, this script acts as an assembly line that standardizes every single image.

Why I Did It:

-I Standardized the Size: Neural networks require every input image to be the exact same size. 
 Since my raw images ranged from tiny thumbnails to massive high-resolution scans, I programmed the script to resize everything to a standard 224x224 pixels.

-I Protected the Aspect Ratio: Simply squashing a tall statue into a square box would distort it (making it look "fat" or stretched), which would confuse the AI. 
 I used a technique called "padding" to add black bars to the sides of the images. 
 This keeps the artifact's original shape perfect while still fitting it into the required square format.

-I Created Study Sets: To train the AI properly, I couldn't just give it all the answers. 
 I programmed the script to randomly split the data into three separate folders:

 *Training (80%): For the AI to study.

 *Validation (10%): For the AI to practice on during training.

 *Testing (10%): For the AI to assess its performance after training.

The Result:
-This transformed a folder of random, disorganized files into a structured, professional dataset where every image is uniform, clean, and ready for the neural network.
'''

import os
import pandas as pd
import random
from PIL import Image, ImageOps
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_DIR = os.path.join("dataset", "csv_files")
MASTER_CSV = os.path.join(CSV_DIR, "master_data.csv")
SOURCE_IMG_DIR = os.path.join("dataset", "images")
OUTPUT_DIR = os.path.join("dataset", "processed")

# Image Settings (Standard for ResNet)
IMG_SIZE = (224, 224) 
CLASSES = ["Jewellery", "Pottery", "Statuary", "Reliefs"]

# Split Ratios (80% Train, 10% Validation, 10% Test)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test is the remainder

def prepare_dataset():
    print("--- Starting The Factory (Preprocessing) ---")
    
    # 1. Load Data
    if not os.path.exists(MASTER_CSV):
        print("Error: Master CSV not found.")
        return
    df = pd.read_csv(MASTER_CSV)
    
    # 2. Filter: Only keep the 4 Main Classes (Drop Unclassified)
    df = df[df['Classification'].isin(CLASSES)]
    print(f"Processing {len(df)} valid images (dropping Unclassified)...")
    
    # 3. Create Output Structure
    # dataset/processed/train/Jewellery, dataset/processed/val/Jewellery, etc.
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
            
    # 4. Processing Loop
    success_count = 0
    skip_count = 0
    
    # Shuffle data to ensure random distribution
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    pbar = tqdm(total=len(df), desc="Resizing & Sorting")
    
    for _, row in df.iterrows():
        img_id = str(row['ID']).strip()
        cls = row['Classification']
        src_path = os.path.join(SOURCE_IMG_DIR, f"{img_id}.jpg")
        
        if not os.path.exists(src_path):
            skip_count += 1
            pbar.update(1)
            continue
            
        # Determine Split (Train/Val/Test)
        # We use a random float to decide where it goes
        rand = random.random()
        if rand < TRAIN_RATIO:
            split = 'train'
        elif rand < (TRAIN_RATIO + VAL_RATIO):
            split = 'val'
        else:
            split = 'test'
            
        dest_path = os.path.join(OUTPUT_DIR, split, cls, f"{img_id}.jpg")
        
        try:
            # Open and Resize
            with Image.open(src_path) as img:
                # Convert to RGB (fixes black/white or transparent images)
                img = img.convert('RGB')
                
                # Resize with padding (maintains aspect ratio, adds black bars if needed)
                # This prevents "stretching" artifacts which confuses AI
                img = ImageOps.pad(img, IMG_SIZE, color='black')
                
                img.save(dest_path, quality=90)
                success_count += 1
        except Exception as e:
            # If image is corrupt, just skip
            print(f"Error processing {img_id}: {e}")
            skip_count += 1
            
        pbar.update(1)
        
    pbar.close()
    
    print(f"\n--- Factory Shutdown ---")
    print(f"Successfully Processed: {success_count}")
    print(f"Skipped/Missing:        {skip_count}")
    print(f"Output Location:        {OUTPUT_DIR}")
    print("Ready for AI Training!")

if __name__ == "__main__":
    prepare_dataset()