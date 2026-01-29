import os
import pandas as pd
import random
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_DIR = os.path.join("dataset", "csv_files")
MASTER_CSV = os.path.join(CSV_DIR, "master_data.csv")
SOURCE_IMG_DIR = os.path.join("dataset", "images")
OUTPUT_DIR = os.path.join("dataset", "processed")

# Image Settings
IMG_SIZE = (224, 224) 
CLASSES = ["Jewellery", "Pottery", "Statuary", "Reliefs", "Unclassified"]

# Augmentation Settings
TARGET_TRAIN_COUNT = 1000
CLASSES_TO_AUGMENT = ["Jewellery", "Pottery", "Statuary", "Reliefs"] # Skip Unclassified

# Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test is remainder

def augment_image(img):
    """Generates a variation of the input image."""
    # 1. Flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    # 2. Rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False)
    # 3. Brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    # 4. Zoom/Crop
    if random.random() > 0.5:
        w, h = img.size
        crop_val = random.uniform(0.0, 0.1) 
        border = (w * crop_val, h * crop_val, w * (1 - crop_val), h * (1 - crop_val))
        img = img.crop(border).resize((w, h), Image.Resampling.LANCZOS)
    return img

def step_1_process_and_split():
    print("--- STEP 1: Processing & Splitting ---")
    
    if not os.path.exists(MASTER_CSV):
        print("Error: Master CSV not found.")
        return

    df = pd.read_csv(MASTER_CSV)
    df = df[df['Classification'].isin(CLASSES)]
    
    # Create Folders
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
            
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    pbar = tqdm(total=len(df), desc="Splitting")
    
    for _, row in df.iterrows():
        img_id = str(row['ID']).strip()
        cls = row['Classification']
        src_path = os.path.join(SOURCE_IMG_DIR, f"{img_id}.jpg")
        
        if not os.path.exists(src_path):
            pbar.update(1)
            continue
            
        # Determine Split
        rand = random.random()
        if rand < TRAIN_RATIO:
            split = 'train'
        elif rand < (TRAIN_RATIO + VAL_RATIO):
            split = 'val'
        else:
            split = 'test'
            
        dest_path = os.path.join(OUTPUT_DIR, split, cls, f"{img_id}.jpg")
        
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                img = ImageOps.pad(img, IMG_SIZE, color='black')
                img.save(dest_path, quality=90)
        except:
            pass
            
        pbar.update(1)
    pbar.close()

def step_2_augment_training():
    print("\n--- STEP 2: Balancing Training Set ---")
    train_dir = os.path.join(OUTPUT_DIR, 'train')
    
    for cls in CLASSES_TO_AUGMENT:
        cls_path = os.path.join(train_dir, cls)
        
        # Get list of existing real images
        images = [f for f in os.listdir(cls_path) if f.endswith('.jpg')]
        count = len(images)
        
        if count >= TARGET_TRAIN_COUNT:
            print(f"[{cls}] {count} images. Enough.")
            continue
            
        needed = TARGET_TRAIN_COUNT - count
        print(f"[{cls}] {count} images. Generating {needed} more...")
        
        pbar = tqdm(total=needed, desc=f"Augmenting {cls}")
        generated = 0
        
        while generated < needed:
            # Pick random real image
            src = random.choice(images) 
            src_path = os.path.join(cls_path, src)
            
            try:
                with Image.open(src_path) as img:
                    img = img.convert('RGB')
                    aug_img = augment_image(img)
                    
                    save_name = f"aug_{generated}_{src}"
                    aug_img.save(os.path.join(cls_path, save_name), quality=90)
                    generated += 1
                    pbar.update(1)
            except:
                pass
        pbar.close()

if __name__ == "__main__":
    step_1_process_and_split()
    step_2_augment_training()
    print("\n--- DATASET READY FOR TRAINING ---")