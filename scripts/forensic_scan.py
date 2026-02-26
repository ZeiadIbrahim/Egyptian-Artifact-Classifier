import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil

# --- UPDATED PATHS ---
MODEL_PATH = r"C:\Users\zeiad\Desktop\Egypt_Artifact_Project\models\clean_model.pth"  
ALL_IMAGES_DIR = r"C:\Users\zeiad\Desktop\Egypt_Artifact_Project\dataset\images"
PROCESSED_DIR = r"C:\Users\zeiad\Desktop\Egypt_Artifact_Project\dataset\processed"
OUTPUT_DIR = r"C:\Users\zeiad\Desktop\Egypt_Artifact_Project\forensic_results"
DATA_DIR = os.path.join(PROCESSED_DIR, "train")

CONFIDENCE_THRESHOLD = 0.90  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_classes():
    if not os.path.exists(DATA_DIR):
        return ["Jewellery", "Pottery", "Reliefs", "Statuary"]
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

def load_model(num_classes):
    print(f"Loading {MODEL_PATH}...")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    return model

def scan_unclassified():
    classes = get_classes()
    model = load_model(len(classes))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Figure out which images are ALREADY classified
    known_images = set()
    for root, dirs, files in os.walk(PROCESSED_DIR):
        for file in files:
            known_images.add(file)
            
    # 2. Get all images from the main folder
    all_images = [f for f in os.listdir(ALL_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 3. The "Unclassified" ones are the ones left over
    unclassified_images = [img for img in all_images if img not in known_images]

    print(f"\nFound {len(known_images)} images already sorted in 'processed'.")
    print(f"Found {len(unclassified_images)} 'Unclassified' images in the main folder.")
    
    if len(unclassified_images) == 0:
        print("No unclassified images found to scan!")
        return

    # Create subfolders for sorting
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "Still_Unclassified"), exist_ok=True)

    print(f"\nScanning the {len(unclassified_images)} Unclassified images...")
    rescued_count = 0

    with torch.no_grad():
        for img_name in unclassified_images:
            img_path = os.path.join(ALL_IMAGES_DIR, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_catid = torch.topk(probabilities, 1)
                
                confidence = top_prob[0].item()
                predicted_class = classes[top_catid[0]]

                if confidence >= CONFIDENCE_THRESHOLD:
                    shutil.copy(img_path, os.path.join(OUTPUT_DIR, predicted_class, img_name))
                    rescued_count += 1
                else:
                    shutil.copy(img_path, os.path.join(OUTPUT_DIR, "Still_Unclassified", img_name))
            
            except Exception as e:
                pass

    print("\n" + "="*30)
    print("SCAN COMPLETE")
    print(f"Artifacts Rescued (>90%): {rescued_count}")
    print(f"Check results in: {OUTPUT_DIR}")
    print("="*30)

if __name__ == "__main__":
    scan_unclassified()