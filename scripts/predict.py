import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- CONFIGURATION ---
MODEL_PATH = "clean_model.pth"  # Ensure this points to your clean model
IMAGE_PATH = r"C:\Users\zeiad\Desktop\Egypt_Artifact_Project\statue1.jpg"
DATA_DIR = os.path.join("dataset", "processed", "train") # We read class names from here

# --- DEVICE CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_classes():
    """
    Automatically reads the folder names to get the correct alphabetical order.
    """
    if not os.path.exists(DATA_DIR):
        print(f"Error: Could not find dataset at {DATA_DIR}")
        # Fallback to manual list if folder is missing (ALPHABETICAL ORDER!)
        return ["Jewellery", "Pottery", "Reliefs", "Statuary"]
        
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Detected Classes (Alphabetical): {classes}")
    return classes

def load_model(num_classes):
    print(f"Loading {MODEL_PATH}...")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Load weights
    if torch.cuda.is_available():
        state_dict = torch.load(MODEL_PATH)
    else:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_image():
    # 1. Get the correct classes
    classes = get_classes()
    
    # 2. Prepare the image (Resize to see the whole object)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(IMAGE_PATH).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Could not find image at {IMAGE_PATH}")
        return

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 3. Load Model
    model = load_model(len(classes))

    # 4. Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get Top 1 Prediction
        top_prob, top_catid = torch.topk(probabilities, 1)
        
        print("\n" + "="*30)
        print(f"PREDICTION: {classes[top_catid[0]]}")
        print(f"CONFIDENCE: {top_prob[0].item() * 100:.2f}%")
        print("="*30)

        # Show all probabilities
        print("\nFull Breakdown:")
        for i, prob in enumerate(probabilities):
            print(f"{classes[i]}: {prob.item() * 100:.2f}%")

if __name__ == "__main__":
    predict_image()