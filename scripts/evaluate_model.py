import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = os.path.join("dataset", "processed")
MODEL_PATH = "clean_model.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    """
    Loads the Validation data only.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # We use the 'val' folder for testing the model
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return val_loader, val_dataset.classes

def load_model(path, num_classes):
    """
    Loads the saved ResNet50 model.
    """
    print(f"Loading model from {path}...")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    # Re-create the same head structure we used in training
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    # Load the weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        
    model = model.to(DEVICE)
    model.eval() # Set to evaluation mode (no learning, just predicting)
    return model

def evaluate():
    val_loader, class_names = load_data()
    print(f"Classes found: {class_names}")
    
    # Load the model matching the number of classes
    model = load_model(MODEL_PATH, len(class_names))
    
    all_preds = []
    all_labels = []
    
    print("Running Evaluation... (This takes a moment)")
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # --- Generate Report ---
    print("\n" + "="*30)
    print("FINAL CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # --- Generate Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {MODEL_PATH}')
    
    # Save the plot
    output_filename = f"confusion_matrix_{len(class_names)}_classes.png"
    plt.savefig(output_filename)
    print(f"\nGraph saved as '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    evaluate()