import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = os.path.join("dataset", "processed")
MODEL_SAVE_PATH = "dirty_model.pth"
BATCH_SIZE = 16          # Reduced batch size for better generalization
LEARNING_RATE = 0.0001   # Low learning rate for fine-tuning
EPOCHS = 10              # Enough time to converge
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders():
    """
    Creates the training and validation data loaders with augmentation.
    """
    # Strong Augmentation for Training
    # This prevents the AI from seeing the exact same image twice
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load images from folders
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names

def build_model(num_classes):
    """
    Downloads ResNet50 and adjusts it for our specific classes.
    """
    print("Downloading ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # --- FINE TUNING STRATEGY ---
    # 1. Freeze the early layers (keep the basic knowledge)
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. UNFREEZE the last block (Layer 4)
    # This allows the AI to adapt its high-level understanding to Egyptian Art
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # 3. Replace the Head (Output Layer)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5), # Drops 50% of connections randomly to prevent memorization
        nn.Linear(num_ftrs, num_classes)
    )
    
    model = model.to(DEVICE)
    return model

def train_model():
    dataloaders, dataset_sizes, class_names = get_data_loaders()
    actual_num_classes = len(class_names)
    print(f"Classes found: {class_names}")
    
    model = build_model(actual_num_classes)
    
    criterion = nn.CrossEntropyLoss()
    
    # --- OPTIMIZER WITH REGULARIZATION ---
    # weight_decay=1e-4 acts as the "Handcuffs" to stop overfitting
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=LEARNING_RATE, 
                          weight_decay=1e-4)
    
    # Scheduler: Slows down learning rate every 7 epochs so it doesn't overshoot
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print(f"\n--- Starting Training on {DEVICE} (10 Epochs) ---")
    print("Optimization: Adam with Weight Decay (L2 Regularization)")
    
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} Phase"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + Optimize only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the new best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"--> New Record! Model Saved (Acc: {best_acc:.4f})")

        print()

    print(f'Final Best val Acc: {best_acc:4f}')
    print("Training Complete.")

if __name__ == "__main__":
    train_model()