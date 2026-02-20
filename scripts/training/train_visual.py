import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import sys
import os
import time
from tqdm import tqdm

# Define project_dir at the top
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src to path
sys.path.append(os.path.join(project_dir, "backend", "src"))

from config.settings import *
from mers.core.visual_model import VisualEmotionModel

def train_visual_model(data_dir, epochs=25, batch_size=32, learning_rate=0.0001):
    print(f"Training Visual Model on {DEVICE}")
    print(f"Data Source: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        print("Please run extract_frames.py first to generate the dataset.")
        return

    # Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
    }

    # Check for pre-split structure (train/val folders)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    has_split = os.path.exists(train_dir) and os.path.exists(val_dir)

    if has_split:
        print(f"Detected pre-split dataset structure at {data_dir}")
        train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
        val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
        class_names = train_dataset.classes
    else:
        print(f"Using single directory structure. Splitting 80/20.")
        full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
        class_names = full_dataset.classes
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Classes found: {class_names}")

    # Label Mapping (Local Index -> Global MERS Index)
    # Ensure that the dataset labels match the model's expected output neurons (EMOTIONS list)
    target_emotions = EMOTIONS # ['Angry', 'Happy', 'Sad', ...]
    
    local_to_global = {}
    for idx, cls_name in enumerate(class_names):
        name = cls_name.capitalize()
        if name in target_emotions:
            local_to_global[idx] = target_emotions.index(name)
        else:
            print(f"Warning: Class '{cls_name}' not in MERS config. Ignored.")
            local_to_global[idx] = -1 # Should filter these out if possible

    print(f"Label Mapping (Dataset Index -> Model Index): {local_to_global}")

    # Apply target transform to map labels correctly
    # Note: random_split datasets don't have target_transform attribute directly, 
    # they wrap the underlying dataset.
    
    target_transform_func = lambda y: local_to_global.get(y, y)

    if has_split:
        train_dataset.target_transform = target_transform_func
        val_dataset.target_transform = target_transform_func
    else:
        # For random_split, we need to apply it to the underlying dataset
        # But both splits share the same underlying dataset, so this works for both.
        full_dataset.target_transform = target_transform_func

    # DataLoaders
    # num_workers=0 for Windows stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    # Model
    model = VisualEmotionModel(NUM_CLASSES).to(DEVICE)
    
    # If finetuning, maybe load pretrained visual_model.pth if exists?
    # if os.path.exists(VISUAL_MODEL_PATH):
    #     model.load_state_dict(torch.load(VISUAL_MODEL_PATH))
    
    # Ignore index -1 for unknown classes
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # Fine-tuning: lower LR
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(DEVICE)
            targets = labels.to(DEVICE)
            
            # Filter out ignored classes (-1) if any
            # Although ignore_index handles loss, we shouldn't count them in accuracy
            valid_mask = targets != -1
            if not valid_mask.any():
                continue
                
            # If we want to strictly filter inputs for the forward pass:
            # inputs = inputs[valid_mask]
            # targets = targets[valid_mask]
            # But CrossEntropyLoss(ignore_index=-1) handles it for loss.
            # For accuracy, we need to be careful.
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            # Accuracy calc (only on valid targets)
            valid_targets = targets[valid_mask]
            valid_predicted = predicted[valid_mask]
            total += valid_targets.size(0)
            correct += (valid_predicted == valid_targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_acc = 100 * correct / total if total > 0 else 0
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                targets = labels.to(DEVICE)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # Filter valid
                valid_mask = targets != -1
                if valid_mask.any():
                    valid_targets = targets[valid_mask]
                    valid_predicted = predicted[valid_mask]
                    val_total += valid_targets.size(0)
                    val_correct += (valid_predicted == valid_targets).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        print(f"Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), VISUAL_MODEL_PATH)
            print(f"Saved Best Model to {VISUAL_MODEL_PATH}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Check default location
        visual_data = os.path.join(project_dir, "datasets", "Images") # Updated default
        
        if os.path.exists(visual_data):
            train_visual_model(visual_data)
        else:
            # Fallback to older default
            visual_data_old = os.path.join(project_dir, "datasets", "visual_data")
            if os.path.exists(visual_data_old):
                train_visual_model(visual_data_old)
            else:
                print("Usage: python train_visual.py <path_to_visual_image_root>")
                print(f"Default path checked not found: {visual_data}")
    else:
        train_visual_model(sys.argv[1])
