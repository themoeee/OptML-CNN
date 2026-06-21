import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional, Callable
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as T
import copy

import sys
from pathlib import Path


TASK_1_DIR = Path(__file__).resolve().parents[1] / "task_1_hyperparameter"
sys.path.append(str(TASK_1_DIR))
from model_task1 import SimpleCNN
from model_task1 import ImageClassificationDataset




def load_dataset(category: str, split: str) -> TensorDataset:
    """Load a dataset split as a PyTorch TensorDataset."""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]      #had to fix this for laptop
    target_path = PROJECT_ROOT / "data" / "processed" / category / f"{split}.npz"        
    data = np.load(target_path)
    images = torch.from_numpy(data["images"])
    labels = torch.from_numpy(data["labels"])
    return TensorDataset(images, labels)

def train_model(category, batch_size, data_augmentation, num_epochs, learning_rate, kernel_size, optimizer_name, momentum):
    # Training setup
    print("="*60 + "\n TRAINING MODEL")

    #category = "NT"                                                                        # SPECIFIED CATEGORY HERE
    category = category
    batch_size = batch_size
    data_augmentation = data_augmentation                                            # TOGGLE DATA AUGMENTATION ON/OFF   
    num_epochs = num_epochs
    learning_rate =learning_rate
    kernel_size = kernel_size                                               # I know that this block is useless per se and redundant, just helps me keep track of stuff 
    optimizer_name = optimizer_name
    momentum = momentum

    # Define augmentation transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    print('LOADING DATA')
    data_path = Path(__file__).resolve().parents[2]  / "data" / "processed" / category / "train.npz"         # CHOOSING {category} SAMPLES FOR CNN
    #data_path = Path(f"C:/Code/Opt-ML/Project2/data/processed/{category}/train.npz")          # CHOOSING {category} SAMPLES FOR CNN 
    data = np.load(data_path)

    # Access images and labels
    images = data["images"]
    labels = data["labels"]

    unique, counts = np.unique(labels, return_counts=True)                          
    print("Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({100*c/len(labels):.1f}%)")

    splits = ["train", "val", "test"]

    print("Dataset Summary:")
    print("=" * 60)
        
    for split in splits:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]      #had to fix this for laptop
        target_path = PROJECT_ROOT / "data" / "processed" / category / f"{split}.npz"        
        data = np.load(target_path)             #HERE ALSO PATH SPECIFIED FOR {category}
        imgs, lbls = data["images"], data["labels"]
        class_dist = dict(zip(*np.unique(lbls, return_counts=True)))
        print(f"  {split:5s}: {len(lbls):4d} samples | Class 0: {class_dist.get(0, 0):4d}, Class 1: {class_dist.get(1, 0):4d}")

   # run_times = {}

   # run_start = time.perf_counter()
    print(f"\n STARTING RUN WITH DATA AUGMENTATION: {data_augmentation} \n")
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print("GPU name:         ", torch.cuda.get_device_name(0))

    model = SimpleCNN(kernel_size)                                                          # Create model
    print(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    elif optimizer_name == "SGD":  
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    train_dataset = ImageClassificationDataset(category=category,
        split="train",
        transform=train_transform if data_augmentation else None
    )
    val_dataset = load_dataset(f"{category}", "val")
    test_dataset = load_dataset(f"{category}", "test")


    # DATA LOADER DOES THE BATCHING AND SHUFFLING FOR ME
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # Test iteration
    # for batch_images, batch_labels in train_loader:
    #     print(f"Batch images shape: {batch_images.shape}")  # (32, 1, 128, 128)
    #     print(f"Batch labels shape: {batch_labels.shape}")  # (32,)
    #     print(f"Image dtype: {batch_images.dtype}")
    #     print(f"Label dtype: {batch_labels.dtype}")
    #     break

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()           # RESETS ALL GRADIENTS TO ZERO
            outputs = model(images)         # SEND IMAGES INTO MODEL (INPUTS) AND GET OUTPUTS (LOGITS)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss_epoch =val_loss/len(val_loader)
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_state = copy.deepcopy(model.state_dict())                         # Save the best model state

        # Print metrics
        train_acc = 100 * train_correct / len(train_dataset)
        val_acc = 100 * val_correct / len(val_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))


    print(f"\nFinished run)\n")

    model.load_state_dict(best_model_state)

    return best_val_loss, model



    
