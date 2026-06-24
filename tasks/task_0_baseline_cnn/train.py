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

from model import SimpleCNN
from model import ImageClassificationDataset

from pathlib import Path



def load_dataset(category: str, split: str) -> TensorDataset:
    """Load a dataset split as a PyTorch TensorDataset."""
    #data = np.load(f"C:/Code/Opt-ML/Project2/data/processed/{category}/{split}.npz")      
    PROJECT_ROOT = Path(__file__).resolve().parents[2]      #had to fix this for laptop
    target_path = PROJECT_ROOT / "data" / "processed" / category / f"{split}.npz"                                 # PATH TO DATASET SPLITS
    data = np.load(f"{target_path}")
    images = torch.from_numpy(data["images"])
    labels = torch.from_numpy(data["labels"])
    return TensorDataset(images, labels)



if __name__ == "__main__":
    
    category = "NT"                                                                           # SPECIFIED CATEGORY HERE
    BATCH_SIZE = 32
    #DATA_AUGMENTATION = True
    data_augmentation_options = [False, True]                                                   # TOGGLE DATA AUGMENTATION ON/OFF   
    NUM_EPOCHS = 150
    learning_rate = 1e-3

    # Define augmentation transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    print('LOADING DATA')
    data_path = Path(__file__).resolve().parents[2]  / "data" / "processed" / category / "train.npz"         # CHOOSING {category} SAMPLES FOR CNN 
    data = np.load(data_path)

    # Access images and labels
    images = data["images"]
    labels = data["labels"]

    print('DATA SUMMARY')                                                                            # SANITY CHECKS
    print(f"Images shape: {images.shape}")  # (N, 1, 128, 128)
    print(f"Labels shape: {labels.shape}")  # (N,)
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Pixel value range: [{images.min():.2f}, {images.max():.2f}]")

    unique, counts = np.unique(labels, return_counts=True)                          
    print("Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({100*c/len(labels):.1f}%)")

    splits = ["train", "val", "test"]

    print("Dataset Summary:")
    print("=" * 60)
        
    for split in splits:
        data = np.load(Path(__file__).resolve().parents[2] / "data" / "processed" / category / f"{split}.npz")             #HERE ALSO PATH SPECIFIED FOR {category}
        imgs, lbls = data["images"], data["labels"]
        class_dist = dict(zip(*np.unique(lbls, return_counts=True)))
        print(f"  {split:5s}: {len(lbls):4d} samples | Class 0: {class_dist.get(0, 0):4d}, Class 1: {class_dist.get(1, 0):4d}")

    run_times = {}

    for DATA_AUGMENTATION in data_augmentation_options:
        run_start = time.perf_counter()
        print(f"\n STARTING RUN WITH DATA AUGMENTATION: {DATA_AUGMENTATION} \n")
        # Training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print("GPU name:         ", torch.cuda.get_device_name(0))

        model = SimpleCNN()                                                          # Create model
        print(model)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)    

        train_dataset = ImageClassificationDataset(category=category,
            split="train",
            transform=train_transform if DATA_AUGMENTATION else None
        )
        val_dataset = load_dataset(f"{category}", "val")
        test_dataset = load_dataset(f"{category}", "test")


        # DATA LOADER DOES THE BATCHING AND SHUFFLING FOR ME
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Test iteration
        for batch_images, batch_labels in train_loader:
            print(f"Batch images shape: {batch_images.shape}")  # (32, 1, 128, 128)
            print(f"Batch labels shape: {batch_labels.shape}")  # (32,)
            print(f"Image dtype: {batch_images.dtype}")
            print(f"Label dtype: {batch_labels.dtype}")
            break

        train_losses = []
        val_losses = []

        for epoch in range(NUM_EPOCHS):
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

            # Print metrics
            train_acc = 100 * train_correct / len(train_dataset)
            val_acc = 100 * val_correct / len(val_dataset)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")

            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))

        model.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_correct += (outputs.argmax(1) == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100 * test_correct / len(test_dataset)

        print(f"  Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        run_duration = time.perf_counter() - run_start
        run_times[DATA_AUGMENTATION] = run_duration
        minutes = int(run_duration // 60)
        seconds = run_duration % 60

        # Plot training curves
        plt.figure(figsize=(10, 5)) 
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Curves for {category} Category, Baseline CNN, Data augmentation: {DATA_AUGMENTATION}")
        plt.legend()
        plt.grid()
        results_dir = Path("./results")
        results_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(results_dir / f"training_curves_{category}_baseline_cnn_aug{DATA_AUGMENTATION}.png")
        plt.close()

        model_path = results_dir / f"model_{category}_baseline_cnn_aug{DATA_AUGMENTATION}.pt"           # SAVE MODELS AFTER TRAINING
        torch.save({
            "model_state_dict": model.state_dict(),
            "category": category,
            "data_augmentation": DATA_AUGMENTATION,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": learning_rate,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "runtime_seconds": run_duration,
        }, model_path)

        print(f"Saved model to: {model_path}")

        print(f"\nFinished run with data augmentation: {DATA_AUGMENTATION}")
        print(f"Runtime: {minutes} min {seconds:.1f} sec")

    
    

   

