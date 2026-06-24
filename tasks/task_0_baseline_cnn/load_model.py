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


def load_finished_model(model_path):
    """Load a trained model from a .pt file."""
    try:
        checkpoint = torch.load(model_path)
        model = SimpleCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return None, None
    model.eval()
    return model, checkpoint

if __name__ == "__main__":
   
    category = "NT"  
   
    DATA_AUGMENTATION = [False, True]
    
    for aug in DATA_AUGMENTATION:
        print(f"Loading model with data augmentation: {aug}")
        model_path = Path("./results") / f"model_{category}_baseline_cnn_aug{aug}.pt"
        model, checkpoint = load_finished_model(model_path)
        test_loss, test_acc = checkpoint["test_loss"], checkpoint["test_acc"]
        print(f"Loaded model with data augmentation: {aug} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        runtime = checkpoint['runtime_seconds']
        minutes = int(runtime // 60)
        seconds = runtime % 60
        print(f"Runtime of model: {minutes} min {seconds:.2f} sec")

