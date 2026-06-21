import torch
import torch.nn as nn

from torch.utils.data import Dataset
from typing import Optional, Callable

from pathlib import Path
import numpy as np


# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),         #padding needed to keep image size the same
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
   


class ImageClassificationDataset(Dataset):
    """Custom PyTorch Dataset for image classification."""
    
    def __init__(
        self, 
        category: str, 
        split: str, 
        data_dir: str = Path(__file__).resolve().parents[2] / "data" / "processed",
        transform: Optional[Callable] = None
    ):
        """
        Args:
            category: One of 'ASB', 'NT', 'UT' # ASB currently not working!
            split: One of 'train', 'val', 'test'
            data_dir: Path to the processed data directory
            transform: Optional transform to apply to images
        """
        data_path = Path(data_dir) / category / f"{split}.npz"
        data = np.load(data_path)
        
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transform
        self.category = category
        self.split = split
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx])
        label = torch.tensor(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __repr__(self) -> str:
        return f"ImageClassificationDataset(category='{self.category}', split='{self.split}', size={len(self)})"
    

