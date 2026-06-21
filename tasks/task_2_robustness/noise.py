import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

from pathlib import Path
import sys


TASK_1_DIR = Path(__file__).resolve().parents[1] / "task_1_hyperparameter"
sys.path.append(str(TASK_1_DIR))
from model_task1 import SimpleCNN


def add_gaussian_noise ( images , sigma ) :
    noise = torch . randn_like ( images ) * sigma
    return torch . clamp ( images + noise , 0 , 1)


def load_dataset_with_noise(category: str, split: str, sigma: float) -> TensorDataset:
    """Load a dataset split as a PyTorch TensorDataset."""
    PROJECT_ROOT = Path(__file__).resolve().parents[2]      #had to fix this for laptop
    target_path = PROJECT_ROOT / "data" / "processed" / category / f"{split}.npz"        
    data = np.load(target_path)
    images = torch.from_numpy(data["images"])
    labels = torch.from_numpy(data["labels"]) 

    images = add_gaussian_noise(images, sigma=sigma)              # Add Gaussian noise to the images!

    return TensorDataset(images, labels)

def load_trained_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    best_params = checkpoint["best_params"]

    model = SimpleCNN(kernel_size=best_params["kernel_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

def get_sample_image_from_noise_level(sigma: float):
    test_dataset = load_dataset_with_noise("NT", "test", sigma)
    return test_dataset[0][0], test_dataset[1][0], test_dataset[2][0]       # Return the first 3 images from the dataset


if __name__ == "__main__":
   
    noise_levels = [0, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.5]  # Different levels of Gaussian noise to test
    batch_size = 32
    data_augmentation = True
    num_epochs = 50
    learning_rate = 1e-3
    kernel_size = 3
    optimizer_name = 'Adam'
    momentum = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_to_model = Path(__file__).resolve().parents[1] / "task_1_hyperparameter" / "results" / "final_best_model_task1.pth"
    model = load_trained_model(path_to_model, device=device)

    
    if torch.cuda.is_available():
        print("GPU name:         ", torch.cuda.get_device_name(0))


    accuracies = []
    

    for noise_level in noise_levels:
        print(f"\nTesting with Gaussian noise (sigma={noise_level})")
        
        # I DONT HAVE TO TRAIN NEW MODELS WITH DIFFERENT NOISE LEVELS (WAS MY FIRST THOUGHT), INSTEAD JUST USE BEST MODEL FROM TASK 1 AND EVALUATE IT WITH DIFFERENT NOISE LEVELS

        test_dataset = load_dataset_with_noise("NT", "test", noise_level)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()

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

        accuracies.append(test_acc)

        print(f"  Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        #HERE I GENERATE THE SAMPLE IMAGES

        image1, image2, image3 = get_sample_image_from_noise_level(noise_level)
        
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, image in zip(axes, images):
            ax.imshow(image.cpu().squeeze(), cmap='gray')
            ax.axis("off")
        fig.suptitle(f"Gaussian Noise: σ = {noise_level}")
        fig.tight_layout()
        results_dir = Path("./results")
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / f"sample_images_sigma_{noise_level}.png")
        plt.show()


    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, accuracies, marker='o')
    plt.title("Model Accuracy vs. Gaussian Noise Level")
    plt.xlabel("Gaussian Noise Sigma")
    plt.ylabel("Test Accuracy (%)")
    plt.xticks(noise_levels, [str(n) for n in noise_levels])
    plt.grid()
    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / f"accuracy_vs_noise.png")
    plt.show()
