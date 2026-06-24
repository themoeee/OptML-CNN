from pathlib import Path
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parents[2]

TASK1_DIR = ROOT / "tasks" / "task_1_hyperparameter"
sys.path.append(str(TASK1_DIR))
from model_task1 import SimpleCNN  

def load_dataset(category: str, split: str) -> TensorDataset:
    """Load a dataset split as a PyTorch TensorDataset."""
    #data = np.load(f"C:/Code/Opt-ML/Project2/data/processed/{category}/{split}.npz")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]      #had to fix this for laptop
    target_path = PROJECT_ROOT / "data" / "processed" / category / f"{split}.npz"                                 # PATH TO DATASET SPLITS
    data = np.load(f"{target_path}")
    images = torch.from_numpy(data["images"])
    labels = torch.from_numpy(data["labels"])
    return TensorDataset(images, labels)

def get_sample_images(dataset, num_samples=5, seed = 42):
    #Return specified number of random images, return equal (+-1) number of each class
    np.random.seed(seed)
    class_0_indices = np.where(dataset.tensors[1].numpy() == 0)[0]
    class_1_indices = np.where(dataset.tensors[1].numpy() == 1)[0]
    num_class_0 = num_samples // 2
    num_class_1 = num_samples - num_class_0
    selected_class_0 = np.random.choice(class_0_indices, num_class_0, replace=False)
    selected_class_1 = np.random.choice(class_1_indices, num_class_1, replace=False)
    selected_indices = np.concatenate([selected_class_0, selected_class_1])
    np.random.shuffle(selected_indices) 
    all_images = dataset.tensors[0][selected_indices]
    all_labels = dataset.tensors[1][selected_indices]
    return all_images, all_labels

def load_model(checkpoint_path, device):
    try:
        model = SimpleCNN(kernel_size=5)
    except TypeError:
        print("Model loading failed with kernel_size=5, trying default constructor...")
        model = SimpleCNN()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def plot_grad_cam(model, image, label=None, target_class=None, save_path=None, show=True):
    """
    Grad-CAM for one grayscale image.
    image shape: (1, 128, 128) or (128, 128)
    """
    device = next(model.parameters()).device

    if image.ndim == 2:   # image shape (128, 128), make it to (1, 128, 128)
        image = image.unsqueeze(0)

    input_tensor = image.unsqueeze(0).float().to(device)  # now (1, 1, 128, 128)

    model.eval()
    output = model(input_tensor) # the two logits of class 0 and class 1, shape (1, 2), e.g. tensor([[-7.9993,  8.6769]]
    #print(output)

    pred = output.argmax(dim=1).item() #take highest logit (predicted class)
    prob = torch.softmax(output, dim=1)[0, pred].item() #make it a probability (between 0 and 1) using softmax

    if target_class is None:        #we could also visualize why the model might predict a specific class, we only show class with highest probability
        target_class = pred

    target_layers = [model.features[6]]  # last Conv2d layer
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  # (128, 128), already upsampled - important, because after conv3 picture is only size 32x32, needs to be upsampled to 128x128 

    # normalize to [0, 1]
    img = image[0].detach().cpu().numpy()
    img = img - img.min()
    img = img / (img.max() + 1e-8)

    rgb_img = np.stack([img, img, img], axis=-1).astype(np.float32) #since gradcam expects RGB image

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Input\nlabel={label}")
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title(f"Grad-CAM heatmap\nclass={target_class}")
    axes[1].axis("off")

    axes[2].imshow(visualization)
    axes[2].set_title(f"Overlay\npred={pred}, p={prob:.2f}")
    axes[2].axis("off")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    return grayscale_cam, pred, prob


if __name__ == "__main__":
    # Display options. Figures are always saved; False only suppresses plt.show().
    SHOW_GRAD_CAM = True

    SEED = 42

    num_samples = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    results_dir = ROOT / "tasks" / "task_3_interpretability" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = ROOT / "tasks" / "task_1_hyperparameter" / "results" / "final_best_model_task1.pth"
    model = load_model(checkpoint_path, device) # Load the model from task 1 hyperparam optim using optuna


    # Load a sample image from the training set
    print("Loading sample images from the test dataset")
    test_dataset = load_dataset(category="NT", split="test") #debate test vs train: test is new data and can therefore show what the model learned
    sample_images, sample_labels = get_sample_images(test_dataset, num_samples=num_samples, seed=SEED) #get 1 sample image from the test dataset
    
    for i in range(len(sample_images)):
        print(f"Sample {i} - Label: {sample_labels[i].item()}")
        save_path = results_dir / f"grad_cam_sample_{i}.png"

        plot_grad_cam(model=model, image=sample_images[i], label=sample_labels[i].item(), target_class=None, save_path=save_path, show=SHOW_GRAD_CAM)
