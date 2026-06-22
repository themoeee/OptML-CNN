from pathlib import Path
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


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

def get_conv_layers(model):
    #Every named module is a conv layer in our model
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d): #only return Conv2d layers
            conv_layers.append((name, module))
    return conv_layers
def plot_filter(model, filter_idx=0, conv_layer_idx=0):
    conv_layers = get_conv_layers(model)

    if conv_layer_idx >= len(conv_layers):
        raise ValueError(f"Invalid conv_layer_idx. Available: 0 to {len(conv_layers)-1}")

    layer_name, conv_layer = conv_layers[conv_layer_idx]
    weights = conv_layer.weight.detach().cpu()
    # shape: (out_channels, in_channels, kernel_h, kernel_w)

    if filter_idx >= weights.shape[0]:
        raise ValueError(f"Invalid filter_idx. Available: 0 to {weights.shape[0]-1}")

    if weights.shape[1] == 1:
        # Conv1: directly interpretable grayscale filter
        filt = weights[filter_idx, 0]
    else:
        # Conv2/Conv3: filter spans multiple input feature channels
        filt = weights[filter_idx].mean(dim=0)

    vmax = filt.abs().max()

    plt.figure(figsize=(3, 3))
    plt.imshow(filt, cmap="seismic", vmin=-vmax, vmax=vmax)
    plt.title(f"{layer_name} | Filter {filter_idx}")
    plt.colorbar()
    plt.axis("off")
    plt.show()

def plot_all_filters(model):
    conv1 = model.features[0]
    weights = conv1.weight.detach().cpu()   # shape: (32, 1, 5, 5)

    n_filters = weights.shape[0]
    cols = 8
    rows = math.ceil(n_filters / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(n_filters):
        filt = weights[i, 0]
        vmax = filt.abs().max()

        axes[i].imshow(filt, cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"F{i}", fontsize=8)
        axes[i].axis("off")

    for j in range(n_filters, len(axes)):
        axes[j].axis("off")

    plt.suptitle("All filters of first convolutional layer", fontsize=14)
    plt.tight_layout()
    plt.savefig(ROOT / "tasks" / "task_3_interpretability" / "results" / "all_filters_conv1.png", dpi=300, bbox_inches="tight")
    plt.show()

def activation_maps(model, image):
    # Get activations from the first convolutional layer for a given input image and plot them
    conv1 = model.features[0]
    with torch.no_grad():
        activations = conv1(image.unsqueeze(0).to(next(model.parameters()).device))

    #Plot activations of the first conv layer for all 32 filters
    n_filters = activations.shape[1]
    cols = 8
    rows = math.ceil(n_filters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(n_filters):
        act = activations[0, i].cpu()
        vmax = act.abs().max()

        axes[i].imshow(act, cmap="viridis", vmin=0, vmax=vmax)
        axes[i].set_title(f"Filter {i}", fontsize=8)
        axes[i].axis("off")
    for j in range(n_filters, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Activation maps of first conv layer", fontsize=14)
    plt.tight_layout()
    plt.show()

    return activations.squeeze(0).cpu()

def get_activation_maps(model, image, conv_layer_idx):
    """
    conv_layer_idx:
    0 -> Conv1 + ReLU: (32, 128, 128)
    1 -> Conv2 + ReLU: (64, 64, 64)
    2 -> Conv3 + ReLU: (128, 32, 32)
    """
    device = next(model.parameters()).device

    if image.ndim == 2:
        image = image.unsqueeze(0)

    image_batch = image.unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        if conv_layer_idx == 0:
            x = model.features[:2](image_batch)
        elif conv_layer_idx == 1:
            x = model.features[:5](image_batch)
        elif conv_layer_idx == 2:
            x = model.features[:8](image_batch)
        else:
            raise ValueError("conv_layer_idx must be 0, 1, or 2")

    return x.detach().cpu()

def plot_all_activation_maps_with_input(model, image, label=None, conv_layer_idx=0, save_path=None, show=True, scale="per_map"):
    """
    scale:
        "global"  -> same activation scale for all channels
        "per_map" -> each activation map normalized individually
    """
    layer_names = {
        0: "Conv1",
        1: "Conv2",
        2: "Conv3"
    }

    if conv_layer_idx in [0, 1]:
        cols = 8
    elif conv_layer_idx == 2:
        cols = 16
    else:
        raise ValueError("conv_layer_idx must be 0, 1, or 2")

    device = next(model.parameters()).device

    if image.ndim == 2:
        image = image.unsqueeze(0)

    # prediction
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).item()
        prob = torch.softmax(output, dim=1).max().item()

    activations = get_activation_maps(model, image, conv_layer_idx)
    fmap = activations[0]  # (C, H, W)

    n_channels, h, w = fmap.shape
    rows = math.ceil(n_channels / cols)

    fig = plt.figure(figsize=(2.0 * (cols + 1), 2.0 * rows))
    gs = fig.add_gridspec(
        rows,
        cols + 1,
        width_ratios=[1.35] + [1] * cols
    )

    # input image upper left
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(image[0].cpu(), cmap="gray")
    ax_input.set_title(
        f"Input\nlabel={label}, pred={pred}, p={prob:.2f}",
        fontsize=8
    )
    ax_input.axis("off")

    if scale == "global":
        vmin = 0
        vmax = fmap.max().item()
        if vmax == 0:
            vmax = 1.0
    elif scale == "per_map":
        vmin, vmax = 0, 1
    else:
        raise ValueError("scale must be 'global' or 'per_map'")

    im = None

    for ch in range(n_channels):
        row = ch // cols
        col = ch % cols + 1

        ax = fig.add_subplot(gs[row, col])
        act = fmap[ch]

        if scale == "per_map":
            act = act - act.min()
            act = act / (act.max() + 1e-8)

        im = ax.imshow(act, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Ch {ch}", fontsize=7)
        ax.axis("off")

    # Colorbar links unter dem Input
    if rows > 1 and im is not None:
        cax = fig.add_subplot(gs[1:, 0])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Activation", fontsize=8)
        cax.tick_params(labelsize=7)

    fig.suptitle(
        f"{layer_names[conv_layer_idx]} activation maps "
        f"({n_channels} channels, {h}x{w})",
        fontsize=14
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint_path = ROOT / "tasks" / "task_1_hyperparameter" / "results" / "final_best_model_task1.pth"
    model = load_model(checkpoint_path, device)

    conv_layers = get_conv_layers(model)
    for name, layer in conv_layers:
        print(f"Layer: {name}, Kernel Size: {layer.kernel_size}, Out Channels: {layer.out_channels}")

    #plot_filter(model, filter_idx=0)
    plot_all_filters(model)
    
    # Load a sample image from the training set
    train_dataset = load_dataset(category="NT", split="train")
    sample_images, sample_labels = get_sample_images(train_dataset, num_samples=5)
    

    for i in range(5):
        print(f"Sample {i} - Label: {sample_labels[i].item()}")
        plt.imshow(sample_images[i][0], cmap="gray")
        plt.title(f"Sample {i} - Label: {sample_labels[i].item()}")
        plt.axis("off")
        plt.show()

       # activations = activation_maps(model, sample_images[i])


        #plot_all_activation_maps_with_input(model, sample_images[i], sample_labels[i].item(), conv_layer_idx=0)
        #plot_all_activation_maps_with_input(model, sample_images[i], sample_labels[i].item(), conv_layer_idx=1)
        plot_all_activation_maps_with_input(model, sample_images[i], sample_labels[i].item(), conv_layer_idx=2)
