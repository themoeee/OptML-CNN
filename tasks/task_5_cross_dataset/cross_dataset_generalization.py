# IN THIS FILE WE WANT TO TEST OUR PREVIOUS MODEL (TRAINED ON NT SAMPLES) FOR ASB AND UT SAMPLES AND SEE HOW WELL IT GENERALIZES

# FOR THIS, WE WILL AT FIRST LOAD OUR PREVIOUS FINAL MODEL JUST AS IN TASK 2

# addendum: as we want the full 3x3 cross generalization matrix, first we have to train all 3 models
# I will use the same workflow and pipeline as before, using optuna to find the best hyperparams and then training a final model on that
# afterwards, this file here will serve the purpose of loading these 3 models and testing them on all 3 test datasets


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

TASK_1_DIR = Path(__file__).resolve().parents[1] / "task_1_hyperparameter"
sys.path.append(str(TASK_1_DIR))
from model_task1 import SimpleCNN

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(relative_path):
    return str(PROJECT_ROOT / relative_path)

def load_trained_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_params = checkpoint["best_params"]

    model = SimpleCNN(kernel_size=best_params["kernel_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model

def load_dataset(category: str, split: str) -> TensorDataset:
    """Load a dataset split as a PyTorch TensorDataset."""
    data = np.load(project_path(f"data/processed/{category}/{split}.npz"))
    images = torch.from_numpy(data["images"])
    labels = torch.from_numpy(data["labels"])
    return TensorDataset(images, labels)

def create_generalization_matrix(results):          # THIS FUNCTION USES THE RESULTS FROM BEFORE TO PLOT THE GENERALIZATION MATRIX
    # I want to plot a 3x3 matrix with the test accuracy for each model on each test set, using the results dictionary
    model_names = list(results.keys())
    dataset_names = list(results[model_names[0]].keys())    
    matrix = np.zeros((len(model_names), len(dataset_names)))
    for i, model_name in enumerate(model_names):
        for j, dataset_name in enumerate(dataset_names):
            matrix[i, j] = results[model_name][dataset_name]["test_accuracy"]
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(np.arange(len(dataset_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(dataset_names)
    ax.set_yticklabels(model_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(model_names)):
        for j in range(len(dataset_names)):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}%", ha="center", va="center", color="w")
    ax.set_title("Cross-Dataset Generalization Matrix (Test Accuracy %)")
    fig.tight_layout()  
    plt.colorbar(im)
    plt.savefig("results/cross_generalization_matrix.png")    
    plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    model_path = PROJECT_ROOT 

   # print(str(PROJECT_ROOT / "tasks/task_1_hyperparameter/results/final_best_model_task1.pth"))

    model_NT = load_trained_model(str(PROJECT_ROOT / "tasks/task_1_hyperparameter/results/final_best_model_task1.pth"), device=device)
    model_ASB = load_trained_model(str(PROJECT_ROOT / "tasks/task_5_cross_dataset/results/final_best_model_ASB.pth"), device=device)
    model_UT = load_trained_model(str(PROJECT_ROOT / "tasks/task_5_cross_dataset/results/final_best_model_UT.pth"), device=device)

    models = {
        "NT": model_NT,
        "ASB": model_ASB,
        "UT": model_UT}
    
    if torch.cuda.is_available():
        print("GPU name:         ", torch.cuda.get_device_name(0))

    batch_size = 32

    test_dataset_NT = load_dataset("NT", "test")
    test_dataset_ASB = load_dataset("ASB", "test")
    test_dataset_UT = load_dataset("UT", "test")

    test_loader_NT = DataLoader(test_dataset_NT, batch_size=batch_size, shuffle=False)
    test_loader_ASB = DataLoader(test_dataset_ASB, batch_size=batch_size, shuffle=False)
    test_loader_UT = DataLoader(test_dataset_UT, batch_size=batch_size, shuffle=False)


    test_loaders = {
    "NT": DataLoader(test_dataset_NT, batch_size=batch_size, shuffle=False),
    "ASB": DataLoader(test_dataset_ASB, batch_size=batch_size, shuffle=False),
    "UT": DataLoader(test_dataset_UT, batch_size=batch_size, shuffle=False),
    }

    criterion = nn.CrossEntropyLoss()

    results = {}
#     "ASB": {
#         "ASB": {"test_loss": ..., "test_accuracy": ..., "n_samples": ...},
#         "NT":  {"test_loss": ..., "test_accuracy": ..., "n_samples": ...},
#         "UT":  {"test_loss": ..., "test_accuracy": ..., "n_samples": ...},
#     },
#     "NT": {...},
#     "UT": {...},
# }


    for model_name, model in models.items():
        
        print(f"\nTesting model trained on {model_name} samples:")
        
        model.eval()
        results[model_name] = {}
        
        with torch.no_grad():
            for dataset_name, test_loader in test_loaders.items():
                
                test_loss = 0.0
                test_correct = 0
                test_size = len(test_loader.dataset)
                test_set_name = dataset_name
                #print("Test size", test_size)

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    test_correct += (outputs.argmax(1) == labels).sum().item()

                test_loss = test_loss / test_size
                test_acc = 100 * test_correct / test_size
                
                results[model_name][test_set_name] = {
                                                        "test_loss": test_loss,
                                                        "test_accuracy": test_acc,
                                                        "n_samples": test_size,
                                                    }

        
    #HERE PLOT THE MATRIX
    create_generalization_matrix(results)    