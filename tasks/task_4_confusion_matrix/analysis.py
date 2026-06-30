# This script is meant to show the confusion matrix on the test set. This is meant to show, where
# the model fails and trying to understand why and if its harder to detect a fracture or no fracture. Also looks for patterns in the errors

# compute and visualize the confusion matrix on the test set
# • Calculate precision, recall, F1-score per class
# • Identify and visualize misclassified examples
# • Analyze: Are there patterns in the errors? (e.g., ambiguous images, edge cases

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
 
def calculate_precision(true_labels, pred_labels):
    """Calculate precision for each class."""
    cm = confusion_matrix(true_labels, pred_labels)
    precision_per_class = cm.diagonal() / cm.sum(axis=0)
    return precision_per_class

def calculate_recall(true_labels, pred_labels):
    """Calculate recall for each class."""
    cm = confusion_matrix(true_labels, pred_labels)
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    return recall_per_class


def calculate_f1_score(precision, recall):
    """Calculate F1-score for each class."""
    f1_per_class = 2 * (precision * recall) / (precision + recall)
    return f1_per_class 

def plot_f1_matrix(f1_scores, class_names):
    """Plot F1-score matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(f1_scores.reshape(1, -1), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("F1-score Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks([])  # No y-ticks since it's a single row
    for i, j in np.ndindex(f1_scores.reshape(1, -1).shape):
        plt.text(j, i, format(f1_scores[j], '.2f'),
                 horizontalalignment="center",
                 color="white" if f1_scores[j] > 0.5 else "black")
    plt.title('F1-score Matrix')
    plt.xlabel('Classes')
    plt.ylabel('Classes')
    plt.show()

def show_misclasified_examples(images, true_labels, pred_labels, num_examples=5):
    """Visualize misclassified examples."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    misclassified_indices = np.where(true_labels != pred_labels)[0] #before: error bc return single boolean not element-wise comparison
    if len(misclassified_indices) == 0:
        print("No misclassified examples found.")
        return

    num_examples = min(num_examples, len(misclassified_indices))
    selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(selected_indices):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.title(f"True: {true_labels[idx]}, Pred: {pred_labels[idx]} \n Index: {idx}")
        plt.axis('off')
    plt.savefig("figures/misclassified_examples.png")
    plt.show()

if __name__ == "__main__":
    # Load the test dataset
    category = "NT"  # Change this to the desired category
    test_dataset_nt = load_dataset(category=category, split="test")
    test_loader_nt = DataLoader(test_dataset_nt, batch_size=32, shuffle=False)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path=ROOT / "tasks" / "task_1_hyperparameter" / "results" / "final_best_model_task1.pth", device=device)

    # Initialize lists to store true and predicted labels
    all_true_labels = []
    all_pred_labels = []

    # Iterate through the test dataset and collect predictions
    with torch.no_grad():
        for images, labels in test_loader_nt:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(preds.cpu().numpy())

    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    print("Confusion Matrix:\n", cm)

    # Calculate precision, recall, F1-score per class
    report = classification_report(all_true_labels, all_pred_labels, target_names=["No Fracture", "Fracture"])
    print("Classification Report:\n", report)

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Fracture", "Fracture"], rotation=45)
    plt.yticks(tick_marks, ["No Fracture", "Fracture"])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.title('Confusion Matrix for {category} Category'.format(category=category))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png")
    plt.show()

    # precision
    precision = calculate_precision(all_true_labels, all_pred_labels)
    print("Precision per class:", precision)

    # recall
    recall = calculate_recall(all_true_labels, all_pred_labels)
    print("Recall per class:", recall)

    # F1-score
    f1_score = calculate_f1_score(precision, recall)
    print("F1-score per class:", f1_score)

    # Show misclassified examples
    test_images = test_dataset_nt.tensors[0].numpy()
    show_misclasified_examples(test_images, all_true_labels, all_pred_labels, num_examples=5)
