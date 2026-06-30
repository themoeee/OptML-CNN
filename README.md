# OptML CNN Project

This repository contains the code, experiments, and result artifacts for the
Computer Vision project in the ETH Zurich course **Optimization and Machine
Learning**.

The project studies binary fracture classification on grayscale microscopy
images from mechanical fracture experiments. Each image is classified as:

- `0`: no fracture
- `1`: fracture

The images have shape `128 x 128` pixels and are stored in channel-first format
for PyTorch: `(N, 1, 128, 128)`.

## Project Scope

The main goal is to train and analyze CNN-based classifiers for detecting
fractures. The repository is organized by project task:

- train a baseline CNN classifier
- optimize CNN hyperparameters with Optuna
- evaluate robustness under Gaussian noise
- visualize learned CNN filters, activations, and Grad-CAM heatmaps
- analyze test-set errors with a confusion matrix
- test cross-dataset generalization
- keep an optional CNN vs. Vision Transformer task template

Most final experiments use the `NT` dataset as the main training and evaluation
dataset. The cross-dataset experiment focuses on transfer between `NT` and `UT`;
`ASB` data is present in the repository, but the final cross-evaluation script
comments it out.

## Data

The project uses three dataset categories:

- `ASB`
- `NT`
- `UT`

The original MATLAB files are expected in:

```text
data/mmc1/
```

The training scripts use processed NumPy files in:

```text
data/processed/
|-- ASB/
|   |-- train.npz
|   |-- val.npz
|   `-- test.npz
|-- NT/
|   |-- train.npz
|   |-- val.npz
|   `-- test.npz
`-- UT/
    |-- train.npz
    |-- val.npz
    `-- test.npz
```

Each `.npz` file contains:

- `images`: image tensors with shape `(N, 1, 128, 128)`
- `labels`: binary labels with shape `(N,)`

If the processed files need to be regenerated from the `.mat` files, run:

```bash
python utils/export_dataset.py
```

The export script creates stratified train/validation/test splits with a
`60/20/20` ratio and random seed `42`.

## Setup

Create and activate a Python environment:

```bash
conda create -n optml-cnn python=3.10
conda activate optml-cnn
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, follow the note in `requirements.txt` and install the
matching PyTorch build for your system.

## Repository Structure

```text
OptML-CNN/
|-- data/
|   |-- mmc1/
|   `-- processed/
|-- notebooks/
|   `-- nb_beginner_guide.ipynb
|-- report/
|   `-- report/
|       `-- report_CNN.tex
|-- results/
|-- tasks/
|   |-- task_0_baseline_cnn/
|   |-- task_1_hyperparameter/
|   |-- task_2_robustness/
|   |-- task_3_interpretability/
|   |-- task_4_confusion_matrix/
|   |-- task_5_cross_dataset/
|   `-- task_6_cnn_vs_vit/
|-- utils/
|   `-- export_dataset.py
|-- PROJECT_INSTRUCTIONS.md
|-- requirements.txt
`-- README.md
```

## Task Overview

| Task | Folder | Main files | Status |
| --- | --- | --- | --- |
| Task 0: Baseline CNN | `tasks/task_0_baseline_cnn/` | `model.py`, `train.py`, `load_model.py` | Implemented |
| Task 1: Hyperparameter optimization | `tasks/task_1_hyperparameter/` | `model_task1.py`, `train_task1.py`, `optuna_search.py` | Implemented |
| Task 2: Robustness analysis | `tasks/task_2_robustness/` | `noise.py` | Implemented |
| Task 3: Interpretability | `tasks/task_3_interpretability/` | `visualize_filters.py`, `gradcam.py` | Implemented |
| Task 4: Confusion matrix | `tasks/task_4_confusion_matrix/` | `analysis.py` | Implemented |
| Task 5: Cross-dataset generalization | `tasks/task_5_cross_dataset/` | `train_UT_ASB.py`, `cross_dataset_generalization.py` | Implemented for NT/UT |
| Task 6: CNN vs. ViT | `tasks/task_6_cnn_vs_vit/` | `README.md` | Optional template |

Each implemented task folder contains its own `README.md` with task-specific
details, result summaries, and output files.

## Model

The baseline CNN is a compact three-layer convolutional network:

```text
Conv2d(1, 32) -> ReLU -> MaxPool2d(2)
Conv2d(32, 64) -> ReLU -> MaxPool2d(2)
Conv2d(64, 128) -> ReLU -> AdaptiveAvgPool2d(1)
Linear(128, 2)
```

Task 1 keeps the same general architecture but allows Optuna to tune the kernel
size, optimizer, learning rate, batch size, momentum, and data augmentation.

## Main Results

### Task 0: Baseline CNN

The baseline model was trained on the `NT` category for 150 epochs, once with
data augmentation and once without it.

| Setting | Test accuracy | Training time |
| --- | ---: | ---: |
| Without data augmentation | 96.45% | 1 min 58.7 sec |
| With data augmentation | 96.45% | 6 min 23.8 sec |

Outputs include trained checkpoints and loss curves in
`tasks/task_0_baseline_cnn/results/`.

### Task 1: Hyperparameter Optimization

Optuna was used to minimize validation loss over the following search space:

- learning rate: `1e-5` to `1e-2`
- batch size: `8`, `16`, `32`, `64`, `128`
- kernel size: `3`, `5`, `7`
- optimizer: `Adam` or `SGD`
- momentum: `0.0` to `0.95` for SGD
- data augmentation: `True` or `False`

Best recorded configuration:

```text
validation loss: 0.01706585101136524
learning_rate: 0.006578226274922781
batch_size: 8
kernel_size: 5
optimizer: Adam
data_augmentation: True
```

The final checkpoint is saved as:

```text
tasks/task_1_hyperparameter/results/final_best_model_task1.pth
```

### Task 2: Robustness Under Gaussian Noise

The best Task 1 model was evaluated on noisy `NT` test images without
retraining.

| Noise sigma | Test accuracy |
| ---: | ---: |
| 0.000 | 96.45% |
| 0.050 | 96.45% |
| 0.075 | 90.78% |
| 0.100 | 73.76% |
| 0.125 | 64.54% |
| 0.150 | 54.96% |
| 0.200 | 50.35% |
| 0.300 | 50.35% |
| 0.500 | 50.35% |

The model starts to fail around `sigma = 0.15`; above `sigma = 0.2`, the images
are too noisy for meaningful classification and performance approaches chance
level.

### Task 3: Interpretability

The interpretability task visualizes:

- learned convolutional filters
- activation maps across convolutional layers
- Grad-CAM heatmaps for selected test samples

The visualizations suggest that early layers detect local contrast and dark
regions, while later layers respond more strongly to crack regions and intact
material regions. Result images are stored in
`tasks/task_3_interpretability/results/`.

### Task 4: Confusion Matrix and Error Analysis

The confusion matrix analysis evaluates the best Task 1 model on the `NT` test
set.

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| `0` no fracture | 95.21% | 97.89% | 96.53% |
| `1` fracture | 97.79% | 95.00% | 96.38% |

Misclassified examples are often visually ambiguous, such as very thin fracture
lines or air pockets that resemble cracks.

### Task 5: Cross-Dataset Generalization

The final cross-dataset experiment compares models trained on `NT` and `UT` and
tests them on both datasets.

| Train / Test | NT | UT |
| --- | ---: | ---: |
| NT | 96.45% | 94.68% |
| UT | 99.44% | 96.61% |

The transfer results indicate that the CNN learns fracture-related features
that transfer well between these two specimen categories.

## Running the Code

Run task scripts from inside the corresponding task folder because several
scripts save outputs to relative `results/` or `figures/` directories.

### Baseline CNN

```bash
cd tasks/task_0_baseline_cnn
python train.py
```

To load and evaluate the saved baseline checkpoints:

```bash
cd tasks/task_0_baseline_cnn
python load_model.py
```

### Hyperparameter Optimization

```bash
cd tasks/task_1_hyperparameter
python optuna_search.py
```

To inspect the Optuna study with the dashboard:

```bash
cd tasks/task_1_hyperparameter
optuna-dashboard sqlite:///results/optuna_study.db --host 127.0.0.1 --port 50000
```

Then open:

```text
http://127.0.0.1:50000
```

To evaluate the saved best Task 1 model:

```bash
cd tasks/task_1_hyperparameter
python train_task1.py
```

### Robustness Analysis

```bash
cd tasks/task_2_robustness
python noise.py
```

This script depends on:

```text
tasks/task_1_hyperparameter/results/final_best_model_task1.pth
```

### Interpretability

```bash
cd tasks/task_3_interpretability
python visualize_filters.py
python gradcam.py
```

These scripts also use the final Task 1 checkpoint.

### Confusion Matrix

```bash
cd tasks/task_4_confusion_matrix
python analysis.py
```

Generated figures are saved to:

```text
tasks/task_4_confusion_matrix/figures/
```

### Cross-Dataset Generalization

```bash
cd tasks/task_5_cross_dataset
python cross_dataset_generalization.py
```

This script loads the saved `NT` and `UT` checkpoints and creates the
cross-generalization matrix in:

```text
tasks/task_5_cross_dataset/results/
```

## Reproducibility Notes

- Several scripts automatically use CUDA when available and otherwise fall back
  to CPU.
- Task 2, Task 3, Task 4, and Task 5 depend on the final Task 1 checkpoint.
- Some generated files, such as model checkpoints and plots, are already stored
  in the task-specific `results/` or `figures/` folders.
- The code uses relative paths from the repository root for data loading, but
  several output paths are relative to the current task folder.
- The test split is used only for final evaluation and analysis.

## Reference

Adrien Mueller et al. (2021). *Machine Learning Classifiers for Surface Crack
Detection in Fracture Experiments.*
