# Task 0: Baseline CNN Classifier

## Objective
Build and train a simple CNN for binary image classification.

## Your Implementation
Used the CNN architecture from "nb_beginner_guide.ipynb" and implemented additional functionality such as plotting of train and validation losses. Also implemented data augmentation and compared training runs with and without data augmentation for N_EPOCHS = 150

## Results
The script ran locally on my GTX 1060 6GB

With data augmentation:
- Test Accuracy: 96.81%
- Training Time: 5 min 31.0 sec


Without data augmentation:
- Test Accuracy: 97.16%
- Training Time: 1 min 54.1 sec


## Files
- `model.py` - CNN architecture
- `train.py` - Training script
- `results/` - Saved models and metrics
