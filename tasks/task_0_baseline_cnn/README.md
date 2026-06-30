# Task 0: Baseline CNN Classifier

## Objective
Build and train a simple CNN for binary image classification.

## Your Implementation
Used the CNN architecture from "nb_beginner_guide.ipynb" and implemented additional functionality such as plotting of train and validation losses. Also implemented data augmentation and compared training runs with and without data augmentation for N_EPOCHS = 150. The model consists of 3 convolutional layers in total, the full model architecture can be checken in file model.py

## Results
The script ran locally on my GTX 1060 6GB

With data augmentation:
- Test Accuracy: 96.45%
- Training Time: 6 min 23.8 sec


Without data augmentation:
- Test Accuracy: 96.45%
- Training Time: 1 min 58.7 sec


## Files
- `model.py` - CNN architecture
- `train.py` - Training script
- `results/` - Saved models and metrics
