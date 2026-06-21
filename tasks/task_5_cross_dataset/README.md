# Task 5: Cross-Dataset Generalization

## Objective
Evaluate how well models generalize across different datasets.

## Generalization Matrix
| Train \ Test | NT | UT |
|--------------|-----|----|
| **NT** | 96.45% | 94.68% | 
| **UT** | 99.44% | 96.61% |

## Analysis
*Which transfers work well? Which fail? Why?*
Generally, the transfer seems to work quite well, especially if the model trained on NT is tested on UT. This results in a Accuracy which is even higher than both datasets trained on themselves, which is quite remarkable. The results from the UT model on NT testdata is the worst result, but still way above 90%. 
These results show that our model seems to learn certain features quite well (e.g. crack edges), which are quite well transferable to other specimen. Maybe the NT dataset contains some images with fractures not seen inside the UT dataset, which would explain the lower accuracy of the NT model here. In turn, UT might only containt "easier" samples, which therefore can be picked up by the NT model (which raises the question why the UT model on itself has a lower accuracy - maybe optuna did not find very good hyperparamers)

## Domain Adaptation (Bonus)
*If attempted, describe your approach*

## Files
- `cross_evaluation.py` - Cross-dataset evaluation script
- `results/` - Evaluation metrics
