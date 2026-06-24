# Task 2: Robustness Analysis (Gaussian Noise)

## Objective
Analyze how model accuracy degrades with increasing Gaussian noise.

## Noise Levels Tested
σ ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}

## Results
| σ (Noise Level) | Accuracy (%) |
|-----------------|--------------|
| 0.00 | 96.45% |
| 0.05 | 96.45% |
| 0.075 | 90.78% |
| 0.1 | 73.76% |
| 0.125 | 64.54% |
| 0.15 | 54.96% |
| 0.2 | 50.35% |
| 0.3 | 50.35% |
| 0.5 | 50.35% |

## Analysis
*Discuss at what noise level the model fails and why*
The model fails around a noise level of around 0.15, which at first glanced seemed to be quite low for me. After taking a look at actual pictures with this amount of noise I was however not surprised anymore, as I myself had a hard time distinguishing between good and bad samples at this noise level. I think the model could be improved to get to higher noise levels (for instance be incorpporating blurred samples into the training process), but above a noise level of 0.2 I think the data is to noisy to make meaningful predictions. Model fails as differences in pixels cannot be attributed to features in material (e.g. edges)

## Files
- `noise_analysis.py` - Analysis script
- `figures/` - Accuracy vs. noise plot, example images
