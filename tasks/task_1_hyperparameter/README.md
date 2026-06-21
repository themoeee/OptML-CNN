# Task 1: Hyperparameter Studies & Fine-tuning

## Objective
Systematically explore how hyperparameters affect model performance.

## Approach
*Describe whether you used Grid Search, Random Search, or Optuna*
I used Optuna for hyperparameter optimization, as the software seemed really capable and I liked the built-in validation capabilities using their Browser interface.

## Hyperparameters Explored
- Learning rate: [...]
- Batch size: [...]
- *Add others*

## Results
Optuna results can be seen by typing "optuna-dashboard sqlite:///results/optuna_study.db --host 127.0.0.1 --port 50000" into bash/powershell inside the task_1_hyperparameter folder
*Add your results table/heatmap here*

## Best Configuration
*Describe the best hyperparameter combination found*

## Files
- `grid_search.py` or `optuna_search.py` - Search script
- `results/` - Search results and visualizations
