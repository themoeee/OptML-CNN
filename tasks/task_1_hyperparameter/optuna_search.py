# IDEA IS TO IMPLEMENT HHYPERPARAMETER OPTIMIZATION USING OPTUNA 
# READ PAPER "Optuna: A Next-generation Hyperparameter OptimizationFramework.”

# Optuna formulates the hyperparameter optimization
# as a process of minimizing/maximizing an objective function
# that takes a set of hyperparameters as an input and returns its
# (validation) score.

#okay, lets first think about which hyperparameters there are to optimize.
#1. learning rate (maybe even learning rate scheduler?)
#2. batch size
#3. number of epochs
#4. number of layers
#5. model architecture
#6. kernel size

# HOW TO USE OPTUNA:
# Wrap model training with an objective function and return accuracy
# Suggest hyperparameters using a trial object
# Create a study object and execute the optimization

# WHICH HYPERPARAMETERS TO OPTIMIZE:   Learning rate, batch size, kernel size, num_epochs (and optimizer)

from pyexpat import model

import optuna
import torch

from train_task1 import train_model
from model_task1 import SimpleCNN, ImageClassificationDataset


# 1. Define an objective function to be maximized.
def objective(trial):

    # 2. Suggest values of the hyperparameters using a trial object.
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)               # suggest learning rate between 1e-5 and 1e-2 on a log scale
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])               # suggest batch size of either 8, 16, 32, 64 or 128
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])                        # use kernel size of either 3, 5 or 7
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])                # suggest optimizer of either Adam or SGD
    momentum = trial.suggest_float("momentum", 0.0, 0.95) if optimizer_name == "SGD" else 0.0   # only suggest momentum if optimizer is SGD, otherwise set it to 0.0
    #num_epochs = trial.suggest_int('num_epochs', 50, 200)                                   # suggest number of epochs between 5 and 20
    num_epochs = 50                                                                          # for debugging just fixed number of epochs
    data_augmentation = trial.suggest_categorical('data_augmentation', [True, False])        # suggest whether to use data augmentation or not

    best_val_loss_epoch, __model = train_model(batch_size, data_augmentation, num_epochs, learning_rate, kernel_size, optimizer_name, momentum)

    return best_val_loss_epoch
    

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(
    direction="minimize",                           #BE VERY CAREFUL HERE, EITHER MAXIMIZE ACCURACY OR MINIMIZE LOSS, IN THIS CASE WE MINIMIZE VAL LOSS
    study_name="cnn_nt_hyperparameter_search_min_loss",
    storage="sqlite:///results/optuna_study.db",
    load_if_exists=True
)
study.optimize(objective, n_trials=50) # do 50 runs in total, each with different hyperparameters suggested by optuna

print("="*60)
print("OPTUNA FINISHED OPTIMIZATION")
print("Best validation loss:", study.best_value)
print("Best hyperparameters:", study.best_params)


# NOW WE KNOW THE BEST HYPERPARAMETERS, WE CAN USE THEM TO TRAIN A FINAL MODEL 
# this way we can also save the final model

print("="*60)
print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")

best_params = study.best_params
final_num_epochs = 100

final_best_val_loss, final_model = train_model(
    batch_size=best_params["batch_size"],
    data_augmentation=best_params["data_augmentation"],
    num_epochs=final_num_epochs,
    learning_rate=best_params["learning_rate"],
    kernel_size=best_params["kernel_size"],
    optimizer_name=best_params["optimizer"],
    momentum=best_params.get("momentum", 0.0)
)

torch.save({
    "model_state_dict": final_model.state_dict(),
    "best_params": best_params,
    "num_epochs": final_num_epochs,
    "best_val_loss": final_best_val_loss
}, "results/final_best_model_task1.pth")

print(f"Finished training the final model with the best hyperparameters. Best validation loss: {final_best_val_loss}")
