import os

import torch.nn as nn
import torch.optim as optim

# Import models
from src.models.CNNten import CNNten, CNNten_multi_task
from src.models.smallFCN import smallFCN, smallFCN_multi_task, smallFCN_SelfAttention_multi_task
from src.models.ViT import ViT1D_multi_task
from src.models.MLPten import MLPten

# TODO: Add a wandb option

# Paths
DATA_DIR = 'training_data/simXRD_partial_data'
MODEL_SAVE_DIR = 'trained_models'

# Data
TRAIN_DATA = os.path.join(DATA_DIR, 'train.db')
VAL_DATA = os.path.join(DATA_DIR, 'val.db')
TEST_DATA = os.path.join(DATA_DIR, 'test.db')

# For file naming purposes
NAME_OF_DATA_USED = "simXRD_partial_data"

# Model Setup
MODEL_TYPE = "ViT1D_multi_task"                    # Options: "CNNten", CNNten_multi_task", "smallFCN", "smallFCN_multi_task", "smallFCN_SelfAttention_multi_task", "ViT1D_multi_task"
MULTI_TASK = True                                  # Set to True for multi-task learning (points train function to train_multi_spg_cryssystem_blt_element.py)

# IF SINGLE TASK, loss
CRITERION_TYPE = "CrossEntropyLoss"  # Options: "CrossEntropyLoss", "MSELoss"

# IF MULTI-TASK, loss
MULTI_TASK_CRITERIA = {
    'spg': nn.CrossEntropyLoss(),
    'crysystem': nn.CrossEntropyLoss(),
    'blt': nn.CrossEntropyLoss(),
    'composition': nn.BCEWithLogitsLoss()
}

# Hyper Params
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
NUM_EPOCHS = 20

# Optimiser
OPTIMIZER_TYPE = "Adam" # Options: "Adam", "SGD"

# Data Loading Settings
NUM_WORKERS = 6

# WandB configuration (Note that there is already a basic WandB log in train.py)
# TODO: USE_WANDB = True
WANDB_PROJECT_NAME = "FirstModelExperiments"
WANDB_SAVE_DIR = "/wandb"
SAVE_MODEL_TO_WANDB_SERVERS = False
WANDB_LOG_ARCHITECTURE = False

###########################################################################################
############# DON'T TOUCH - CThese classes contain the options for above ##################
MODEL_CLASS = {
    "CNNten": CNNten,
    "CNNten_multi_task": CNNten_multi_task,
    "smallFCN": smallFCN,
    "smallFCN_multi_task": smallFCN_multi_task,
    "smallFCN_SelfAttention_multi_task": smallFCN_SelfAttention_multi_task,
    "MLPten": MLPten,
    "ViT1D_multi_task": ViT1D_multi_task
}
CRITERION_CLASS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss
}
OPTIMIZER_CLASS = {
    "Adam": optim.Adam,
    "SGD": optim.SGD
}