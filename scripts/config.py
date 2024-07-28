import os

######################### READ: idiosyncratic path error #########################################
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# This sets the current path to the parent directory. I was getting annoyed at being cd into the wrong places.
# You shouldn't need this and can comment out this block.
##################################################################################################

import torch.nn as nn
import torch.optim as optim

# Import models
from src.models.CNNten import CNNten, CNNten_multi_task
from src.models.smallFCN import smallFCN, smallFCN_multi_task
from src.models.MLPten import MLPten
from src.models.Jackson import Jackson

# Paths
DATA_DIR = '/monfs01/projects/ys68/XRD_ML/simXRD_partial_data'
MODEL_SAVE_DIR = '/monfs01/projects/ys68/XRD_ML/trained_models'

# Data
TRAIN_DATA = os.path.join(DATA_DIR, 'train.db')
VAL_DATA = os.path.join(DATA_DIR, 'val.db')
TEST_DATA = os.path.join(DATA_DIR, 'test.db')

# For file naming purposes
NAME_OF_DATA_USED = "simXRD_partial_data"

# Model Setup
MODEL_TYPE = "smallFCN_multi_task"   # Options: "CNNten", CNNten_multi_task", "smallFCN", "smallFCN_multi_task", "Jackson"
MULTI_TASK = True                    # Set to True for multi-task learning (points train function to train_multi_spg_cryssystem_blt_element.py)
CRITERION_TYPE = "CrossEntropyLoss"  # Options: "CrossEntropyLoss", "MSELoss"
OPTIMIZER_TYPE = "Adam"              # Options: "Adam", "SGD"

# Multi-task specific settings. Ignore if not multi
MULTI_TASK_CRITERIA = {
    'spg': nn.CrossEntropyLoss(),
    'crysystem': nn.CrossEntropyLoss(),
    'blt': nn.CrossEntropyLoss(),
    'composition': nn.BCEWithLogitsLoss()
}

# Hyper Parasms
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 70

# Data Loading Settings
NUM_WORKERS = 8

# WandB configuration (Note that there is already a basic WandB log in train.py)
WANDB_PROJECT_NAME = "FirstModelExperiments"
WANDB_SAVE_DIR = "/monfs01/projects/ys68/XRD_ML"
SAVE_MODEL_TO_WANDB_SERVERS = False
WANDB_LOG_ARCHITECTURE = False


############# DON'T TOUCH - Classes are the options for above ##################
MODEL_CLASS = {
    "CNNten": CNNten,
    "CNNten_multi_task": CNNten_multi_task,
    "smallFCN": smallFCN,
    "smallFCN_multi_task": smallFCN_multi_task,
    "MLPten": MLPten,
    "Jackson": Jackson
}
CRITERION_CLASS = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss
}
OPTIMIZER_CLASS = {
    "Adam": optim.Adam,
    "SGD": optim.SGD
}