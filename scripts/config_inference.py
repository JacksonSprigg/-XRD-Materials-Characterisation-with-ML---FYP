import os

######################### READ: idiosyncratic path error #########################################
# TODO: FIX THIS PATH ERROR
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# This sets the current path to the parent directory. I was getting annoyed at being cd into the wrong places.
# You shouldn't need this and can comment out this block.
##################################################################################################

import torch.nn as nn

# Import models
from src.models.CNNten import CNNten, CNNten_multi_task
from src.models.smallFCN import smallFCN, smallFCN_multi_task
from src.models.MLPten import MLPten
from src.models.Jackson import Jackson

# Paths
DATA_DIR = '/monfs01/projects/ys68/XRD_ML/simXRD_partial_data'
MODEL_SAVE_DIR = '/monfs01/projects/ys68/XRD_ML/trained_models'

# Data
INFERENCE_DATA = os.path.join(DATA_DIR, 'test.db')

# Model settings
MODEL_TYPE = "smallFCN_multi_task"   # Options: "CNNten", CNNten_multi_task", "smallFCN", "smallFCN_multi_task", "Jackson"
MULTI_TASK = True                    # Note, you can use multi-task models on single task inference!
MODEL_NAME = "smallFCN_multi_task_spg_acc_94.6500_20240728_235958.pth" # Copy the model name as a string. For example: "smallFCN_multi_task_spg_acc_94.6500_20240728_235958.pth"

# Inference settings
BATCH_SIZE = 32
NUM_WORKERS = 8

# Task settings
TASKS = ['spg', 'crysystem', 'blt', 'composition'] if MULTI_TASK else ['spg']

# Label information
LABELS = {
    'spg': list(range(230)),
    'crysystem': list(range(7)),
    'blt': ['P', 'I', 'F', 'A', 'B', 'C'],
    'composition': list(range(118))
}

####################################################################################
############# DON'T TOUCH - Classes contain the options for above ##################
MODEL_CLASS = {
    "CNNten": CNNten,
    "CNNten_multi_task": CNNten_multi_task,
    "smallFCN": smallFCN,
    "smallFCN_multi_task": smallFCN_multi_task,
    "MLPten": MLPten,
    "Jackson": Jackson
}