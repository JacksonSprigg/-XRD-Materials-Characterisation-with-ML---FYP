########## Comment this out if you are having path errors #########################
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
##################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import datetime
from src.data_handling.simXRD_data_loader import create_data_loaders
from src.training.train import train
from src.models.CNNten import CNNten

# Name the model:
model_type = "CNNten"

# Initialize wandb
wandb.init(
    project="test-run", 

    config={
    "batch_size": 32,
    "num_workers": 3,
    "learning_rate": 0.001,
    "num_epochs": 2,        # Reduced number of epochs for the test run
    "max_batches": 10,  # Limit the number of batches per epoch
    "val_subset": 100  # Number of samples to use for validation
})

# Access hyperparameters from wandb config
config = wandb.config

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders('simXRD_partial_data/train.db', 'simXRD_partial_data/val.db', 'simXRD_partial_data/test.db', config.batch_size, config.num_workers)

# Initialize model, loss, and optimizer
model = CNNten()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Log the model architecture
wandb.watch(model)

# Train the model
trained_model, final_loss = train(model, train_loader, val_loader, criterion, optimizer, config.num_epochs)

# Create a unique model name including the model type and loss
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{model_type}_loss{final_loss:.4f}_{current_time}.pth"

# Save the trained model
torch.save(trained_model.state_dict(), f'trained_models/{model_name}')
wandb.save(f'trained_models/{model_name}')

print(f"Training completed. Model saved as '{model_name}'.")
wandb.finish()