import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from src.data_handling.simXRD_data_loader import create_data_loaders
from src.training.train import train
from src.models.CNNten import CNNten

# Initialize wandb
wandb.init(
    project="xrd-classification", 

    config={
    "batch_size": 32,
    "num_workers": 3,
    "learning_rate": 0.001,
    "num_epochs": 2,        # Reduced number of epochs for the test run
    "input_size": 3501,
    "num_classes": 230,
    "max_batches": 10,  # Limit the number of batches per epoch
    "val_subset": 100  # Number of samples to use for validation
})

# Access hyperparameters from wandb config
config = wandb.config

# Create data loaders
# create_data_loaders(train_path, val_path, test_path, batch_size=32, num_workers=3)
train_loader, val_loader, test_loader = create_data_loaders('data/train.db', 'data/val.db', 'data/test.db', config.batch_size, config.num_workers)

# Initialize model, loss, and optimizer
model = CNNten(input_size=config.input_size, num_classes=config.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# Log the model architecture
wandb.watch(model)

# Train the model
trained_model = train(model, train_loader, val_loader, criterion, optimizer, config.num_epochs)

# Save the trained model
torch.save(trained_model.state_dict(), 'trained_model.pth')
wandb.save('trained_model.pth')

print("Training completed. Model saved as 'trained_model.pth'.")
wandb.finish()