import torch
import torch.nn as nn
import torch.optim as optim
from src.data_handling.simXRD_data_loader import create_data_loaders
from src.training.train import train
from src.models.CNNten import CNNten

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
input_size = 100
num_classes = 230

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders('data/train.db', 'data/val.db', 'data/test.db', batch_size)

# Initialize model, loss, and optimizer
model = CNNten(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
trained_model = train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Save the trained model
torch.save(trained_model.state_dict(), 'trained_model.pth')

print("Training completed. Model saved as 'trained_model.pth'.")