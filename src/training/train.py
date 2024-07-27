import torch
import wandb
from tqdm import tqdm

def train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            
            # Reshape data: [batch_size, 3501] -> [batch_size, 1, 3501]
            # Move to device
            data = data.unsqueeze(1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate on Val
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        # Log metrics to wandb every epoch
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
        
        print(f'Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Finish with an evaluate on the test set
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    
    print(f'Test loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    })

    return model, test_loss, test_accuracy

# Used for both val and test data_sets
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluation"):
            
            # Reshape data: [batch_size, 3501] -> [batch_size, 1, 3501]
            # Move to device
            data = data.unsqueeze(1).to(device)
            target = target.to(device)
            
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy