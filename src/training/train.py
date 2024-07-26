import torch
import wandb
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):

            # Reshape data: [batch_size, 3501] -> [batch_size, 1, 3501]
            data = data.unsqueeze(1)

            # Move data and target to the appropriate device
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")):
                
                # Reshape data: [batch_size, 3501] -> [batch_size, 1, 3501]
                data = data.unsqueeze(1)

                # Move data and target to the appropriate device
                data = data.to(device)
                target = target.to(device)
                
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= total
        accuracy = 100. * correct / total
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy
        })
        
        print(f'Epoch {epoch+1}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model, val_loss, accuracy