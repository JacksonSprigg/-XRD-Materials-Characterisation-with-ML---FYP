import torch
import wandb
from tqdm import tqdm

def train_multi_spg_cryssystem_blt_element(model, train_loader, val_loader, test_loader, criteria, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_losses = {task: 0.0 for task in criteria.keys()}
        
        for batch_idx, (data, spg, crysystem, blt, composition) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            data = data.unsqueeze(1).to(device)
            targets = {
                'spg': spg.to(device),
                'crysystem': crysystem.to(device),
                'blt': blt.to(device),
                'composition': composition.to(device)
            }
            
            optimizer.zero_grad()
            outputs = model(data)

            losses = {task: criteria[task](outputs[task], targets[task]) for task in criteria.keys()}
            total_loss = sum(losses.values())
            
            total_loss.backward()
            optimizer.step()
            
            for task in losses:
                train_losses[task] += losses[task].item()
        
        for task in train_losses:
            train_losses[task] /= len(train_loader)
        
        # Evaluate on Val
        val_metrics = evaluate_multi_task(model, val_loader, criteria, device)
        
        # Log metrics to wandb every epoch
        wandb_log = {f"train_{task}_loss": loss for task, loss in train_losses.items()}
        wandb_log.update({f"val_{k}": v for k, v in val_metrics.items()})
        wandb.log(wandb_log)
        
        print(f'Epoch {epoch+1}:')
        for task, loss in train_losses.items():
            print(f'Train {task} loss: {loss:.4f}')
        for k, v in val_metrics.items():
            print(f'Val {k}: {v:.4f}')

    # Finish with an evaluate on the test set
    test_metrics = evaluate_multi_task(model, test_loader, criteria, device)
    
    print('Test Results:')
    for k, v in test_metrics.items():
        print(f'Test {k}: {v:.4f}')
    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

    return model, test_metrics

def evaluate_multi_task(model, data_loader, criteria, device):
    model.eval()
    total_losses = {task: 0.0 for task in criteria.keys()}
    correct = {task: 0 for task in ['spg', 'crysystem', 'blt']}
    total = 0
    
    with torch.no_grad():
        for data, spg, crysystem, blt, composition in tqdm(data_loader, desc="Evaluation"):
            data = data.unsqueeze(1).to(device)
            targets = {
                'spg': spg.to(device),
                'crysystem': crysystem.to(device),
                'blt': blt.to(device),
                'composition': composition.to(device)
            }
            
            outputs = model(data)
            
            losses = {task: criteria[task](outputs[task], targets[task]) for task in criteria.keys()}
            
            for task in losses:
                total_losses[task] += losses[task].item()
            
            for task in ['spg', 'crysystem', 'blt']:
                pred = outputs[task].argmax(dim=1, keepdim=True)
                correct[task] += pred.eq(targets[task].view_as(pred)).sum().item()
            
            total += data.size(0)
    
    avg_losses = {task: total_losses[task] / len(data_loader) for task in total_losses}
    accuracies = {task: 100. * correct[task] / total for task in ['spg', 'crysystem', 'blt']}
    
    metrics = {f"{task}_loss": loss for task, loss in avg_losses.items()}
    metrics.update({f"{task}_accuracy": acc for task, acc in accuracies.items()})
    
    return metrics