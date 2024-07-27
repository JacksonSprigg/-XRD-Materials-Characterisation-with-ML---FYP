import torch
import torch.nn as nn
import wandb
import datetime
import config
from src.data_handling.simXRD_data_loader import create_data_loaders
from src.training.train_single_spg import train_single_spg
from src.training.train_multi_spg_cryssystem_blt_element import train_multi_spg_cryssystem_blt_element

def setup_wandb():
    wandb.require("core") # This line maybe fixes a retry upload bug I was having. See: https://github.com/wandb/wandb/issues/4929
    return wandb.init(
        project=config.WANDB_PROJECT_NAME, 
        dir=config.WANDB_SAVE_DIR, 
        config={
            "model_type": config.MODEL_TYPE,
            "multi_task": config.MULTI_TASK,
            "criterion_type": config.CRITERION_TYPE,
            "optimizer_type": config.OPTIMIZER_TYPE,
            "batch_size": config.BATCH_SIZE,
            "num_workers": config.NUM_WORKERS,       
            "learning_rate": config.LEARNING_RATE,
            "num_epochs": config.NUM_EPOCHS,
        }
    )

def setup_model():
    # Initialize the model, loss function, and optimiser
    model_class = config.MODEL_CLASS[config.MODEL_TYPE]
    model = model_class()
    
    if config.MULTI_TASK:
        criterion = config.MULTI_TASK_CRITERIA
    else:
        criterion_class = config.CRITERION_CLASS[config.CRITERION_TYPE]
        criterion = criterion_class()
    
    optimizer_class = config.OPTIMIZER_CLASS[config.OPTIMIZER_TYPE]
    optimizer = optimizer_class(model.parameters(), lr=config.LEARNING_RATE)
    
    return model, criterion, optimizer

# THIS NEED TO BE UPDATED, THE METHOD IS NOT STANDARD
# I AM NOT SURE HOW WELL IT WILL HANDLE MULTIPLE GPUs
def setup_device(model):
    # Setup GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device

def save_model(model, final_metrics, multi_task=False):
    # Save model with a unique, comprehensive string of important data
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if multi_task:
        metrics_str = "_".join([f"{k}_{v:.4f}" for k, v in final_metrics.items()])
    else:
        metrics_str = f"test_loss_{final_metrics['test_loss']:.4f}_test_accuracy_{final_metrics['test_accuracy']:.2f}"

    model_name = f"{config.MODEL_TYPE}_{metrics_str}_data_{config.NAME_OF_DATA_USED}_time_{current_time}.pth"
    full_path = f'{config.MODEL_SAVE_DIR}/{model_name}'
    torch.save(model.state_dict(), full_path)
    return full_path, model_name

def main():
    # Start WandB
    wandb_run = setup_wandb()

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config.TRAIN_DATA, config.VAL_DATA, config.TEST_DATA, 
        config.BATCH_SIZE, config.NUM_WORKERS
    )

    # Setup model, loss, and optimizer
    model, criterion, optimizer = setup_model()

    # Setup device
    model, device = setup_device(model)

    # Log the model architecture
    if config.WANDB_LOG_ARCHITECTURE:
        wandb.watch(model)

    # Train the model depending on task
    if config.MULTI_TASK:
        trained_model, final_metrics = train_multi_spg_cryssystem_blt_element(
            model, train_loader, val_loader, test_loader, criterion, optimizer, 
            device, config.NUM_EPOCHS
        )
    else:
        trained_model, test_loss, test_accuracy = train_single_spg(
            model, train_loader, val_loader, test_loader, criterion, optimizer, 
            device, config.NUM_EPOCHS
        )
        final_metrics = {'test_loss': test_loss, 'test_accuracy': test_accuracy}

    # Save the model
    save_path, model_name = save_model(trained_model, final_metrics, multi_task=config.MULTI_TASK)

    if config.SAVE_MODEL_TO_WANDB_SERVERS:
        wandb.save(save_path)

    print(f"Training completed. Model saved as '{model_name}'.")
    wandb_run.finish()

if __name__ == "__main__":
    main()