######################### READ: idiosyncratic path error #########################################
# TODO: FIX THIS PATH ERROR
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# This sets the current path to the parent directory of this file. I was getting annoyed at being cd into the wrong places.
# You shouldn't need this and can comment out this block.
##################################################################################################

import torch

# Import config
import scripts.config_inference as config_inference

# Import functions
from src.data_handling.simXRD_data_loader import create_inference_loader
from scripts.main_training import setup_device


def load_model(model_path):
    model_class = config_inference.MODEL_CLASS[config_inference.MODEL_TYPE]
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    return model

def run_inference(model, test_loader, device):
    model.eval()
    all_predictions = {task: [] for task in config_inference.TASKS}
    all_labels = {task: [] for task in config_inference.TASKS}

    with torch.no_grad():
        for batch in test_loader:
            intensity, spg, crysystem, blt, composition = [t.to(device) for t in batch]
            
            if config_inference.MULTI_TASK:
                outputs = model(intensity.unsqueeze(1))
                
                for task in config_inference.TASKS:
                    preds = outputs[task].argmax(dim=1)
                    all_predictions[task].extend(preds.cpu().numpy())
                    
                all_labels['spg'].extend(spg.cpu().numpy())
                all_labels['crysystem'].extend(crysystem.cpu().numpy())
                all_labels['blt'].extend(blt.cpu().numpy())
                all_labels['composition'].extend(composition.cpu().numpy())
            else:
                output = model(intensity.unsqueeze(1))
                preds = output.argmax(dim=1)
                all_predictions['spg'].extend(preds.cpu().numpy())
                all_labels['spg'].extend(spg.cpu().numpy())

    return all_predictions, all_labels

def main(model_path):
    # Load the model
    model = load_model(model_path)
    
    # Setup device
    model, device = setup_device(model)

    # Create data loader for test data
    test_loader = create_inference_loader(
        config_inference.INFERENCE_DATA, 
        config_inference.BATCH_SIZE, 
        config_inference.NUM_WORKERS
    )

    # Run inference
    predictions, labels = run_inference(model, test_loader, device)

    # Save results
    results = {
        'predictions': predictions,
        'labels': labels
    }

    return print(results)
    

if __name__ == "__main__":
    model_path = f'{config_inference.MODEL_SAVE_DIR}/{config_inference.MODEL_NAME}'
    main(model_path)