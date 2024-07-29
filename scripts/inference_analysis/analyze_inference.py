######################### READ: idiosyncratic path error #########################################
# TODO: FIX THIS PATH ERROR
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# This sets the current path to the parent directory of this file. I was getting annoyed at being cd into the wrong places.
# You shouldn't need this and can comment out this block.
##################################################################################################

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import scripts.config_analysis as config_analysis

def calculate_metrics(predictions, labels):
    metrics = {}
    for task in config_analysis.TASKS:
        metrics[task] = {
            'accuracy': accuracy_score(labels[task], predictions[task]),
            'confusion_matrix': confusion_matrix(labels[task], predictions[task]),
            'classification_report': classification_report(labels[task], predictions[task], output_dict=True)
        }
    return metrics

def plot_confusion_matrix(cm, task, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {task}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def analyze_results(results):
    predictions = results['predictions']
    labels = results['labels']
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    
    # Generate visualizations
    for task in config_analysis.TASKS:
        cm = metrics[task]['confusion_matrix']
        plot_confusion_matrix(cm, task, f"{config_analysis.RESULTS_DIR}/figures/{task}_confusion_matrix.png")
    
    return metrics

def main(results_path):
    # Load inference results
    results = load_inference_results(results_path)
    
    # Analyze results
    analysis_results = analyze_results(results)
    
    # Save analysis results
    save_analysis_results(analysis_results, f"{config_analysis.RESULTS_DIR}/processed/analysis_results.json")
    
    # Print summary
    for task in config_analysis.TASKS:
        print(f"\nResults for {task}:")
        print(f"Accuracy: {analysis_results[task]['accuracy']:.4f}")
        print("Classification Report:")
        print(classification_report(results['labels'][task], results['predictions'][task]))

if __name__ == "__main__":
    results_path = f"{config_analysis.RESULTS_DIR}/raw/{config_analysis.INFERENCE_RESULTS_FILE}"
    main(results_path)