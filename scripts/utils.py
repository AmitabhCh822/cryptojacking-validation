"""
Utility functions for Cryptojacking Validation Experiment
Author: Amitabh Chakravorty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import time
import pickle
import os


def load_processed_data(dataset_name, data_path='data/processed'):
    """
    Load preprocessed data for a given dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset ('ds2os' or 'nsl_kdd')
    data_path : str
        Path to processed data directory
        
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test)
    """
    X_train = np.load(f'{data_path}/X_train_{dataset_name}.npy')
    X_test = np.load(f'{data_path}/X_test_{dataset_name}.npy')
    y_train = np.load(f'{data_path}/y_train_{dataset_name}.npy')
    y_test = np.load(f'{data_path}/y_test_{dataset_name}.npy')
    
    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
    }
    
    return metrics


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, 
                       dataset_name, save_model=True, models_path='models'):
    """
    Train a model and collect comprehensive metrics.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to train
    model_name : str
        Name of the model
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    dataset_name : str
        Name of the dataset
    save_model : bool
        Whether to save the trained model
    models_path : str
        Path to save models
        
    Returns:
    --------
    tuple : (results_dict, confusion_matrix)
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name} on {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Prediction
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Calculate metrics
    metrics = evaluate_model(y_test, y_pred)
    
    # Display results
    print(f"\nResults:")
    print(f"  Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-Score:     {metrics['f1_score']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  Train time:   {train_time:.2f}s")
    print(f"  Inference:    {inference_time:.4f}s")
    
    # Save model
    if save_model:
        os.makedirs(models_path, exist_ok=True)
        model_filename = f'{models_path}/{model_name.replace(" ", "_")}_{dataset_name}.pkl'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Model saved: {model_filename}")
    
    # Compile results
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'train_time_sec': train_time,
        'inference_time_sec': inference_time,
        'samples_train': len(y_train),
        'samples_test': len(y_test),
        'features': X_train.shape[1],
        **{k: v for k, v in metrics.items() if k not in ['accuracy', 'f1_score', 'precision', 'recall']}
    }
    
    return results, confusion_matrix(y_test, y_pred)


def plot_confusion_matrices(cms_list, dataset_name, save_path='results/figures'):
    """
    Plot confusion matrices for multiple models.
    
    Parameters:
    -----------
    cms_list : list of tuples
        List of (model_name, confusion_matrix) tuples
    dataset_name : str
        Name of the dataset
    save_path : str
        Path to save figures
    """
    n_models = len(cms_list)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    for idx, (model_name, cm) in enumerate(cms_list):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_xticklabels(['Normal', 'Attack'])
        axes[idx].set_yticklabels(['Normal', 'Attack'])
    
    # Hide unused subplots
    for idx in range(len(cms_list), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Confusion Matrices - {dataset_name.upper()}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/confusion_matrices_{dataset_name}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance_comparison(results_df, save_path='results/figures'):
    """
    Create performance comparison visualizations.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing model results
    save_path : str
        Path to save figures
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    datasets = results_df['dataset'].unique()
    x = np.arange(len(results_df['model'].unique()))
    width = 0.35
    
    for i, dataset in enumerate(datasets):
        subset = results_df[results_df['dataset'] == dataset]
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, subset['accuracy'], width, label=dataset.upper())
        
        for bar, acc in zip(bars, subset['accuracy']):
            ax.annotate(f'{acc:.2%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'].unique(), rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0.9, 1.02)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Accuracy Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_data = results_df.pivot_table(
        values='accuracy',
        index='model',
        columns='dataset'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn',
                vmin=0.85, vmax=1.0, cbar_kws={'label': 'Accuracy'},
                linewidths=0.5, ax=ax)
    ax.set_title('Accuracy Heatmap: Models × Datasets', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(results_df):
    """
    Print summary statistics from results.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing model results
    """
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print(f"\nOverall Statistics:")
    print(f"  Total models trained: {len(results_df)}")
    print(f"  Datasets: {results_df['dataset'].nunique()}")
    
    print(f"\nBest Performers:")
    best_acc = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"  Highest Accuracy: {best_acc['model']} on {best_acc['dataset'].upper()} - {best_acc['accuracy']:.4f}")
    
    best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    print(f"  Highest F1-Score: {best_f1['model']} on {best_f1['dataset'].upper()} - {best_f1['f1_score']:.4f}")
    
    fastest = results_df.loc[results_df['train_time_sec'].idxmin()]
    print(f"  Fastest Training: {fastest['model']} on {fastest['dataset'].upper()} - {fastest['train_time_sec']:.2f}s")
    
    print(f"\nAverage Performance:")
    print(f"  Mean Accuracy:  {results_df['accuracy'].mean():.4f}")
    print(f"  Mean F1-Score:  {results_df['f1_score'].mean():.4f}")
    print(f"  Mean Precision: {results_df['precision'].mean():.4f}")
    print(f"  Mean Recall:    {results_df['recall'].mean():.4f}")
    
    print(f"\nComputational Summary:")
    print(f"  Total training time: {results_df['train_time_sec'].sum():.2f}s")
    print(f"  Average per model:   {results_df['train_time_sec'].mean():.2f}s")


if __name__ == "__main__":
    print("Utility functions loaded successfully.")
    print("Import this module to use helper functions for the validation experiment.")
