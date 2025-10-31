"""
Visualization + clinical report
- Prints classification report (Healthy/Cancer)
- Plots confusion matrix @ optimal threshold
- Plots training curves (AUC, Loss)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Cancer'],
                yticklabels=['Healthy', 'Cancer'])
    plt.title('Confusion Matrix (Optimal Threshold)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history_dict):
    # Two side-by-side plots: AUC and Loss
    plt.figure(figsize=(12,5))

    # AUC
    plt.subplot(1,2,1)
    plt.plot(history_dict['auc'], label='Train AUC')
    plt.plot(history_dict['val_auc'], label='Val AUC')
    plt.title('AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Load predictions + history
    pred = np.load('predictions.npz', allow_pickle=True)
    y_test = pred['y_test']
    y_pred_optimal = pred['y_pred_optimal']
    optimal_threshold = float(pred['threshold'])

    history = np.load('training_history.npy', allow_pickle=True).item()

    # Clinical report
    print("\nCLINICAL PERFORMANCE")
    print(classification_report(y_test, y_pred_optimal, target_names=['Healthy', 'Cancer']))
    print(f"Optimal Threshold: {optimal_threshold:.3f}")

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred_optimal)

    # Learning curves
    plot_learning_curves(history)

if __name__ == "__main__":
    main()