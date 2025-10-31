"""
Prediction + optimal-threshold selection (F1)
- Loads best_densenet.keras
- Scores y_pred on test set
- Picks threshold by maximizing F1 via precision-recall curve
- Saves y_test, y_pred_optimal, and chosen threshold
"""

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_curve

def main():
    # Load trained model (compiled custom loss not needed for inference)
    model = load_model('best_densenet.keras', compile=False)

    # Load test set
    data = np.load('test_data.npz', allow_pickle=True)
    X_test, y_test = data['X_test'], data['y_test']

    # Predict probabilities
    y_prob = model.predict(X_test, verbose=0)

    # Pick threshold that maximizes F1
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
    best_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[max(best_idx, 0)] if thresholds.size > 0 else 0.5

    # Binarize
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

    # Save predictions for visualization
    np.savez('predictions.npz',
             y_test=y_test,
             y_prob=y_prob,
             y_pred_optimal=y_pred_optimal,
             threshold=optimal_threshold)

    print("Saved predictions.npz")
    print(f"   Optimal threshold (F1): {optimal_threshold:.3f}")
    print(f"   Test samples: {len(y_test)}")

if __name__ == "__main__":
    main()