import numpy as np
import pandas as pd

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculates the confusion matrix for true and predicted class labels.
    
    Rows represent Actual / True classes (i), 
    Columns represent Predicted classes (j).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(classes)
    
    # Map class labels to contiguous indices (0, 1, ..., num_classes-1)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
        
    return cm

def calc_recall_precision(cm: np.ndarray) -> tuple[list[float], list[float]]:
    """
    Calculates Per-Class Recall and Precision from a Confusion Matrix.
    
    - Recall (Sensitivity) = TP / (TP + FN) = TP / (Actual Positives in Row)
    - Precision (PPV)       = TP / (TP + FP) = TP / (Predicted Positives in Column)
    """
    recall_per_class = []
    precision_per_class = []
    num_classes = cm.shape[0]
    
    for cls in range(num_classes):
        true_positive = cm[cls, cls]
        # False Positives: sum of predicted column minus true positive
        false_positive = np.sum(cm[:, cls]) - true_positive
        # False Negatives: sum of actual row minus true positive
        false_negative = np.sum(cm[cls, :]) - true_positive
        
        # Guard against zero-division (e.g., if a class is never predicted or never appears)
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        
        recall_per_class.append(float(recall))
        precision_per_class.append(float(precision))
        
    return recall_per_class, precision_per_class

def main():
    print("=== Binary Classification Example (Customer Churn: 0=Retained, 1=Churned) ===")
    y_true_binary = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    y_pred_binary = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1])
    
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    recall_bin, prec_bin = calc_recall_precision(cm_binary)
    
    print("Confusion Matrix (Rows: Actual, Cols: Predicted):")
    print(cm_binary)
    print("\nPer-Class Metrics:")
    for cls_idx, (r, p) in enumerate(zip(recall_bin, prec_bin)):
        print(f"  Class {cls_idx} -> Recall: {r:.4f}, Precision: {p:.4f}")
        
    print("\n" + "="*70 + "\n")
    
    print("=== Multi-Class Example (3 Classes: 0, 1, 2) ===")
    y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 2, 2, 1])
    y_pred_multi = np.array([0, 2, 2, 0, 1, 1, 0, 2, 2, 1])
    
    cm_multi = confusion_matrix(y_true_multi, y_pred_multi)
    recall_multi, prec_multi = calc_recall_precision(cm_multi)
    
    print("Confusion Matrix:")
    print(cm_multi)
    print("\nPer-Class Metrics:")
    for cls_idx, (r, p) in enumerate(zip(recall_multi, prec_multi)):
        print(f"  Class {cls_idx} -> Recall: {r:.4f}, Precision: {p:.4f}")

if __name__ == "__main__":
    main()
