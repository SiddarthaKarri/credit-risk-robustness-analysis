import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

def cost_weighted_error(y_true, y_pred, cost_fn_fp=1, cost_fn_fn=5):
    """
    Calculate cost-weighted error.
    Args:
        y_true: True labels (0: Good, 1: Default)
        y_pred: Predicted labels
        cost_fn_fp: Cost of False Positive
        cost_fn_fn: Cost of False Negative
    Returns:
        float: Weighted error cost
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * cost_fn_fp) + (fn * cost_fn_fn)
    return total_cost / len(y_true)

def evaluate_model(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'cost_error': cost_weighted_error(y_true, y_pred)
    }
