import torchmetrics
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def calculate_tpr_at_fpr(y_true, y_scores, fpr_target):
    """Calculate the TPR at a given FPR target using ROC curve data."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    if fpr_target in fpr:
        return tpr[np.where(fpr == fpr_target)[0][0]]
    else:
        return np.interp(fpr_target, fpr, tpr)

def evaluate_model_tpr_at_fpr(model, loader, device, num_classes, fpr_targets=[0.01, 0.1, 0.5]):
    """Evaluate model to find TPR at given FPR targets for each class in multiclass classification."""
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    predictions = []
    true_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = softmax(outputs).cpu().numpy()  # Get class probabilities
            predictions.append(probs)
            true_labels.append(Y_batch.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Calculate TPR at specified FPRs for each class
    tprs_at_fixed_fprs = {f"FPR {int(fpr*100)}%": [] for fpr in fpr_targets}

    # Iterate over each class
    for class_index in range(num_classes):
        y_true_class = true_labels[:, class_index]
        y_scores_class = predictions[:, class_index]
        
        # For each FPR target, calculate the TPR
        for fpr_target in fpr_targets:
            tpr = calculate_tpr_at_fpr(y_true_class, y_scores_class, fpr_target)
            tprs_at_fixed_fprs[f"FPR {int(fpr_target*100)}%"].append(tpr)

    return tprs_at_fixed_fprs

def evaluate_confidence_thresholding(model, loader, device, thresholds=[0.5, 0.9, 0.95, 0.99]):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    correct_counts = {thresh: 0 for thresh in thresholds}
    total_counts = {thresh: 0 for thresh in thresholds}
    total_above_threshold_counts = {thresh: 0 for thresh in thresholds}

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            outputs = model(X_batch)
            probs = softmax(outputs)
            max_probs, preds = torch.max(probs, dim=1)

            for thresh in thresholds:
                above_thresh = max_probs > thresh
                total_counts[thresh] += above_thresh.sum().item()
                correct_counts[thresh] += (preds[above_thresh] == torch.argmax(Y_batch[above_thresh], axis=1)).sum().item()
                total_above_threshold_counts[thresh] += len(max_probs[max_probs > thresh])

    confidence_accuracy = {thresh: (correct_counts[thresh] / total_counts[thresh]) if total_counts[thresh] > 0 else 0 for thresh in thresholds}
    proportion_above_threshold = {thresh: (total_above_threshold_counts[thresh] / len(loader.dataset)) for thresh in thresholds}

    return confidence_accuracy, proportion_above_threshold