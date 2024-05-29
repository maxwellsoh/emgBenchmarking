import torchmetrics
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import wandb

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
            if isinstance(outputs, dict):
                outputs = outputs['logits']
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
            if isinstance(outputs, dict):
                outputs = outputs['logits']
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

def evaluate_model_on_test_set(model, test_loader, device, numGestures, criterion, utils, gesture_labels, testrun_foldername, args, formatted_datetime):
    # Testing
    # Initialize metrics for testing with macro and micro averaging
    test_macro_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=numGestures, average="macro").to(device)
    test_macro_precision_metric = torchmetrics.Precision(task="multiclass", num_classes=numGestures, average="macro").to(device)
    test_macro_recall_metric = torchmetrics.Recall(task="multiclass", num_classes=numGestures, average="macro").to(device)
    test_macro_f1_score_metric = torchmetrics.F1Score(task="multiclass", num_classes=numGestures, average="macro").to(device)
    test_macro_top5_acc_metric = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=numGestures, average="macro").to(device)
    test_micro_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=numGestures, average="micro").to(device)
    test_micro_top5_acc_metric = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=numGestures, average="micro").to(device)

    # Assuming model, criterion, device, and test_loader are defined
    model.eval()
    test_loss = 0.0

    # Reset test metrics
    test_macro_acc_metric.reset()
    test_macro_precision_metric.reset()
    test_macro_recall_metric.reset()
    test_macro_f1_score_metric.reset()
    test_macro_top5_acc_metric.reset()
    test_micro_acc_metric.reset()
    test_micro_top5_acc_metric.reset()

    pred = []
    true = []

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device).to(torch.float32)
            Y_batch = Y_batch.to(device).to(torch.float32)
            Y_batch_long = torch.argmax(Y_batch, dim=1)

            output = model(X_batch)
            if isinstance(output, dict):
                output = output['logits']
            pred.extend(torch.argmax(output, dim=1).cpu().detach().numpy())
            true.extend(Y_batch_long.cpu().detach().numpy())

            test_loss += criterion(output, Y_batch).item()
            test_macro_acc_metric(output, Y_batch_long)
            test_macro_precision_metric(output, Y_batch_long)
            test_macro_recall_metric(output, Y_batch_long)
            test_macro_f1_score_metric(output, Y_batch_long)
            test_macro_top5_acc_metric(output, Y_batch_long)
            test_micro_acc_metric(output, Y_batch_long)
            test_micro_top5_acc_metric(output, Y_batch_long)

    # Calculate average loss and metrics
    test_loss /= len(test_loader)
    test_macro_acc = test_macro_acc_metric.compute()
    test_macro_precision = test_macro_precision_metric.compute()
    test_macro_recall = test_macro_recall_metric.compute()
    test_macro_f1_score = test_macro_f1_score_metric.compute()
    test_macro_top5_acc = test_macro_top5_acc_metric.compute()
    test_micro_acc = test_micro_acc_metric.compute()
    test_micro_top5_acc = test_micro_top5_acc_metric.compute()
    tpr_results = evaluate_model_tpr_at_fpr(model, test_loader, device, numGestures)
    confidence_levels, proportions_above_confidence_threshold = evaluate_confidence_thresholding(model, test_loader, device)

    print(f"Test Loss: {test_loss:.4f} | Test Macro Accuracy: {test_macro_acc:.4f} | Test Micro Accuracy: {test_micro_acc:.4f}")
    print(f"Test Macro Precision: {test_macro_precision:.4f} | Test Macro Recall: {test_macro_recall:.4f} | Test Macro F1 Score: {test_macro_f1_score:.4f} | Test Macro Top-5 Accuracy: {test_macro_top5_acc:.4f}")
    print(f"Test Micro Top-5 Accuracy: {test_micro_top5_acc:.4f}")
    # for fpr, tprs in tpr_results.items():
    #     print(f"TPR at {fpr}: {', '.join(f'{tpr:.4f}' for tpr in tprs)}")
    # for confidence_level, acc in confidence_levels.items():
    #     print(f"Accuracy at {confidence_level} confidence level: {acc:.4f}")

    wandb.log({
        "test/Test Loss": test_loss,
        "test/Test Macro Accuracy": test_macro_acc,
        "test/Test Micro Accuracy": test_micro_acc,
        "test/Test Macro Precision": test_macro_precision,
        "test/Test Macro Recall": test_macro_recall,
        "test/Test Macro F1": test_macro_f1_score,
        "test/Test Macro Top-5 Accuracy": test_macro_top5_acc,
        "test/Test Micro Top-5 Accuracy": test_micro_top5_acc,
        # **{f"tpr_at_fixed_fpr/Test TPR at {fpr} FPR - Gesture {idx}": tpr for fpr, tprs in tpr_results.items() for idx, tpr in enumerate(tprs)},
        **{f"tpr_at_fixed_fpr/Average Test TPR at {fpr} FPR": np.mean(tprs) for fpr, tprs in tpr_results.items()},
        **{f"confidence_level_accuracies/Test Accuracy at {int(confidence_level*100)}% confidence": acc for confidence_level, acc in confidence_levels.items()},
        **{f"proportion_above_confidence_threshold/Test Proportion above {int(confidence_level*100)}% confidence": prop for confidence_level, prop in proportions_above_confidence_threshold.items()}
    })

    # Confusion Matrix
    # Plot and log confusion matrix in wandb


