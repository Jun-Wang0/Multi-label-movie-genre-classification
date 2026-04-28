from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
import torch


def evaluate_metrics(preds, labels):
    """
    Input:
        preds: Tensor or ndarray, shape (N, C), each element is 0/1
        labels: Same as above
    Output:
        dict, including accuracy, f1, precision, recall, and hamming_loss
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Per-label average accuracy: Average across labels first, then across samples
    acc = (preds == labels).mean(axis=0).mean()

    # Other metrics
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)

    macro_prec = precision_score(labels, preds, average='macro', zero_division=0)
    micro_prec = precision_score(labels, preds, average='micro', zero_division=0)

    macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
    micro_recall = recall_score(labels, preds, average='micro', zero_division=0)

    hamming = hamming_loss(labels, preds)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_prec,
        "micro_precision": micro_prec,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "hamming_loss": hamming
    }