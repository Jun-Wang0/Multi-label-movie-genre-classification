import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import logging
from sklearn.metrics import classification_report

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.BILSTMModel import BiLSTMClassifier
from Model.RobertaMultiInputMultiLabelClassifier import RobertaMultiInputMultiLabelClassifier
from Model.DebertaMultiInputMultiLabelClassifier import DebertaMultiInputMultiLabelClassifier
from TorchDataset.EnsembleDataset import EnsembleDataset
from metrixCaler import evaluate_metrics


# ==== Suppress transformers warnings ====
logging.set_verbosity_error()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_multiclass_roc(y_true, y_score, num_classes, save_path=None):
    """
    Plots the multiclass ROC curve and optionally saves the image.
    
    Args:
    y_true: Ground truth labels of length N (integer classes).
    y_score: N x C probability array, where C is the number of classes.
    num_classes: Number of classes.
    save_path: If not None, saves the plot to this path (e.g., 'roc_curve.png').
    """
    y_test = label_binarize(y_true, classes=range(num_classes))
    y_test = np.array(y_test)
    y_score = np.array(y_score)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True)

    # Save the image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC plot saved to: {save_path}")

    # plt.show()

# ==== Load sub-models ====
def load_model(path='bilstm_model.pt', device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model = BiLSTMClassifier(num_classes=27).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def strict_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate strict accuracy: all labels must match perfectly to be considered correct.
    
    Args:
        preds (torch.Tensor): Predictions, shape (batch_size, num_labels), values 0 or 1.
        labels (torch.Tensor): True labels, same shape as preds.
        
    Returns:
        float: Strict exact match accuracy.
    """
    correct = (preds == labels).all(dim=1).float()  # Check if every label in a row matches
    acc = correct.mean().item()
    return acc

# ==== Ensemble Model Architecture ====
class EnsembleClassifier(nn.Module):
    def __init__(self, bilstm, roberta, deberta, hidden_dim=512, num_labels=27):
        super().__init__()
        self.bilstm = bilstm.eval()
        self.roberta = roberta.eval()
        self.deberta = deberta.eval()

        self.bilstm_output_dim = 27
        self.roberta_output_dim = 27
        self.deberta_output_dim = 27

        total_dim = self.bilstm_output_dim + self.roberta_output_dim + self.deberta_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, bilstm_input, roberta_input, deberta_input):
        with torch.no_grad():
            v1 = self.bilstm(bilstm_input)
            v2 = self.roberta(
                roberta_input['input1'], roberta_input['input2'], roberta_input['input3']
            )
            v3 = self.deberta(
                deberta_input['input1'], deberta_input['input2'], deberta_input['input3']
            )
        
        fused = torch.cat([v1, v2, v3], dim=1)
        logits = self.classifier(fused)
        return logits


# ==== Test Function ====
def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Added tracking for raw probabilities for the ROC curve

    with torch.no_grad():
        for batch in dataloader:
            bilstm_input = batch['lstm_input'].to(device)
            roberta_input = {
                'input1': {k: v.to(device) for k, v in batch['roberta_input']['input1'].items()},
                'input2': {k: v.to(device) for k, v in batch['roberta_input']['input2'].items()},
                'input3': {k: v.to(device) for k, v in batch['roberta_input']['input3'].items()},
            }
            deberta_input = {
                'input1': {k: v.to(device) for k, v in batch['deberta_input']['input1'].items()},
                'input2': {k: v.to(device) for k, v in batch['deberta_input']['input2'].items()},
                'input3': {k: v.to(device) for k, v in batch['deberta_input']['input3'].items()},
            }
            labels = batch['labels'].to(device)

            logits = model(bilstm_input, roberta_input, deberta_input)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())  # Store valid sigmoid probabilities

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    probs_tensor = torch.cat(all_probs)  # Compile the continuous probabilities
    
    metrix = evaluate_metrics(preds, labels)
    print(metrix)
    
    acc = strict_accuracy(preds, labels)
    print(f'acc = {acc}')
    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, zero_division=0))

    # 🎯 Plot ROC Curve (now uses correct sigmoid probabilities instead of softmaxing binary predictions)
    plot_multiclass_roc(
        y_true=labels.cpu().numpy(),
        y_score=probs_tensor.cpu().numpy(), 
        num_classes=probs_tensor.size(1),
        save_path='roc_curve.png'
    )
    
    return preds


# ==== Main Entry Point ====
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load sub-models
    bilstm = load_model('../weights/bilstm_model_0.9.pt', device=device)
    roberta = RobertaMultiInputMultiLabelClassifier()
    roberta.load_state_dict(torch.load('../weights/best_model_sr.pt'))
    deberta = DebertaMultiInputMultiLabelClassifier()
    deberta.load_state_dict(torch.load('../weights/best_model_sd.pt'))

    # Build the ensemble model and load weights
    model = EnsembleClassifier(bilstm, roberta, deberta).to(device)
    model.load_state_dict(torch.load('../weights/best_ensemble.pt', map_location=device))
    model.eval()

    # Load test data
    test_dataset = EnsembleDataset(csv_file='/root/autodl-tmp/MovieLabeling/datasets/test_data.csv')
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Start testing
    test(model, test_loader, device)


if __name__ == '__main__':
    main()
