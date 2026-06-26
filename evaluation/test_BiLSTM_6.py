import torch
import torch.nn as nn

import sys
import os
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.BILSTMModel import BiLSTMClassifier
from TorchDataset.RobertaTextDataset import RobertaTextDataset
from metrixCaler import evaluate_metrics


def collate_fn(batch):
    all_inputs = []
    all_labels = []

    for item in batch:
        # emb_list = [
        #     item['title_embedding'],
        #     item['summaried_embedding'],
        #     item['synopsis_embedding'],
        #     item['title_tf'],
        #     item['summaried_tf'],
        #     item['synopsis_tf'],
        #     item['multi_embedding'],
        # ]
        # emb_tensor = torch.stack(emb_list)
        all_inputs.append(item['embeddings'])

        all_labels.append(item['labels'])  # Multi-label vector

    inputs_tensor = torch.stack(all_inputs)  # (batch, 7, 1024)
    labels_tensor = torch.stack(all_labels)  # (batch, 27)
    return inputs_tensor, labels_tensor

def load_model(path='bilstm_model.pt', device='cpu'):
    checkpoint = torch.load(path, map_location=device)

    if checkpoint['model_class'] == 'BiLSTMClassifier':
        model = BiLSTMClassifier(num_classes=27).to(device)
    else:
        raise ValueError(f"Unknown model type: {checkpoint['model_class']}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model successfully loaded from {path}")
    return model

def test_model(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # logits
            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # === Calculate Exact Match Accuracy ===
    exact_match = torch.all(all_preds == all_labels, dim=1).float()  # Check if the entire row matches perfectly
    exact_acc = exact_match.mean().item()

    print(f"\n✅ Exact Match Accuracy: {exact_acc:.4f}")

    correct_per_sample = (all_preds == all_labels).sum(dim=1)  # Number of correct predictions out of 27 labels per sample
    accuracy_per_sample = correct_per_sample / all_labels.size(1)  # Divide by 27
    mean_accuracy = accuracy_per_sample.mean().item()  # Average accuracy across all samples
    print(f"\nExample-based Multi-label Accuracy (mean sample-wise ACC): {mean_accuracy:.4f}")

    acc = 0 # (preds == labels).float().mean(dim=0).mean().item()
    metrix = evaluate_metrics(all_preds, all_labels)
    print(metrix)
    acc = metrix['accuracy']
    
    print("\nPrediction complete! Showing the first 5 prediction results:")
    for i in range(min(5, all_preds.size(0))):
        print(f"Predicted: {all_preds[i].tolist()}")
        print(f"Actual:    {all_labels[i].tolist()}")
        print("---")

    return all_preds, all_labels, exact_acc


if __name__ == '__main__':
    # === Calculate Accuracy ===
    
    path = '/root/autodl-tmp/MovieLabeling/datasets/test_data.csv'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize dataset and model
    dataset = RobertaTextDataset(csv_file=path, device=device)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = load_model('../weights/bilstm_model_6.pt', device=device)

    # Run evaluation
    all_preds, all_labels, acc = test_model(model, dataloader, device=device)
