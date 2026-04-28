import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from torch.optim import AdamW

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.RobertaMultiInputMultiLabelClassifier import RobertaMultiInputMultiLabelClassifier
from TorchDataset.RobertaMultiInputDataset import RobertaMultiInputDataset


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input1 = {k: v.to(device) for k, v in batch['input1'].items()}
            input2 = {k: v.to(device) for k, v in batch['input2'].items()}
            input3 = {k: v.to(device) for k, v in batch['input3'].items()}
            labels = batch['labels'].to(device)

            logits = model(input1, input2, input3)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    report = classification_report(labels, preds, zero_division=0, output_dict=True)
    acc = (preds == labels).float().mean(dim=0).mean().item()

    print(f"✅ Val Accuracy: {acc:.4f}")
    return acc

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RobertaMultiInputDataset('/root/autodl-tmp/MovieLabeling/datasets/train_data.csv')
    val_dataset = RobertaMultiInputDataset('/root/autodl-tmp/MovieLabeling/datasets/test_data.csv')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = RobertaMultiInputMultiLabelClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    os.makedirs("../weights", exist_ok=True)

    for epoch in range(1, 200+1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input1 = {k: v.to(device) for k, v in batch['input1'].items()}
            input2 = {k: v.to(device) for k, v in batch['input2'].items()}
            input3 = {k: v.to(device) for k, v in batch['input3'].items()}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input1, input2, input3)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"../weights/best_model_sr.pt")
            print(f"✅ Saved best model (epoch {epoch}, acc {val_acc:.4f})")

if __name__ == "__main__":
    train()
