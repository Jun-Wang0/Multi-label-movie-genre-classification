import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.RobertaEncoderOnEmbeddings import RobertaEncoderOnEmbeddings
from TorchDataset.RobertaTextDatasetAllInOne import RobertaTextDataset

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# =============== Eval ===============
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        inputs = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)

        logits = model(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # 方式二：逐标签准确率
    #acc = ((preds == labels).float().mean(dim=0)).mean().item()
    acc = (preds == labels).all(dim=1).float().mean().item()
    print(f"✅ Multi-label Accuracy: {acc:.4f}")
    return acc

# =============== Main ===============
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainpath = '/root/autodl-tmp/MovieLabeling/datasets/train_data.csv'
    traindataset = RobertaTextDataset(csv_file=trainpath, device='cuda')
    testpath = '/root/autodl-tmp/MovieLabeling/datasets/test_data.csv'
    testdataset = RobertaTextDataset(csv_file=testpath, device='cuda')
    
    train_loader = DataLoader(traindataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(testdataset, batch_size=32)

    model = RobertaEncoderOnEmbeddings(num_labels=27).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_acc = 0.0
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}")

        acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Val Macro F1: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "../weights/best_model.pt")
            print(f"✅ Saved new best model at epoch {epoch+1} (F1: {acc:.4f})")

if __name__ == "__main__":
    main()