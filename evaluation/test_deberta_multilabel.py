import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from torch.optim import AdamW

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.DebertaMultiInputMultiLabelClassifier import DebertaMultiInputMultiLabelClassifier
from TorchDataset.DebertaMultiInputDataset import DebertaMultiInputDataset
from metrixCaler import evaluate_metrics

def load_model(model_path, device='cuda'):
    # 初始化模型结构（必须与保存时一致）
    model = DebertaMultiInputMultiLabelClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ 模型已加载: {model_path}")
    return model

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
    acc = 0#(preds == labels).float().mean(dim=0).mean().item()
    metrix = evaluate_metrics(preds,labels)
    print(metrix)
    acc = metrix['accuracy']
    # print(f"✅ Val Accuracy: {acc:.4f}")
    return acc
    
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DebertaMultiInputDataset('/root/autodl-tmp/MovieLabeling/datasets/train_data.csv')
    val_dataset = DebertaMultiInputDataset('/root/autodl-tmp/MovieLabeling/datasets/test_data.csv')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = DebertaMultiInputMultiLabelClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    model = load_model("../weights/best_model_sd.pt", device)
    best_acc = 0.0
    val_acc = evaluate(model, val_loader, device)

if __name__ == "__main__":
    train()
    