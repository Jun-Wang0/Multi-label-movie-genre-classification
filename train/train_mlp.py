import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import logging

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.BILSTMModel import BiLSTMClassifier
from Model.RobertaMultiInputMultiLabelClassifier import RobertaMultiInputMultiLabelClassifier
from Model.DebertaMultiInputMultiLabelClassifier import DebertaMultiInputMultiLabelClassifier

from TorchDataset.EnsembleDataset import EnsembleDataset  # 你需实现此类
from sklearn.metrics import classification_report

logging.set_verbosity_error()

def load_model(path='bilstm_model.pt', device='cpu'):
    checkpoint = torch.load(path, map_location=device)

    if checkpoint['model_class'] == 'BiLSTMClassifier':
        model = BiLSTMClassifier(num_classes=27).to(device)
    else:
        raise ValueError(f"未知模型类型: {checkpoint['model_class']}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型已从 {path} 加载")
    return model

# === 集成模型 === #
class EnsembleClassifier(nn.Module):
    def __init__(self, bilstm, roberta, deberta, hidden_dim=512, num_labels=27):
        super().__init__()
        self.bilstm = bilstm.eval()
        self.roberta = roberta.eval()
        self.deberta = deberta.eval()

        self.bilstm_output_dim = 27# bilstm.output_dim
        self.roberta_output_dim = 27# roberta.output_dim
        self.deberta_output_dim = 27# deberta.output_dim

        total_dim = self.bilstm_output_dim + self.roberta_output_dim + self.deberta_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels)
        )

        # 冻结三个模型
        for p in self.bilstm.parameters(): p.requires_grad = False
        for p in self.roberta.parameters(): p.requires_grad = False
        for p in self.deberta.parameters(): p.requires_grad = False

    def forward(self, bilstm_input, roberta_input, deberta_input):
        with torch.no_grad():
            v1 = self.bilstm(bilstm_input)  # (B, D1)
            v2 = self.roberta(roberta_input['input1'],roberta_input['input2'],roberta_input['input3'])  # (B, D2)
            v3 = self.deberta(deberta_input['input1'],deberta_input['input2'],deberta_input['input3'])  # (B, D3)
        print(f'v1')
        print(v1)
        print(f'v2')
        print(v2)
        print(f'v3')
        print(v3)
        fused = torch.cat([v1, v2, v3], dim=1)
        logits = self.classifier(fused)
        print('logits')
        print(logits)
        return logits
        

# === 训练函数 === #
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

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
        labels = batch['labels'].float().to(device)

        logits = model(bilstm_input, roberta_input, deberta_input)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# === 验证函数 === #
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

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
            labels = batch['labels'].float().to(device)

            logits = model(bilstm_input, roberta_input, deberta_input)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    report = classification_report(labels, preds, zero_division=0, output_dict=True)
    print(classification_report(labels, preds, zero_division=0))
    return report["macro avg"]["f1-score"]


# === 主函数 === #
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 加载子模型 === #
    bilstm = load_model('../weights/bilstm_model_0.9.pt')#BiLSTMClassifier(num_classes=27)
    #bilstm.load_state_dict(torch.load('../weights/bilstm_model_0.9.pt'))
    #bilstm.output_dim = 128  # 你必须设置此属性

    roberta = RobertaMultiInputMultiLabelClassifier()
    roberta.load_state_dict(torch.load('../weights/best_model_sr.pt'))
    #roberta.output_dim = 1024

    deberta = DebertaMultiInputMultiLabelClassifier()
    deberta.load_state_dict(torch.load('../weights/best_model_sd.pt'))
    #deberta.output_dim = 1024

    # === 构建集成模型 === #
    model = EnsembleClassifier(bilstm, roberta, deberta).to(device)

    # === 加载数据 === #
    train_dataset = EnsembleDataset(csv_file='/root/autodl-tmp/MovieLabeling/datasets/train_data.csv')
    val_dataset = EnsembleDataset(csv_file='/root/autodl-tmp/MovieLabeling/datasets/test_data.csv')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # === 优化器和损失 === #
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    # === 训练过程 === #
    best_f1 = 0
    for epoch in range(20):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_f1 = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), '../weights/best_ensemble.pt')
            print("✅ Saved new best model!")

if __name__ == '__main__':
    main()
