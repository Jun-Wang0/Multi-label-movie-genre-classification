import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.BILSTMModel import BiLSTMClassifier
from TorchDataset.RobertaTextDataset import RobertaTextDataset

def collate_fn(batch):
    all_inputs = []
    all_labels = []

    for item in batch:
        # 拼接6个embedding向量 → 每个样本为 (7, 1024)
        emb_list = [
            item['title_embedding'],
            item['summaried_embedding'],
            item['synopsis_embedding'],
            item['title_tf'],
            item['summaried_tf'],
            item['synopsis_tf'],
            item['multi_embedding']
        ]
        emb_tensor = torch.stack(emb_list)
        all_inputs.append(emb_tensor)

        # 使用真实的 27 维标签向量
        all_labels.append(item['label'])

    inputs_tensor = torch.stack(all_inputs)  # (batch, 6, 1024)
    labels_tensor = torch.stack(all_labels)  # (batch, 27)

    return inputs_tensor, labels_tensor


def save_model(model, path='bilstm_model.pt'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__
    }, path)
    print(f"模型已保存到: {path}")


def train_model(dataset, device='cuda', num_epochs=5, batch_size=8, lr=1e-4, save_path='../weights/bilstm_model.pt'):
    model = BiLSTMClassifier(num_classes=27).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    best_loss = float('inf')

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0

        for batch_inputs, batch_labels in dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            outputs = model(batch_inputs)  # (batch, 27)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

        # 保存最优模型（按 loss）
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, save_path)

if __name__ == '__main__':
    path = '/root/autodl-tmp/MovieLabeling/datasets/train_data.csv'
    dataset = RobertaTextDataset(csv_file=path, device='cuda')
    train_model(dataset)
