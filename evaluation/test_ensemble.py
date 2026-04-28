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


# ==== 屏蔽transformers的warning ====
logging.set_verbosity_error()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_multiclass_roc(y_true, y_score, num_classes, save_path=None):
    """
    绘制多分类 ROC 曲线，并可保存图片。
    
    参数:
    y_true: 长度为N的真实标签（整数类别）
    y_score: N×C 的概率数组，C为类别数
    num_classes: 类别数量
    save_path: 如果不为 None，则保存图像到该路径（如 'roc_curve.png'）
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

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC 图已保存至: {save_path}")

    #plt.show()

# ==== 加载子模型 ====
def load_model(path='bilstm_model.pt', device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model = BiLSTMClassifier(num_classes=27).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def strict_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算严格准确率：所有标签都完全匹配才算正确。
    
    参数:
        preds (torch.Tensor): 预测结果，形状为 (batch_size, num_labels)，值为 0 或 1。
        labels (torch.Tensor): 真实标签，形状相同。
        
    返回:
        float: 严格准确率。
    """
    correct = (preds == labels).all(dim=1).float()  # 每一行是否全部匹配
    acc = correct.mean().item()
    return acc

# ==== 集成模型结构 ====
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
        
        # print(f'v1')
        # print(v1)
        # print(f'v2')
        # print(v2)
        # print(f'v3')
        # print(v3)
        fused = torch.cat([v1, v2, v3], dim=1)
        logits = self.classifier(fused)
        # print('logits')
        # print(logits)
        return logits


# ==== 测试函数 ====
def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

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

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrix = evaluate_metrics(preds,labels)
    print(metrix)
    
    acc = strict_accuracy(preds,labels)
    print(f'acc = {acc}')
    print("\n📊 Classification Report:")
    print(classification_report(labels, preds, zero_division=0))
    # 概率分布
    probs = torch.softmax(preds.float(), dim=1)

    # 🎯 绘制 ROC 曲线
    plot_multiclass_roc(
        y_true=labels.cpu().numpy(),
        y_score=probs.cpu().numpy(),
        num_classes=probs.size(1),
        save_path='roc_curve.png'
    )
    
    return preds


# ==== 主入口 ====
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载子模型
    bilstm = load_model('../weights/bilstm_model_0.9.pt', device=device)
    roberta = RobertaMultiInputMultiLabelClassifier()
    roberta.load_state_dict(torch.load('../weights/best_model_sr.pt'))
    deberta = DebertaMultiInputMultiLabelClassifier()
    deberta.load_state_dict(torch.load('../weights/best_model_sd.pt'))

    # 构建集成模型并加载权重
    model = EnsembleClassifier(bilstm, roberta, deberta).to(device)
    model.load_state_dict(torch.load('../weights/best_ensemble.pt', map_location=device))
    model.eval()

    # 加载测试数据
    test_dataset = EnsembleDataset(csv_file='/root/autodl-tmp/MovieLabeling/datasets/test_data.csv')
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 开始测试
    test(model, test_loader, device)


if __name__ == '__main__':
    main()
