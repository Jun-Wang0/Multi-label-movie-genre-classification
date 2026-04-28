import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from BILSTMModel import BiLSTMClassifier
from RobertaMultiInputMultiLabelClassifier import RobertaMultiInputMultiLabelClassifier
from DebertaMultiInputMultiLabelClassifier import DebertaMultiInputMultiLabelClassifier

class EnsembleInferenceModel(nn.Module):
    def __init__(self, bilstm_path, roberta_path, deberta_path, hidden_size=512, num_labels=27, device='cuda'):
        super().__init__()
        self.device = device

        # 加载三个子模型
        self.bilstm = BiLSTMClassifier  # 确保结构一致
        self.bilstm.load_state_dict(torch.load(bilstm_path, map_location=device))
        self.bilstm.to(device)
        self.bilstm.eval()

        self.roberta = RobertaMultiInputMultiLabelClassifier
        self.roberta.load_state_dict(torch.load(roberta_path, map_location=device))
        self.roberta.to(device)
        self.roberta.eval()

        self.deberta = DebertaMultiInputMultiLabelClassifier
        self.deberta.load_state_dict(torch.load(deberta_path, map_location=device))
        self.deberta.to(device)
        self.deberta.eval()

        # 获取子模型输出维度
        total_hidden = self.bilstm.output_dim + self.roberta.output_dim + self.deberta.output_dim

        # MLP 分类器
        self.classifier = nn.Sequential(
            nn.Linear(total_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, inputs_bilstm, inputs_roberta, inputs_deberta):
        with torch.no_grad():
            vec1 = self.bilstm(inputs_bilstm)
            vec2 = self.roberta(inputs_roberta)
            vec3 = self.deberta(inputs_deberta)
        
        fused = torch.cat([vec1, vec2, vec3], dim=1)
        logits = self.classifier(fused)
        print('logits')
        print(logits)
        print(ssss)
        return logits
