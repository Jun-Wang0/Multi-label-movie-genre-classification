import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=1, num_classes=27, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # dropout 仅当 num_layers > 1 时有效
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)  # 输出 27 维（多标签 logits）
        )

    def forward(self, x):
        # x: (batch, 7, 1024)
        lstm_out, _ = self.lstm(x)  # -> (batch, 7, hidden_dim * 2)
        pooled = torch.mean(lstm_out, dim=1)  # -> (batch, hidden_dim * 2)
        logits = self.classifier(pooled)      # -> (batch, 27)
        return logits  # 不加激活函数，留给 BCEWithLogitsLoss
