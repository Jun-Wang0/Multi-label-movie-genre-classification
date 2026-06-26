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
            dropout=dropout if num_layers > 1 else 0  # dropout is only effective when num_layers > 1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)  # Outputs 27 dimensions (multi-label logits)
        )

    def forward(self, x):
        # x: (batch, 7, 1024)
        lstm_out, _ = self.lstm(x)  # -> (batch, 7, hidden_dim * 2)
        pooled = torch.mean(lstm_out, dim=1)  # -> (batch, hidden_dim * 2)
        logits = self.classifier(pooled)      # -> (batch, 27)
        return logits  # No activation function applied here; deferred to BCEWithLogitsLoss
