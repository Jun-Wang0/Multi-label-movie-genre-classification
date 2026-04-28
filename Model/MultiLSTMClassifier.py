import torch
import torch.nn as nn

class MultiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, num_labels=27):
        super().__init__()
        self.num_labels = num_labels
        self.lstm_list = nn.ModuleList([
            nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            for _ in range(num_labels)
        ])
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_labels)
        ])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):  # x shape: (batch_size, seq_len, input_dim)
        outputs = []

        for i in range(self.num_labels):
            lstm_out, _ = self.lstm_list[i](x)         # (batch_size, seq_len, hidden_dim)
            final_hidden = lstm_out[:, -1, :]           # 取最后一个时刻的 hidden
            final_hidden = self.dropout(final_hidden)
            out = self.output_layers[i](final_hidden)   # (batch_size, 1)
            outputs.append(out)

        # 拼接为 (batch_size, 27)
        logits = torch.cat(outputs, dim=1)
        return logits  # 不加 sigmoid（交给 BCEWithLogitsLoss）
