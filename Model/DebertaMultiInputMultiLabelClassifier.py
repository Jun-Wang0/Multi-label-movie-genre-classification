import torch
import torch.nn as nn
from transformers import AutoModel  # 使用 AutoModel 兼容性更高
from transformers import T5EncoderModel

class DebertaMultiInputMultiLabelClassifier(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base', num_labels=27):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained('/root/autodl-tmp/huggingface/t5')
        self.hidden_size = self.encoder.config.d_model  # 注意不是 hidden_size，而是 d_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.hidden_size * 3, num_labels)
        self.output_dim = self.encoder.config.hidden_size
        
    def forward(self, input1, input2, input3):
        # T5 不存在 [CLS]，但第一个 token 可以作为全局向量
        cls1 = self.encoder(**input1).last_hidden_state[:, 0, :]  # (B, H)
        cls2 = self.encoder(**input2).last_hidden_state[:, 0, :]
        cls3 = self.encoder(**input3).last_hidden_state[:, 0, :]

        concat = torch.cat([cls1, cls2, cls3], dim=1)  # (B, H*3)
        out = self.dropout(concat)
        logits = self.classifier(out)  # (B, 27)
        return logits
