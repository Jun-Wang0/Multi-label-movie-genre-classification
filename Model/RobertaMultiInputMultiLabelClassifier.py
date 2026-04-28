import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaMultiInputMultiLabelClassifier(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=27):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('/root/autodl-tmp/huggingface/roberta-large')
        self.hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.hidden_size * 3, num_labels)
        self.output_dim = self.roberta.config.hidden_size
        
    def forward(self, input1, input2, input3):
        cls1 = self.roberta(**input1).last_hidden_state[:, 0, :]
        cls2 = self.roberta(**input2).last_hidden_state[:, 0, :]
        cls3 = self.roberta(**input3).last_hidden_state[:, 0, :]

        concat = torch.cat([cls1, cls2, cls3], dim=1)
        out = self.dropout(concat)
        return self.classifier(out)
