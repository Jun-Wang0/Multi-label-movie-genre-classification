from transformers import RobertaModel, RobertaConfig
import torch.nn as nn
import torch

class RobertaEncoderOnEmbeddings(nn.Module):
    def __init__(self, num_labels=27, pretrained_model='/root/autodl-tmp/huggingface/roberta-large'):
        super().__init__()
        self.base_model = RobertaModel.from_pretrained(pretrained_model)
        self.encoder = self.base_model.encoder
        self.embeddings = self.base_model.embeddings
        self.config = self.base_model.config

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, x):  # x: (batch, 7, 1024)
        batch_size, seq_len, _ = x.size()
        device = x.device

        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        hidden_states = x + position_embeddings

        # ✅ 用 base_model 调用 get_extended_attention_mask
        extended_attention_mask = self.base_model.get_extended_attention_mask(
            attention_mask, input_shape=(batch_size, seq_len), device=device
        )

        encoder_outputs = self.encoder(hidden_states, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]  # shape: (batch, 7, 1024)

        pooled = sequence_output[:, 0, :]  # or mean pooling
        out = self.dropout(pooled)
        return self.classifier(out)
