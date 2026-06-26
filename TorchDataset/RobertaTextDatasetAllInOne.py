import torch
from torch.utils.data import Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd

class RobertaTextDataset(Dataset):
    def __init__(self, csv_file, max_length=128, device='cpu'):
        self.device = device
        self.data = pd.read_csv(csv_file)
        self.tokenizer = RobertaTokenizer.from_pretrained('/root/autodl-tmp/huggingface/roberta-large')
        self.model = RobertaModel.from_pretrained('/root/autodl-tmp/huggingface/roberta-large').to(device)
        self.model.eval()
        
        self.max_length = max_length
        self.proj_concat = nn.Linear(3072, 1024).to(device)
        
        encoder_layer = TransformerEncoderLayer(d_model=1024, nhead=8).to(device)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1).to(device)
        self.norm_single = LayerNorm(1024).to(device)   # Used for each individual embedding (title, summary, synopsis)
        self.norm_concat = LayerNorm(3072).to(device)   # Used for the concatenated multi-modal vector

        self.titleembedding = []
        self.summariedembedding = []
        self.synopsisembedding = []
        self.titleembeddingaftertf = []
        self.summariedembeddingaftertf = []
        self.synopsisembeddingaftertf = []
        self.multiembedding = []

        # 27 genre label column names
        self.label_columns = [
            "Drama", "Thriller", "Comedy", "Action", "Adventure", "Crime", "Romance", "Mystery", "Sci-Fi", "Fantasy",
            "Horror", "Dark Comedy", "Family", "Period Drama", "Biography", "Animation", "Romantic Comedy", "Tragedy",
            "Psychological Thriller", "Psychological Drama", "Supernatural Horror", "Slapstick", "War", "History",
            "Coming-of-Age", "Superhero", "Docudrama"
        ]
        
        self._preprocess_all()

    def _get_embedding(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze(0)  # shape: (1024,)

    def _transform_and_norm(self, embedding_list):
        batch_tensor = torch.stack(embedding_list).unsqueeze(1)  # (N, 1, 1024)
        transformed = self.transformer(batch_tensor).squeeze(1)  # (N, 1024)
        return [self.norm_single(tensor) for tensor in transformed]

    def _preprocess_all(self):
        print("Starting text encoding...")
    
        for _, row in self.data.iterrows():
            self.titleembedding.append(self._get_embedding(str(row['title'])))
            self.summariedembedding.append(self._get_embedding(str(row['summaries'])))
            self.synopsisembedding.append(self._get_embedding(str(row['synopsis'])))
    
        print("Applying Transformer processing...")
        self.titleembeddingaftertf = self._transform_and_norm(self.titleembedding)
        self.summariedembeddingaftertf = self._transform_and_norm(self.summariedembedding)
        self.synopsisembeddingaftertf = self._transform_and_norm(self.synopsisembedding)
    
        print("Generating final multi-modal embedding...")
        self.multiembedding = []  # Initialize as an empty list
        for t, s, y in zip(self.titleembeddingaftertf, self.summariedembeddingaftertf, self.synopsisembeddingaftertf):
            concat = torch.cat([t, s, y], dim=0)         # shape: (3072,)
            reduced = self.proj_concat(concat)           # shape: (1024,)
            self.multiembedding.append(self.norm_single(reduced))  # shape: (1024,)
    
        # ✅ Convert all list[Tensor] -> Tensor
        self.titleembedding = torch.stack(self.titleembedding)                     # (N, 1024)
        self.summariedembedding = torch.stack(self.summariedembedding)
        self.synopsisembedding = torch.stack(self.synopsisembedding)
        self.titleembeddingaftertf = torch.stack(self.titleembeddingaftertf)
        self.summariedembeddingaftertf = torch.stack(self.summariedembeddingaftertf)
        self.synopsisembeddingaftertf = torch.stack(self.synopsisembeddingaftertf)
        self.multiembedding = torch.stack(self.multiembedding)
    
        # ✅ Stack multi-modal representations: (N, 7, 1024)
        self.embeddings = torch.stack([
            self.titleembedding,
            self.summariedembedding,
            self.synopsisembedding,
            self.titleembeddingaftertf,
            self.summariedembeddingaftertf,
            self.synopsisembeddingaftertf,
            self.multiembedding
        ], dim=1)  # ⬅ Use dim instead of axis
    
        # ✅ Label tensor (N, 27)
        self.labels = torch.tensor(self.data[self.label_columns].values, dtype=torch.float32).to(self.device)
    
        print("Preprocessing complete.")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx].detach().float(),
            'labels': self.labels[idx].clone().detach().float()
        }
