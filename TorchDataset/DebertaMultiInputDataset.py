import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import T5Tokenizer
from transformers import DebertaV2Tokenizer

class DebertaMultiInputDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='microsoft/deberta-v3-base', max_len=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = T5Tokenizer.from_pretrained('/root/autodl-tmp/huggingface/t5')
        self.max_len = max_len

        self.input_texts = self.data.iloc[:, 0:3].values.tolist()
        self.labels = self.data.iloc[:, 3:].values.astype(float)

    def __len__(self):
        return len(self.data)

    def tokenize_text(self, text):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

    def __getitem__(self, idx):
        t1, t2, t3 = self.input_texts[idx]
        input1 = self.tokenize_text(str(t1))
        input2 = self.tokenize_text(str(t2))
        input3 = self.tokenize_text(str(t3))

        item = {
            'input1': {k: v.squeeze(0) for k, v in input1.items()},
            'input2': {k: v.squeeze(0) for k, v in input2.items()},
            'input3': {k: v.squeeze(0) for k, v in input3.items()},
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item
