import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import RobertaTokenizer

class RobertaMultiInputDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name='/root/autodl-tmp/huggingface/roberta-large', max_len=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        # 假设前3列是输入，后面是27个标签
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

