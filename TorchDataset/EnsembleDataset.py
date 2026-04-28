import torch
from torch.utils.data import Dataset
from TorchDataset.RobertaTextDataset import RobertaTextDataset
from TorchDataset.RobertaMultiInputDataset import RobertaMultiInputDataset
from TorchDataset.DebertaMultiInputDataset import DebertaMultiInputDataset


class EnsembleDataset(Dataset):
    def __init__(self, csv_file, device='cuda'):
        self.lstmdataset = RobertaTextDataset(csv_file=csv_file)
        self.robertadataset = RobertaMultiInputDataset(csv_file=csv_file)
        self.debertadataset = DebertaMultiInputDataset(csv_file=csv_file)

        # 统一长度
        assert len(self.lstmdataset) == len(self.robertadataset) == len(self.debertadataset), "Dataset size mismatch"

    def __len__(self):
        return len(self.lstmdataset)

    def __getitem__(self, idx):
        # 获取三路输入
        lstm_item = self.lstmdataset[idx]
        #lstm_input = lstm_item['embeddings']   # (6, 1024)
        roberta_item = self.robertadataset[idx]   # 包含: input_ids, attention_mask, labels
        deberta_item = self.debertadataset[idx]   # 同上

        # 假设三个 Dataset 返回的标签是一致的，只取一个
        return {
            'lstm_input': lstm_item['embeddings'],     # shape (7, 1024)
            'roberta_input': {
                'input1': roberta_item['input1'],
                'input2': roberta_item['input2'],
                'input3': roberta_item['input3'],
            },
            'deberta_input': {
                'input1': deberta_item['input1'],
                'input2': deberta_item['input2'],
                'input3': deberta_item['input3'],
            },
            'labels': lstm_item['labels']
        }
