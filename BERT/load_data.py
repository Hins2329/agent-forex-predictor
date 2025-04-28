# bert/load_data.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer


class FinancialPhraseDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="distilbert-base-uncased", max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["sentence"]
        label = self.data.iloc[idx]["label"]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # output of encoding is (batch_size, seq_len), squeeze(0) to get rid of batch
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)

        return item


def create_dataloader(csv_path, batch_size=32, shuffle=True):
    dataset = FinancialPhraseDataset(csv_file=csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
