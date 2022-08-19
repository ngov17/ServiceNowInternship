import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from dataset_utils import split_data, datasets

class BertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.inputs = data['text'].apply(lambda x: tokenizer(x, padding="max_length", max_length=max_length, return_tensors="pt", truncation=True))
        self.labels = data['label']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], torch.tensor(self.labels[idx])

"""
:param dataset - the type of dataset: news, spam, etc
"""
def load_dataset(dataset, tokenizer, batch_size=4, train_test_split=0.1):
    dataset = datasets[dataset]()
    
    train_data, test_data = split_data(dataset, train_test_split)
    print(len(train_data), len(test_data))
    d_set_train = BertDataset(train_data, tokenizer)
    d_set_test = BertDataset(test_data, tokenizer)
    
    # TODO: implement batching to optimize training
    # d_loader_train = DataLoader(dataset=d_set_train, batch_size=batch_size, shuffle=True)
    # d_loader_test = DataLoader(dataset=d_set_test, batch_size=batch_size, shuffle=True)

    return d_set_train, d_set_test


        

