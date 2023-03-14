import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


class StockData(Dataset):
    def __init__(self, data, input_length=90, output_length=14):
        current_data = pd.DataFrame(data)
        self.data = []
        self.inputScaler = MinMaxScaler()
        self.outputScaler = MinMaxScaler()

        self.inputScaler.fit(current_data)
        self.outputScaler.fit(current_data['Close'].values.reshape(-1, 1))

        for i in range(input_length, len(current_data) - output_length):
            last_input_length_days = current_data.iloc[i - input_length: i].values
            next_output_length_days = current_data['Close'].iloc[i: i + output_length].values.reshape(-1, 1)

            last_input_length_days = self.inputScaler.transform(last_input_length_days)
            next_output_length_days = self.outputScaler.transform(next_output_length_days)

            self.data.append((last_input_length_days, next_output_length_days))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)

        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(torch.device('cuda'))
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(torch.device('cuda'))

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # dropout and fully connected layer
        out = self.dropout(out[-1])
        out = self.fc(out)
        out = out.unsqueeze(-1)

        return out
