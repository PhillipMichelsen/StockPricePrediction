import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class StockData(Dataset):
    def __init__(self, data, input_length=90, output_length=1):
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

            self.data.append((torch.FloatTensor(last_input_length_days), torch.FloatTensor(next_output_length_days)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ConvLSTM(nn.Module):
    def __init__(self, input_features,
                 conv_hidden_sizes, kernel_size,
                 lstm_hidden_sizes,
                 output_features, dropout=0.1
                 ):

        super(ConvLSTM, self).__init__()

        self.input_features = input_features

        self.conv_hidden_sizes = conv_hidden_sizes
        self.kernel_size = kernel_size
        self.lstm_hidden_sizes = lstm_hidden_sizes

        self.conv_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()

        self.conv1d = nn.Conv1d(15, 32, 3)

        for i, hidden_size in enumerate(self.conv_hidden_sizes):
            if i == 0:
                self.conv_layers.append(nn.Conv1d(
                    self.input_features, hidden_size, self.kernel_size)
                )
            else:
                self.conv_layers.append(nn.Conv1d(
                    self.conv_hidden_sizes[i-1], hidden_size, self.kernel_size)
                )

        for i, hidden_size in enumerate(self.lstm_hidden_sizes):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(
                    self.conv_hidden_sizes[-1], hidden_size, batch_first=True)
                )
            else:
                self.lstm_layers.append(nn.LSTM(
                    self.lstm_hidden_sizes[i-1], hidden_size, batch_first=True)
                )

        self.fc = nn.Linear(lstm_hidden_sizes[-1], output_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()

        print('Before transpotition: ', x.size())
        x = x.transpose(1, 2).float()
        print('After transpotition: ', x.size())
        print(x.size())
        print(x.type())
        x = self.conv1d(x)
        print('After conv1d: ', x.size())

        for conv_layer in self.conv_layers:
            print('Before conv layer: ', x.size())
            print('Conv layer: ', conv_layer)
            x = conv_layer(x)
            print('After conv layer: ', x.size())
            x = torch.relu(x)
            print('After relu: ', x.size())
            x = self.dropout(x)
            print('After dropout: ', x.size())

        for lstm_layer in self.lstm_layers:
            print('Before lstm layer: ', x.size())
            h0 = torch.zeros(1, batch_size, lstm_layer.hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, lstm_layer.hidden_size).to(x.device)
            print('h0: ', h0.size())
            print('c0: ', c0.size())
            x, _ = lstm_layer(x, (h0, c0))
            print('After lstm layer: ', x.size())
            x = self.dropout(x)
            print('After dropout: ', x.size())

        out = self.fc(x[:, -1, :])
        print('After fc: ', out.size())

        return out
