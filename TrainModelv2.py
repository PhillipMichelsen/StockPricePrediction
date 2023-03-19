import os
import time
import warnings

import random
random.seed(1)

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchinfo import summary
from Classes import StockData, LSTM

import matplotlib.pyplot as plt

# Parameters
TRAIN_DATASETS_DIRECTORY = 'Data/Datasets/Stock_100'
#TEST_DATASET_DIRECTORY = 'Data/Datasets/Stock_100/Stock_100_5.11.pt'
DATASET_SPLITS = [10, 8] # Total, Train
NUM_EPOCHS = 50

MODEL_DIRECTORY = 'Models/I(90,22)_O(14,1)_128HS_5L_32B'
START_MODEL = 'BASE.pt'
START_MODEL_EPOCHS = 0

model_parameters = MODEL_DIRECTORY[7:].split('_')

INPUT_LENGTH = int(model_parameters[0][2:-1].split(',')[0])
INPUT_FEATURES = int(model_parameters[0][2:-1].split(',')[1])
OUTPUT_LENGTH = int(model_parameters[1][2:-1].split(',')[0])
OUTPUT_FEATURES = int(model_parameters[1][2:-1].split(',')[1])

MODEL_HIDDEN_SIZE = int(model_parameters[2][:-2])
MODEL_NUM_LAYERS = int(model_parameters[3][:-1])
BATCH_SIZE = int(model_parameters[4][:-1])

print('Model parameters as follows:')
print(f'Input shape - ({INPUT_LENGTH},{INPUT_FEATURES})')
print(f'Output shape - ({OUTPUT_LENGTH},{OUTPUT_FEATURES})')
print('----------------------------------')
print(f'Hidden size - {MODEL_HIDDEN_SIZE}')
print(f'Number of layers - {MODEL_NUM_LAYERS}')
print(f'Batch size - {BATCH_SIZE}')

input("VERIFY, press Enter...")

torch.cuda.is_available()
device = torch.device('cuda')

dataset_names = os.listdir(TRAIN_DATASETS_DIRECTORY)
dataset_names.sort(key=lambda x: float(x.split('_')[1][2:].rstrip('.pt')))
random.shuffle(dataset_names)

train_dataloaders = []
for dataset_name in dataset_names[0:DATASET_SPLITS[1]]:
    train_dataloaders.append(DataLoader(torch.load(f"{TRAIN_DATASETS_DIRECTORY}/{dataset_name}"), batch_size=32, shuffle=True, drop_last=True))

validation_dataloaders = []
for dataset_name in dataset_names[DATASET_SPLITS[1]:DATASET_SPLITS[0]]:
    validation_dataloaders.append(DataLoader(torch.load(f"{TRAIN_DATASETS_DIRECTORY}/{dataset_name}"), batch_size=32, shuffle=False, drop_last=True))

model = LSTM(INPUT_FEATURES,MODEL_HIDDEN_SIZE,MODEL_NUM_LAYERS,OUTPUT_LENGTH)
model.load_state_dict(torch.load(f'{MODEL_DIRECTORY}/BASE.pt'))
model.to(device)

train_criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

train_losses = []
validation_losses = []
epoch_times = []

# Train + Validate loop
for epoch in range(NUM_EPOCHS):
    start_time = time.time()

    # TRAIN
    model.train()
    train_loss = 0.0
    total_train_batches = 0

    for dataloader in train_dataloaders:
        for inputs, targets in dataloader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            optimizer.zero_grad()
            output = model(inputs)

            batch_train_loss = train_criterion(output, targets)

            batch_train_loss.backward()

            optimizer.step()

            train_loss += batch_train_loss.item()
            total_train_batches += 1

    train_loss /= total_train_batches
    train_losses.append(train_loss)

    # VALIDATE
    model.eval()

    with torch.no_grad():
        validation_loss = 0.0
        total_validation_tests = 0
        for dataloader in validation_dataloaders:
            for inputs, targets in dataloader:
                inputs, targets = inputs.float().to(device), targets.float().to(device)

                output = model(inputs)

                validation_loss += train_criterion(output, targets).item()
                total_validation_tests += 1

        validation_loss /= total_validation_tests
        validation_losses.append(validation_loss)

    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"{MODEL_DIRECTORY}/Model_State_Epoch_{epoch+1+START_MODEL_EPOCHS}.pt")
        print("STATE SAVED!")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.8f}, Validation Loss: {validation_loss:.8f} | Time taken: {epoch_time:.1f} seconds")
