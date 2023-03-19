import os
import time
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchinfo import summary
from Classes import StockData, LSTM

import matplotlib.pyplot as plt

# Parameters
TRAIN_DATASETS_DIRECTORY = 'Data/Datasets/Stock_100'
# TEST_DATASET_FILE = 'Data/Datasets/'
DATASET_SPLIT = [0.8, 0.2]
NUM_EPOCHS = 50

MODEL_DIRECTORY = 'Models/I(90,22)_O(14,1)_256HS_5L_64B'
MODEL_BASE_NAME = 'BASE'

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

torch.cuda.is_available()
device = torch.device('cuda')

dataset_names = os.listdir(TRAIN_DATASETS_DIRECTORY)
dataset_names.sort(key=lambda x: float(x.split('_')[1][2:].rstrip('.pt')))

for i, dataset in enumerate(dataset_names):

    model = LSTM(INPUT_FEATURES, MODEL_HIDDEN_SIZE, MODEL_NUM_LAYERS, OUTPUT_LENGTH)

    print(f"\n----- STARTING RUN {i} -----")

    if os.path.exists(f'{MODEL_DIRECTORY}/Pass{i}.pt'):
        print(f"----- RUN ALREADY COMPLETED PREVIOUSLY, SKIPPING {dataset} | Run {i} -----")
        continue
    elif i == 0:
        model.load_state_dict(torch.load(f'{MODEL_DIRECTORY}/BASE.pt'))
    else:
        model.load_state_dict(torch.load(f'{MODEL_DIRECTORY}/Pass{i - 1}.pt'))

    model.to(device)

    main_dataset = torch.load(f'{TRAIN_DATASETS_DIRECTORY}/{dataset}')
    # test_dataset = torch.load(TEST_DATASET_FILE)

    train_dataset, validation_dataset = torch.utils.data.random_split(main_dataset, DATASET_SPLIT)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)

    train_losses = []
    test_losses = []
    epoch_times = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss = 0

        # Train for epoch
        model.train()
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs).float()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Evaluate for epoch
        with torch.no_grad():
            validation_loss = 0
            model.eval()

            for inputs, targets in validation_dataloader:
                inputs, targets = inputs.float().to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                validation_loss += loss.item()

            validation_loss /= len(validation_dataloader)
            test_losses.append(validation_loss)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(
            f"RUN {i}, {dataset}| Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.7f}, Validation Loss: {validation_loss:.7f} | Time taken: {epoch_time:.1f} seconds")

    print(f"\n----- RUN COMPLETE -----")
    torch.save(model.state_dict(), f'{MODEL_DIRECTORY}/Pass{i}.pt')
