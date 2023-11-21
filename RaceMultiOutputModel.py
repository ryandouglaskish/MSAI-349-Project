import numpy as np
import pandas as pd

import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler



from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns

import time
from tqdm import tqdm



# Define a custom Dataset
class RaceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

class RaceMultiOutputModel(nn.Module):
    def __init__(self):
        super(RaceMultiOutputModel, self).__init__()
        self.fc1 = nn.Linear(240, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 24)  # Output layer: 24 driver positions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation, as this is a regression problem
        return x



#def custom_performance_metrics(true_pos, pred_pos):

    #pred_pos = pred_pos.detach().numpy()
    #pred_pos



if __name__ == "__main__":
    # Load data
    X_train = pd.read_csv('Data/RaceMultiOutPutModel/X_train.csv')
    y_train = pd.read_csv('Data/RaceMultiOutPutModel/Y_test.csv')
    X_test = pd.read_csv('Data/RaceMultiOutPutModel/X_test.csv')
    y_test = pd.read_csv('Data/RaceMultiOutPutModel/Y_test.csv')



    X_train = X_train[0:3].copy()
    y_train = y_train[0:3].copy()


    
    # Create Dataset and DataLoader for train and test sets
    train_dataset = RaceDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = RaceDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    N_SPLITS = 8

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(train_dataset)):
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
        test_loader = DataLoader(train_dataset, batch_size=64, sampler=test_sampler)

        # Initialize the model for this fold
        model = RaceMultiOutputModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train for each epoch
        num_epochs = 10
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()



    # Instantiate the model, define loss function and optimizer
    model = RaceMultiOutputModel()
    criterion = nn.MSELoss()  # or a custom ranking loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)