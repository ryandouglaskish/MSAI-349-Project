import numpy as np
import pandas as pd

import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F

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
    

class RaceMultiOutputModel1(nn.Module):
    '''Basic model -- linear predictions
    '''
    def __init__(self):
        super(RaceMultiOutputModel1, self).__init__()
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

class RaceMultiOutputModel2(nn.Module):
    '''This model applies sigmoid
    '''
    def __init__(self):
        super(RaceMultiOutputModel2, self).__init__()
        self.fc1 = nn.Linear(240, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 24)  # Output layer: 24 driver positions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation, as this is a regression problem
        x = torch.sigmoid(x)
        x = x * 25
        return x
    



class RaceMultiOutputModel3(nn.Module):
    def __init__(self):
            super(RaceMultiOutputModel3, self).__init__()
            self.fc1 = nn.Linear(240, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 24)  # Output for 24 drivers

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Using raw scores for ranking
        return x


def spearman_rank_loss(predictions, targets):
    """
    predictions: Tensor of shape (batch_size, num_items) - model's output scores
    targets: Tensor of shape (batch_size, num_items) - true rankings
    """
    # Soft rank approximation using softmax
    soft_ranks_pred = torch.softmax(predictions, dim=1)
    soft_ranks_target = torch.softmax(targets, dim=1)

    # Calculate mean-subtracted versions
    pred_mean_sub = soft_ranks_pred - torch.mean(soft_ranks_pred, dim=1, keepdim=True)
    target_mean_sub = soft_ranks_target - torch.mean(soft_ranks_target, dim=1, keepdim=True)

    # Spearman rank correlation approximation
    covariance = torch.mean(pred_mean_sub * target_mean_sub, dim=1)
    pred_std = torch.sqrt(torch.mean(pred_mean_sub ** 2, dim=1))
    target_std = torch.sqrt(torch.mean(target_mean_sub ** 2, dim=1))

    corr_approx = covariance / (pred_std * target_std)
    loss = 1 - corr_approx  # Minimize loss by maximizing correlation
    return torch.mean(loss)
        
class RaceMultiOutputModel4(nn.Module):
    def __init__(self):
            super(RaceMultiOutputModel4, self).__init__()
            self.fc1 = nn.Linear(24, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 24)  # Output for 24 drivers

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Using raw scores for ranking
        return x
    
class RaceMultiOutputModel4Classification(nn.Module):
    def __init__(self):
            super(RaceMultiOutputModel4Classification, self).__init__()
            self.fc1 = nn.Linear(24, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 24)  # Output for 24 drivers

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Using raw scores for ranking
        #x = F.softmax(x, dim=1)
        return x

class RaceMultiOutputModel5(nn.Module):
    def __init__(self):
            super(RaceMultiOutputModel5, self).__init__()
            self.fc1 = nn.Linear(24, 24)
            self.fc2 = nn.Linear(24, 24)


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class RacePredictionModel6(nn.Module):
    def __init__(self, num_drivers, embedding_dim, hidden_dim):
        super(RacePredictionModel6, self).__init__()
        self.embedding = nn.Embedding(num_drivers, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 10, hidden_dim)  # Assuming 10 laps
        # ... other layers ...

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten the embeddings
        # ... rest of the forward pass ...

            
                
def weighted_ranking_loss(predictions, targets, winner_weight=5.0):
    """
    Custom ranking loss that prioritizes the first position and 
    ignores positions 10-24.

    predictions: The output from the model (batch_size x num_drivers).
    targets: The actual finishing positions (batch_size x num_drivers).
    winner_weight: The weight to apply for the first position.
    """
    loss = 0.0
    batch_size, num_drivers = predictions.shape
    for batch in range(batch_size):
        for i in range(num_drivers):
            for j in range(i + 1, num_drivers):
                # Access the position of drivers i and j in the current batch
                target_i = targets[batch, i].item()
                target_j = targets[batch, j].item()

                # Apply weights based on positions
                weight = 1.0
                if target_i == 1 or target_j == 1:
                    weight = winner_weight  # Increase weight for the first position
                elif target_i >= 10 and target_j >= 10:
                    weight = 0.0  # Ignore positions 10-24

                # Calculate pairwise loss
                predicted_diff = predictions[batch, i] - predictions[batch, j]
                true_diff = torch.tensor(target_i - target_j, dtype=predictions.dtype, device=predictions.device)  # Convert true_diff to a tensor
                loss += weight * F.relu(torch.sign(predicted_diff) * torch.sign(true_diff) - predicted_diff * true_diff)

    return loss / (batch_size * num_drivers * (num_drivers - 1) / 2)


# if __name__ == "__main__":
#     # Load data
#     X_train = pd.read_csv('Data/RaceMultiOutPutModel/X_train.csv')
#     y_train = pd.read_csv('Data/RaceMultiOutPutModel/Y_test.csv')
#     X_test = pd.read_csv('Data/RaceMultiOutPutModel/X_test.csv')
#     y_test = pd.read_csv('Data/RaceMultiOutPutModel/Y_test.csv')

#     X_train.drop(['raceId'],axis=1, inplace=True)
#     y_train.drop(['raceId'],axis=1, inplace=True)
#     X_test.drop(['raceId'],axis=1, inplace=True)
#     y_test.drop(['raceId'],axis=1, inplace=True)

#     X_train = X_train[0:4].copy()
#     y_train = y_train[0:4].copy()


    
#     # Create Dataset and DataLoader for train and test sets
#     train_dataset = RaceDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
#     test_dataset = RaceDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     N_SPLITS = 8

#     kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1)

#     for fold, (train_idx, test_idx) in enumerate(kfold.split(train_dataset)):
#         train_sampler = SubsetRandomSampler(train_idx)
#         test_sampler = SubsetRandomSampler(test_idx)

#         train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
#         test_loader = DataLoader(train_dataset, batch_size=64, sampler=test_sampler)

#         # Initialize the model for this fold
#         model = RaceMultiOutputModel()
#         criterion = nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#         # Train for each epoch
#         num_epochs = 10
#         for epoch in range(num_epochs):
#             # Training phase
#             model.train()
#             for inputs, targets in train_loader:
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()



#     # Instantiate the model, define loss function and optimizer
#     model = RaceMultiOutputModel()
#     criterion = nn.MSELoss()  # or a custom ranking loss
#     optimizer = optim.Adam(model.parameters(), lr=0.001)