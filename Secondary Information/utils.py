# define the needed modules for the utils
import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import torch.nn.init as init
#%%
# define the class that will call the data set from an excel file, the last coloumn is the target. 
class MyDataset(Dataset): # the class takes the folder name as an input and create a x and y in tensor form for the pytorch. 
    def __init__(self, excel_file):
        self.data = pd.read_excel(excel_file)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x = torch.tensor(sample[:-1].values, dtype=torch.float32)
        y = torch.tensor(sample[-1], dtype=torch.long)
        return x, y

# this class generates data set with uniform distribution with the class to get more good results
class MyDataset1(Dataset):
    def __init__(self, excel_file):
        self.data = pd.read_excel(excel_file)
        
        # separate data into majority and minority classes
        majority_class = self.data[self.data['target'] == 0]
        minority_class_1 = self.data[self.data['target'] == 1]
        minority_class_2 = self.data[self.data['target'] == 2]
        # normalize features
        scaler = StandardScaler()
        majority_class.iloc[:, :-1] = scaler.fit_transform(majority_class.iloc[:, :-1])
        minority_class_1.iloc[:, :-1] = scaler.fit_transform(minority_class_1.iloc[:, :-1])
        minority_class_2.iloc[:, :-1] = scaler.fit_transform(minority_class_2.iloc[:, :-1])

        # resample minority classes to have same number of samples as majority class
        minority_class_1_resampled = resample(minority_class_1,
                                              replace=True,
                                              n_samples=len(majority_class),
                                              random_state=42)
        minority_class_2_resampled = resample(minority_class_2,
                                              replace=True,
                                              n_samples=len(majority_class),
                                              random_state=42)

        # combine minority classes with majority class
        self.data = pd.concat([majority_class, minority_class_1_resampled, minority_class_2_resampled])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x = torch.tensor(sample[:-1].values, dtype=torch.float32)
        y = torch.tensor(sample[-1], dtype=torch.long)
        return x, y



# now defion ethe neural network for tranning 
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 100),
            nn.BatchNorm1d(100),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(100, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x) # define the forward path
    
    
# define a resuidal neural network aslo to compare:
class ResidualNet(nn.Module):
    def __init__(self):
        super(ResidualNet, self).__init__()
        self.layer1 = nn.Linear(5, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.layer2 = nn.Linear(10, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.layer3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)
        self.layer4 = nn.Linear(10, 3)
        self.fc = nn.Linear(5, 3)  # new fully connected layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.layer1(x)))
        x2 = self.relu(self.bn2(self.layer2(x1)))
        x3 = self.relu(self.bn3(self.layer3(x2) + x1))
        x4 = self.softmax(self.layer4(x3) + self.fc(x))  # use fc layer to match sizes
        return x4
#%% get the accurcy 
def compute_accuracy(prediction,gt_logits):
    pred_idx = np.argmax(prediction,1,keepdims=True)
    matches = pred_idx == gt_logits[:,None]
    acc = matches.mean()
    return acc
##################################################
# this is regression part:
class RegressionNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = torch.nn.Sequential(
      torch.nn.Linear(6, 100),
      torch.nn.BatchNorm1d(100),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(100, 100),
      torch.nn.BatchNorm1d(100),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(100, 100),
      torch.nn.BatchNorm1d(100),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(100, 100),
      torch.nn.BatchNorm1d(100),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(100, 1)
    )
    #Apply Glorot weight initialization and zero bias initialization to linear layers
    for layer in self.layers:
      if isinstance(layer, torch.nn.Linear):
        init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0)

  def forward(self, x):
    # Clamp output between 0 and 30
    return torch.clamp(self.layers(x), 0, 30)
    #return self.layers(x)

## dont add unifrom dataset:
class MyDataset3(Dataset):
    def __init__(self, excel_file):
        self.data = pd.read_excel(excel_file)
        self.x_mean = self.data.iloc[:, :-1].mean()
        self.x_std = self.data.iloc[:, :-1].std()
             
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x = torch.tensor((sample[:-1].values - self.x_mean) / self.x_std, dtype=torch.float32)
        #x = torch.tensor(sample[:-1].values, dtype=torch.float32)
        y = torch.tensor(sample[-1], dtype=torch.float32)
        return x, y


##
class MyDataset2(Dataset):
    def __init__(self, excel_file):
        self.data = pd.read_excel(excel_file)
        
        # separate data into majority and minority classes
        majority_class = self.data[self.data['target'] < 0.5]
        minority_class_1 = self.data[self.data['target'] >29.5]
        minority_class_2 = self.data[(self.data['target'] <=29.5)&(self.data['target'] >=0.5)]
        #Nomlize the input data as well:
        # normalize features
        scaler = StandardScaler()
        majority_class.iloc[:, :-1] = scaler.fit_transform(majority_class.iloc[:, :-1])
        minority_class_1.iloc[:, :-1] = scaler.fit_transform(minority_class_1.iloc[:, :-1])
        minority_class_2.iloc[:, :-1] = scaler.fit_transform(minority_class_2.iloc[:, :-1])


        # resample minority classes to have same number of samples as majority class
        minority_class_1_resampled = resample(minority_class_1,
                                              replace=True,
                                              n_samples=len(majority_class),
                                              random_state=42)
        minority_class_2_resampled = resample(minority_class_2,
                                              replace=True,
                                              n_samples=len(majority_class),
                                              random_state=42)
        # add small noise to resampled minority class samples
        noise_scale = np.random.uniform(0.99, 1.01, size=(len(majority_class), minority_class_1_resampled.shape[1]-1))
        minority_class_1_resampled.iloc[:, :-1] *= noise_scale
        noise_scale = np.random.uniform(0.99, 1.01, size=(len(majority_class), minority_class_2_resampled.shape[1]-1))
        minority_class_2_resampled.iloc[:, :-1] *= noise_scale

        # combine minority classes with majority class
        self.data = pd.concat([majority_class, minority_class_1_resampled, minority_class_2_resampled])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x = torch.tensor(sample[:-1].values, dtype=torch.float32)
        y = torch.tensor(sample[-1], dtype=torch.float32)
        return x, y