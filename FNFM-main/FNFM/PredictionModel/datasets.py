import copy
import random
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import *
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from PredictionModel.utils import *

def loadAM(addr, nodeindex):
    A = np.load(addr)
    nodeset = []
    for i in range(A.shape[0]):
        if A[i][nodeindex] != 0  or A[nodeindex][i] != 0:
            nodeset.append(i)
    n = len(nodeset)
    sm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            sm[i][j] = A[nodeset[i]][nodeset[j]]

    return sm, nodeset, A

def load_dynamics_data(file_path):
    """Load dynamics data in .npz format, taking only the first two channels
    Args:
        file_path: path to the .npz file
    Returns:
        Data array with shape (time_steps, nodes, features)
    """
    data = np.load(file_path)
    dynamics = data['X']
    # Ensure the data dimension order is correct: (time_steps, nodes, features)
    if dynamics.shape[-1] > 2:
        dynamics = dynamics[..., :2]
    return dynamics

class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) 
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo
class SingleDataset(Dataset):
    def __init__(self, data, A, mean, std, his_num, pred_num):
        super(SingleDataset, self).__init__()
        self.data=data
        self.A_wave=A
        self.mean = mean
        self.std = std
        self.his_num = his_num
        self.pred_num = pred_num
        
    def __getitem__(self, index):
        """Return the data point at the specified index"""
        x = torch.FloatTensor(self.data.x.float())
        y = torch.FloatTensor(self.data.y.float())
        data=Data(x=x,y=y,edge_index=self.data.edge_index,node_num=self.A_wave.shape[0])
        return data
    def __len__(self):
        """Return the length of the dataset"""
        return 1
    
    def get_normalize_params(self):
        """Return the normalization parameters of the data"""
        return self.mean, self.std

class DynamicsDataset(Dataset):
    def __init__(self, dataset,A,task_args,target_days=19):
        super(DynamicsDataset, self).__init__()
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.target_days = target_days
        self.A = A
        self.X=dataset

    def get_attr_func(self, matrix, edge_feature_matrix_path=None, node_feature_path=None):
        a, b = [], []
        edge_attr = []
        node_feature = None
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a,b]
        edge_index = torch.tensor(edge, dtype=torch.long)
        return edge_index, edge_attr, node_feature
    
    def get_dataset(self):
        """Return a data point"""
        # Ensure the dimensions of X are correct [time_steps, nodes, features]
        if len(self.X.shape) != 3:
            raise ValueError(f"Unexpected shape for dynamics data: {X.shape}")
        X = self.X.astype(np.float32)
        # Get edge information
        self.edge_index, self.edge_attr, _ = self.get_attr_func(self.A)
        # Ensure A is a NumPy array before passing to get_normalized_adj
        if isinstance(self.A, torch.Tensor):
            A_numpy = self.A.cpu().numpy()
        else:
            A_numpy = self.A
        self.A = torch.from_numpy(get_normalized_adj(A_numpy)).float()  # Ensure A is of float32 type

        # Data normalization
        self.means = np.mean(X, axis=(0, 2))
        X = X - self.means.reshape(1, -1, 1)
        self.stds = np.std(X, axis=(0, 2))
        num_zeros = np.sum(self.stds == 0)
        num_nans = np.sum(np.isnan(self.stds))
        if num_zeros!=0 or num_nans!=0:
            print(f"[DEBUG] Number of zeros in stds: {num_zeros}, Number of NaNs: {num_nans}")
            raise ZeroDivisionError()
        X = X / self.stds.reshape(1, -1, 1)
        if X.shape[2] == 1:
            # Method 1: Use NumPy's tile function
            X = np.tile(X, (1,1, 2))
        X = X.transpose((1, 2, 0))#(node_index,feature,time)
        # Select data based on the phase
        total_len=X.shape[2]
        train_ratio=0.8
        val_ratio=0.1
        train_end_idx = int(total_len * train_ratio)
        val_end_idx = int(total_len * (train_ratio + val_ratio))
        
        X_train = X[:, :, :train_end_idx]
        X_val = X[:, :, train_end_idx:val_end_idx]
        X_test = X[:, :, val_end_idx:]

        X_adjust = X[:, :, :self.target_days]
        x_data=[]
        y_data=[]

        # Generate input-output data pairs for training, validation, and test sets respectively
        common_args = {
            'num_timesteps_input': self.task_args['his_num'], 
            'num_timesteps_output': self.task_args['pred_num'],
            'means': self.means,
            'stds': self.stds
        }

        x_train, y_train = generate_dataset(X_train, **common_args)
        x_val, y_val = generate_dataset(X_val, **common_args)
        x_test, y_test = generate_dataset(X_test, **common_args)

        x_data.extend([x_train, x_val, x_test])
        y_data.extend([y_train, y_val, y_test])
        
        # Create a Data object and ensure it uses float32 type
        train_data = Data(
            x=x_data[0],
            y=y_data[0],
            edge_index=self.edge_index,
            node_num=self.A.shape[0]
        )
        trainloader=SingleDataset(train_data,self.A,self.means,self.stds,self.task_args['his_num'],self.task_args['pred_num'])
        val_data = Data(
            x=x_data[1],
            y=y_data[1],
            edge_index=self.edge_index,
            node_num=self.A.shape[0]
        )
        test_data = Data(
            x=x_data[2],
            y=y_data[2],
            edge_index=self.edge_index,
            node_num=self.A.shape[0]
        )
        testloader=SingleDataset(test_data,self.A,self.means,self.stds,self.task_args['his_num'],self.task_args['pred_num'])
        valloader=SingleDataset(val_data,self.A,self.means,self.stds,self.task_args['his_num'],self.task_args['pred_num'])
        return trainloader, testloader, valloader,self.A

    def __len__(self):
        return self.len

    def get_normalize_params(self):
        """Return the normalization parameters of the data"""
        return self.means, self.stds