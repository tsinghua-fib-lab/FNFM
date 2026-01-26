import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import pandas as pd
from datasets import DynamicsDataset
from Models import *
from NetSet import *
from torch.utils.data.sampler import *
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import *
import random
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import importlib
##########
use_pretrain=True
model_name='v_GWN'
###########
WANDB_ENABLE = True  # Must enable wandb upload
wandb_runs = {}  # One run per env
wandb_steps = {}  # One step count per env



def print_memory_usage(message=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    print(f"[Memory] {message}: Allocated={allocated:.2f}MB, Cached={cached:.2f}MB")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_wandb_run(dataname, env_number):
    """Lazy init wandb run; raise error if not installed."""
    global wandb_runs, wandb_steps
    if not WANDB_ENABLE:
        raise RuntimeError("WANDB_ENABLE is forced to True, but detected disabled")
    if env_number in wandb_runs:
        return wandb_runs[env_number]
    spec = importlib.util.find_spec("wandb")
    if spec is None:
        raise ImportError("wandb is required: please pip install wandb first")
    wandb = importlib.import_module("wandb")
    project = os.getenv("WANDB_PROJECT", "crosscity-gwn-disturb")
    entity = os.getenv("WANDB_ENTITY", None)
    wandb_run = wandb.init(
        project=project,
        entity=entity,
        name=f"{dataname}_env{env_number}",
        config={"dataname": dataname, "model": model_name, "use_pretrain": use_pretrain},
    )
    # Set unified step metric, avoiding regression across stages
    wandb_run.define_metric("global_step")
    wandb_run.define_metric("*", step_metric="global_step")
    wandb_runs[env_number] = wandb_run
    wandb_steps[env_number] = 0
    if hasattr(wandb_run, "url"):
        print(f"[wandb] run url: {wandb_run.url}")
    return wandb_run
        
def train_epoch(model, A_wave,train_dataloader, optimizer, loss_criterion, device):
    """Train for one epoch"""
    model.train()
    train_losses = []
    
    # Add data check
    print(f"Number of training data batches: {len(train_dataloader)}")
    for batch_idx,data in enumerate(train_dataloader):
        # Print information of the first batch
        if batch_idx == 0:
            print(f"Data shape: {data.x.shape}")
            print(f"Adjacency matrix shape: {A_wave.shape}")
            print(f"Label shape: {data.y.shape}")
            print(f"Data type: {data.x.dtype}")
            print(f"Data range: [{data.x.min():.4f}, {data.x.max():.4f}]")
            
        optimizer.zero_grad()
        A_wave = A_wave.to(device=device)
        A_wave = A_wave.float()
        data = data.to(device=device)
        
        out, meta_graph = model(data, A_wave)
        loss_predict = loss_criterion(out, data.y)
        loss_reconstruct = loss_criterion(meta_graph, A_wave)
        loss = loss_predict + loss_reconstruct
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.detach().cpu().numpy())
        
    return sum(train_losses)/len(train_losses)
@torch.no_grad()
def get_params(model):
    allparams = list(model.named_parameters())
    iffirst = True
    shapelist  = []
    lengthlist = []
    startlist  = [0]
    for singleparams in allparams:
        astensor = singleparams[1].clone().detach() 
        shapelist.append(astensor.shape)
        tensor1D = astensor.flatten()
        lengthlist.append(tensor1D.shape[0])
        tensor1D = tensor1D.unsqueeze(0)
        if iffirst == True:
            finaltensor = tensor1D
            iffirst = False
        else:
            finaltensor = torch.cat((finaltensor,tensor1D), dim = 1)
    return finaltensor
@torch.no_grad()
def calculate_std(raw_params):
    std_params = []
    if model_name == 'v_GWN':
        list_index = [0,16384,32768,40960,49152,49408,49664,90624,90688,91712,91904]
    else:
        list_index = [0, 256,1024,4096, 16384, 16960]
    for i in range(len(list_index)-1):
        std_params.append(raw_params[:,list_index[i]:list_index[i+1]].std().item())
    return std_params

@torch.no_grad()
def load_params(model,param):
    if model_name == 'v_STGCN5':
        indexstart = [0, 256, 512, 768, 1024, 2048, 3072, 4096, 8192, 12288, 16384, 16960]
        shapes = [(32, 8), (32, 2, 1, 4), (32, 2, 1, 4), (32, 2, 1, 4), (32, 8, 1, 4), 
                (32, 8, 1, 4), (32, 8, 1, 4), (32, 32, 1, 4), (32, 32, 1, 4), 
                (32, 32, 1, 4), (6, 96)]
    elif model_name == 'v_GWN':
        indexstart = [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 
                          18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 
                          33792, 34816, 35840, 36864, 37888, 38912, 39936, 40960, 
                          41984, 43008, 44032, 45056, 46080, 47104, 48128, 49152, 
                          49184, 49216, 49248, 49280, 49312, 49344, 49376, 49408, 
                          49440, 49472, 49504, 49536, 49568, 49600, 49632, 49664, 
                          54784, 59904, 65024, 70144, 75264, 80384, 85504, 90624, 
                          90688, 91712, 91904]
        shapes = [(32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), 
                    (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), 
                    (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), 
                    (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), 
                    (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), 
                    (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), 
                    (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), 
                    (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), 
                    (32,), (32,), (32,), (32,), (32,), (32,), (32,), (32,), 
                    (32,), (32,), (32,), (32,), (32,), (32,), (32,), (32,), 
                    (32, 160, 1, 1), (32, 160, 1, 1), (32, 160, 1, 1), (32, 160, 1, 1), 
                    (32, 160, 1, 1), (32, 160, 1, 1), (32, 160, 1, 1), (32, 160, 1, 1), 
                    (32, 2, 1, 1), (32, 32, 1, 1), (6, 32, 1, 1)]
    param_prepare=param.clone().detach().reshape(-1)
    index = 0
    for key in model.state_dict().keys():
        if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key):
            pa = param_prepare[indexstart[index]:indexstart[index+1]]
            pa = torch.reshape(pa, shapes[index]) 
            model.state_dict()[key].copy_(pa)               
            index = index+1
    return model
@torch.no_grad()
def disturbe_params(model,std_params,disturbe_rate=0.001,disturbe_number=0):
    
    torch.manual_seed(disturbe_number)
    std_index=[]
    params=get_params(model)
    if model_name == 'v_STGCN5':
        for i in range(5):
            std_index.append(0)
        std_index[disturbe_number%5]=std_params[disturbe_number%5]
        list_index=[0, 256,1024,4096, 16384, 16960]
    elif model_name == 'v_GWN':
        list_index = [0,16384,32768,40960,49152,49408,49664,90624,90688,91712,91904]
        for i in range(len(list_index)-1):
            std_index.append(0)
        std_index[disturbe_number%len(list_index)-1]=std_params[disturbe_number%len(list_index)-1]
    for i in range(len(std_params)):
        params[:,list_index[i]:list_index[i+1]]=params[:,list_index[i]:list_index[i+1]]+torch.randn(params.shape[0],list_index[i+1]-list_index[i]).to(params.device)*std_params[i]*disturbe_rate
    load_params(model,params)
    return model


def reconstruct_series(windows: np.ndarray, horizon: int, stride: int = 1) -> np.ndarray:
    """
    Reconstruct single time series from sliding window predictions (overlap averaging).
    windows: shape [num_windows, horizon]
    """
    if windows.ndim != 2 or windows.shape[1] != horizon:
        raise ValueError(f"windows shape {windows.shape} incompatible with horizon {horizon}")
    total_steps = (windows.shape[0] - 1) * stride + horizon
    seq = np.zeros(total_steps, dtype=np.float32)
    counts = np.zeros(total_steps, dtype=np.float32)
    for idx, window in enumerate(windows):
        start = idx * stride
        seq[start:start + horizon] += window
        counts[start:start + horizon] += 1
    counts[counts == 0] = 1
    return seq / counts

def test_epoch(model, test_dataloader, device):
    """Test the model"""
    model.eval()
    with torch.no_grad():
        for step, (data, A_wave) in enumerate(test_dataloader):
            A_wave = A_wave.to(device=device)
            data = data.to(device=device)
            if model_name == 'v_GWN':
                adj_mx = [A_wave.float(), (A_wave.float()).t()]
                out = model(data, adj_mx)
            else:
                out, _ = model(data, A_wave)
            
            if step == 0:
                outputs = out
                y_label = data.y
            else:
                outputs = torch.cat((outputs, out))
                y_label = torch.cat((y_label, data.y))
                
        outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
        y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
    return outputs, y_label

def parse_arguments():
    parser = argparse.ArgumentParser(description='MAML-based')
    parser.add_argument('--dataname', default='collab', type=str, help='Name of the dataset')
    parser.add_argument('--config_filename', default='config.yaml', type=str,
                            help='Configuration filename for restoring the model.')
    parser.add_argument('--test_dataset', default='metr-la', type=str)
    parser.add_argument('--model', default='v_STGCN5', type=str)
    parser.add_argument('--epochs', default=200, type=int)   
    parser.add_argument('--taskmode', default='task4', type = str)
    parser.add_argument("--ifchosenode", action="store_true")
    parser.add_argument('--logindex', default='0', type = str)
    parser.add_argument('--aftername',default='', type = str) 
    parser.add_argument('--data_args',default="Dynamics", type = str)  
    parser.add_argument('--graph_range', type=str, help='Specify which graphs to process: "first500", "last500", or "all".')

    args = parser.parse_args()
    return args
class Task4Trainer:
    def __init__(self, config, logger, dataname):
        """Initialize Task4 trainer
        Args:
            config: Configuration parameter dictionary
            logger: Logger
        """
        self.config = config
        self.task_args = config.task_args
        self.logger = logger
        self.dataname = dataname
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model-related configuration
        self.model_name = config.model
        self.epochs = config.epochs
        # Set save path
        if model_name == 'v_GWN':
            if use_pretrain:
                self.save_path = f'./Param/gwn/{dataname}/disturbe/'
            else:
                self.save_path = f'./Param/gwn/{dataname}/disturbe_no_pretrain/'
        elif model_name == 'v_STGCN5':
            if use_pretrain:
                self.save_path = f'./Param/stgcn/{dataname}/disturbe/'
            else:
                self.save_path = f'./Param/stgcn/{dataname}/disturbe_no_pretrain/'
        else:
            print("model_name is not supported")
        os.makedirs(self.save_path, exist_ok=True)
        self.loss_criterion = nn.MSELoss()
        # graph_index -> serial id, used for wandb step
        self.env_order_map = {}
        
    def _evaluate_and_get_rmse(self, model, data_loader, A_wave):
        """Evaluate on first channel; RMSE = sqrt(mean(MSE))."""
        model.eval()
        all_outputs = []
        all_labels = []
        mse_list = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                A_wave = A_wave.to(self.device)
                if model_name == 'v_GWN':
                    adj_mx = [A_wave.float(), (A_wave.float()).t()]
                    out = model(data, adj_mx)
                else:
                    out, meta_graph = model(data, A_wave)
                out = out[:, :1, :]
                label = data.y[:, :1, :]
                all_outputs.append(out.cpu())
                all_labels.append(label.cpu())
                mse_list.append(F.mse_loss(out, label).item())

        outputs = torch.cat(all_outputs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        rmse = float(np.sqrt(np.mean(mse_list))) if mse_list else 0.0
        return rmse, outputs, labels

    def _get_env_serial(self, graph_index):
        if hasattr(graph_index, "__len__"):
            key = tuple(int(x) for x in graph_index)
        else:
            key = int(graph_index)
        return self.env_order_map.get(key, None)

    def _evaluate_rmse_denorm(self, model, data_loader, A_wave, meanstd, return_outputs=False):
        """Calculate denormalized RMSE on specified loader (first channel only), optionally returning output and labels"""
        model.eval()
        outputs_list = []
        labels_list = []
        mse_list = []
        with torch.no_grad():
            A_wave = A_wave.to(self.device)
            for data in data_loader:
                data = data.to(self.device)
                if model_name == 'v_GWN':
                    adj_mx = [A_wave.float(), (A_wave.float()).t()]
                    out = model(data, adj_mx)
                else:
                    out, _ = model(data, A_wave)
                out = out[:, :1, :]
                label = data.y[:, :1, :]
                outputs_list.append(out.cpu())
                labels_list.append(label.cpu())

                mean_val = torch.tensor(meanstd[0], device=out.device, dtype=out.dtype).reshape(1, -1, 1)
                std_val = torch.tensor(meanstd[1], device=out.device, dtype=out.dtype).reshape(1, -1, 1)
                mean_val = mean_val[:, :1, :]
                std_val = std_val[:, :1, :]
                outputs_denorm = out * std_val + mean_val
                labels_denorm = label * std_val + mean_val
                mse_list.append(F.mse_loss(outputs_denorm, labels_denorm).item())

        outputs = torch.cat(outputs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        rmse = float(np.sqrt(np.mean(mse_list))) if mse_list else 0.0

        if return_outputs:
            return rmse, outputs, labels
        return rmse, None, None
    def get_model(self):
        model = StgnnSet(self.config.data_args, 
                            self.config.task_args,
                            self.config.model_args,
                            model=self.model_name).to(self.device)
        return model
    def train_single_graph(self,
                          env_number,
                          model_ori,
                          A_wave,
                          graph_index,
                          train_loader,
                          test_loader,
                          val_loader,
                          train_meanstd,
                          test_meanstd,
                          node0_x1_series=None,
                          node0_x2_series=None,
                          rmse_pretrain=None):
            """Train and evaluate the model for a single graph
            Args:
                graph_index: Graph index
                train_loader: Training data loader
                test_loader: Test data loader 
                train_meanstd: Mean and standard deviation of the training set
                test_meanstd: Mean and standard deviation of the test set
            Returns:
                outputs: Prediction results
                y_label: True labels
            """
            global wandb_steps
            set_seed(42)
            print_memory_usage(f"Graph {graph_index}: Start processing")
            
            # If model weights exist, skip training
 
            # Initialize model and optimizer
            with torch.no_grad():
                model = deepcopy(model_ori).to(self.device)
            
            # Evaluation before training: pretrained weights (no disturbance)
            rmse_log_path = os.path.join(self.save_path, "rmse_log.txt")
            if rmse_pretrain is not None:
                self.logger.info(f"Graph {graph_index}: RMSE pretrain (no disturbance) = {rmse_pretrain:.6f}")
                with open(rmse_log_path, "a") as f:
                    f.write(f"env={env_number}, graph={graph_index}, stage=pretrain, rmse={rmse_pretrain:.6f}\n")
                    f.flush()
                print(f"[RMSE] env={env_number}, graph={graph_index}, stage=pretrain, rmse={rmse_pretrain:.6f}")

            model_path = os.path.join(self.save_path, f'model_graph_{graph_index}_env_{env_number}.pth')
            if os.path.exists(model_path):
                self.logger.info(f"Found model weights for graph {graph_index} at {model_path}, skipping training")
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                final_rmse, outputs, labels = self._evaluate_rmse_denorm(
                    model, test_loader, A_wave, test_meanstd, return_outputs=True
                )
                self.logger.info(f"Graph {graph_index}: RMSE from saved model (denorm) = {final_rmse:.6f}")
                with open(rmse_log_path, "a") as f:
                    f.write(f"env={env_number}, graph={graph_index}, stage=after, rmse={final_rmse:.6f}\n")
                    f.flush()
                print(f"[RMSE] env={env_number}, graph={graph_index}, stage=after, rmse={final_rmse:.6f}")

                # wandb upload: use graph index as step, plot two curves
                env_serial = self._get_env_serial(graph_index)
                run = get_wandb_run(self.dataname if hasattr(self, 'dataname') else 'unknown', env_number)
                payload = {
                    "env_serial": env_serial,
                    "graph_idx": int(graph_index[0]) if hasattr(graph_index, "__len__") else int(graph_index),
                    "disturb_env": env_number,
                    "rmse_after_train": final_rmse,
                }
                if rmse_pretrain is not None:
                    payload["rmse_pretrain"] = rmse_pretrain
                step = env_serial if env_serial is not None else wandb_steps.get(env_number, 0)
                run.log(payload, step=step)
                if env_serial is None:
                    wandb_steps[env_number] = step + 1
                return outputs, labels, final_rmse
            self.logger.info(f"Model weights for graph {graph_index} not found, starting training")

            lr = 1e-3
            weight_decay = 0
            
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            if model_name == 'v_GWN':
                model_sign='gwn'
            else:
                model_sign='stgcn'
            if use_pretrain:
                rootpath=f'Param/{model_sign}/{self.dataname}/disturbe/'
            else:
                rootpath=f'Param/{model_sign}/{self.dataname}/disturbe_no_pretrain/'
            folder = os.path.exists(rootpath + 'Node_{}'.format(graph_index))
            if not folder:
                os.makedirs(rootpath + 'Node_{}' .format(graph_index))
            # Initialize variables before training
            best_loss = float('inf')
            best_model_state = None
            best_optimizer_state = None
            best_epoch = -1
            bad_epochs = 0
            loss_decreasing_in_first_10_epochs = True
            
            # Data for plotting
            rmse_over_epochs = []
            train_rmse_over_epochs = []
            val_rmse_over_epochs = []
            epoch_points = []
            final_outputs, final_labels = None, None
    
            self.logger.info(f"Starting training for graph {graph_index}")
            print_memory_usage(f"Graph {graph_index}: After loading model and data")
            if not use_pretrain:
                self.epochs=21
            # Training loop
            for epoch in tqdm(range(self.epochs)):
                # Training phase
                model.train()
                train_loss = 0
                train_predict_loss = 0
    
                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    data = data.to(device=self.device)
                    A_wave = A_wave.to(device=self.device)
                    if epoch == 0 and i == 0:
                        print_memory_usage(f"Graph {graph_index}, Epoch {epoch}, Batch {i}: After moving data to GPU")
                    
                    # Forward pass
                    if model_name == 'v_GWN':
                        adj_mx = [A_wave.float(), (A_wave.float()).t()]
                        out = model(data, adj_mx)
                    else:
                        out, meta_graph = model(data, A_wave)
                    loss_predict = self.loss_criterion(out, data.y)
                    
                    #loss_reconstruct = self.loss_criterion(meta_graph, A_wave)
                    loss = loss_predict #+ loss_reconstruct
                    if epoch == 0 and i == 0:
                        print_memory_usage(f"Graph {graph_index}, Epoch {epoch}, Batch {i}: After forward pass")
    
                    # Backward pass
                    loss.backward()
                    if epoch == 0 and i == 0:
                        print_memory_usage(f"Graph {graph_index}, Epoch {epoch}, Batch {i}: After backward pass")
    
                    optimizer.step()
                    if epoch == 0 and i == 0:
                        print_memory_usage(f"Graph {graph_index}, Epoch {epoch}, Batch {i}: After optimizer update")
                    train_loss += loss.item()
                    train_predict_loss += loss_predict.item()
                
                train_loss /= len(train_loader)
                train_predict_loss /= len(train_loader)
                #modelparam=torch.load(f"./Param/Task4/A_C_v_STGCN5_finetune_7days/model_graph_{graph_index}.pth")
                #model.load_state_dict(modelparam["model_state_dict"])
                # Validation phase
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device=self.device)
                        A_wave = A_wave.to(device=self.device)
                        if model_name == 'v_GWN':
                            adj_mx = [A_wave.float(), (A_wave.float()).t()]
                            out = model(data, adj_mx)
                        else:
                            out, _ = model(data, A_wave)
                        loss = self.loss_criterion(out, data.y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
    
                # Evaluate and record RMSE on all sets for each epoch
                current_train_rmse = np.sqrt(train_predict_loss)
                train_rmse_over_epochs.append(current_train_rmse)
                
                # Calculate validation RMSE
                current_val_rmse = np.sqrt(val_loss)
                val_rmse_over_epochs.append(current_val_rmse)
    
                test_rmse, current_outputs, current_labels = self._evaluate_and_get_rmse(model, test_loader, A_wave)
                self.logger.info(f"Graph {graph_index}, Epoch {epoch+1}, Train RMSE: {current_train_rmse:.4f}, Val RMSE: {current_val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                rmse_over_epochs.append(test_rmse)
                epoch_points.append(epoch + 1)
                final_outputs, final_labels = current_outputs, current_labels # Save the last evaluation result
    
                # Only record the best model when validation set is better
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    best_optimizer_state = deepcopy(optimizer.state_dict())
                    best_epoch = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1
    
    
            
            # After training, plot and save the chart
            if epoch_points: # Ensure at least one evaluation
                # 1. Plot RMSE trend
                plt.figure(figsize=(12, 7))
                plt.plot(epoch_points, train_rmse_over_epochs, marker='^', linestyle='--', label='Train RMSE')
                plt.plot(epoch_points, val_rmse_over_epochs, marker='s', linestyle='-', color='orange', label='Validation RMSE')
                plt.plot(epoch_points, rmse_over_epochs, marker='o', linestyle='-', color='green', label='Test RMSE')
                plt.title(f'Train vs. Validation vs. Test RMSE for Graph {graph_index}')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                plt.grid(True)
                plt.xticks(epoch_points)
                plt.legend()
                rmse_trend_path = os.path.join(self.save_path, f'rmse_trend_graph_{graph_index}_env_{env_number}.png')
                plt.savefig(rmse_trend_path)
                plt.close()
                self.logger.info(f"RMSE trend plot saved: {rmse_trend_path}")
    
                # 2. Plot Pred-Raw comparison chart (using the last evaluation result)
                if final_outputs is not None and final_labels is not None:
                    # Denormalization
                    mean_val = test_meanstd[0].reshape(1,-1,1)
                    std_val = test_meanstd[1].reshape(1,-1,1)
                    # print(final_outputs.shape, final_labels.shape)
                    # print(mean_val.shape, std_val.shape)
                    outputs_denorm = final_outputs * std_val + mean_val
                    labels_denorm = final_labels * std_val + mean_val
                    outputs_denorm_np = outputs_denorm.numpy()
                    labels_denorm_np = labels_denorm.numpy()
                    pred_np = outputs_denorm_np.flatten()
                    actual_np = labels_denorm_np.flatten()
                    
                    plt.figure(figsize=(8, 8))
                    plt.scatter(actual_np, pred_np, alpha=0.3, s=10)
                    plt.title(f'Prediction vs. Actual for Graph {graph_index} (Epoch {epoch_points[-1]})')
                    plt.xlabel('Actual Values (Raw)')
                    plt.ylabel('Predicted Values (Raw)')
                    
                    # Plot y=x reference line
                    min_val = min(np.min(actual_np), np.min(pred_np))
                    max_val = max(np.max(actual_np), np.max(pred_np))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                    plt.grid(True)
                    
                    scatter_path = os.path.join(self.save_path, f'pred_vs_raw_graph_{graph_index}_env_{env_number}.png')
                    plt.savefig(scatter_path)
                    plt.close()
                    self.logger.info(f"Pred-Raw comparison chart saved: {scatter_path}")
    
                # 3. Plot node 0 time series comparison (x1)
                pred_windows = outputs_denorm_np[:, 0, :]
                pred_node0_series_partial = reconstruct_series(
                    pred_windows,
                    self.task_args.get('pred_num', pred_windows.shape[1]),
                    stride=self.task_args.get('stride', 1),
                )
                if node0_x1_series is not None:
                    gt_node0_series_full = np.asarray(node0_x1_series).reshape(-1)
                else:
                    gt_node0_series_full = labels_denorm_np[:, 0, :].reshape(-1)
                full_len = len(gt_node0_series_full)
                aligned_pred = np.full(full_len, np.nan, dtype=np.float32)
                start_idx = max(0, full_len - len(pred_node0_series_partial))
                end_idx = start_idx + len(pred_node0_series_partial)
                aligned_pred[start_idx:end_idx] = pred_node0_series_partial[: end_idx - start_idx]
                time_axis = np.arange(full_len)

                plt.figure(figsize=(10, 4))
                plt.plot(time_axis, gt_node0_series_full, label="Ground Truth x1", color="tab:blue")
                plt.plot(time_axis, aligned_pred, label="Predicted x1", color="tab:orange", alpha=0.8)
                plt.title(f"Node 0 x1 Trajectory | Graph {graph_index} (Epoch {epoch_points[-1]})")
                plt.xlabel("Time step")
                plt.ylabel("x1")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.3)
                x1_ts_path = os.path.join(
                    self.save_path,
                    f'node0_x1_timeseries_graph_{graph_index}_env_{env_number}.png'
                )
                plt.savefig(x1_ts_path)
                plt.close()
                self.logger.info(f"Node 0 x1 time-series plot saved: {x1_ts_path}")

                timeseries_npz_path = os.path.join(
                    self.save_path,
                    f'node0_timeseries_graph_{graph_index}_env_{env_number}.npz'
                )
                np.savez(
                    timeseries_npz_path,
                    time=time_axis,
                    pred_x1=aligned_pred,
                    gt_x1=gt_node0_series_full,
                    gt_x2=node0_x2_series if node0_x2_series is not None else None,
                )
                self.logger.info(f"Node 0 time-series data saved: {timeseries_npz_path}")

                # 4. Plot node 0 attractor using (x1, x2) phase portrait
                if node0_x1_series is not None and node0_x2_series is not None:
                    gt_x1_full = np.asarray(node0_x1_series).reshape(-1)
                    gt_x2_full = np.asarray(node0_x2_series).reshape(-1)
                    his_num = self.task_args.get('his_num', 0)
                    pred_first = outputs_denorm_np[:, 0, 0]
                    # Align to the start of each prediction window (no smoothing)
                    start = his_num
                    end = start + len(pred_first)
                    end = min(end, len(gt_x1_full), len(gt_x2_full))
                    aligned_len = end - start
                    if aligned_len > 0:
                        gt_x1 = gt_x1_full[start:start + aligned_len]
                        gt_x2 = gt_x2_full[start:start + aligned_len]
                        pred_x1 = pred_first[:aligned_len]
                        paired_x2 = gt_x2

                        plt.figure(figsize=(6, 6))
                        plt.plot(gt_x1, gt_x2, label="Ground Truth (x1 vs x2)", color="tab:blue", linewidth=2)
                        plt.plot(pred_x1, paired_x2, label="Prediction (x1 vs gt x2)", color="tab:orange", linestyle="--")
                        plt.title(f"Node 0 Attractor | Graph {graph_index} (Epoch {epoch_points[-1]})")
                        plt.xlabel("x1")
                        plt.ylabel("x2")
                        plt.grid(True, linestyle="--", alpha=0.4)
                        plt.legend()
                        attractor_path = os.path.join(
                            self.save_path,
                            f'node0_attractor_graph_{graph_index}_env_{env_number}.png'
                        )
                        plt.savefig(attractor_path)
                        plt.close()
                        self.logger.info(f"Node 0 attractor plot saved: {attractor_path}")
    
            # Save the best model only once after training
            if best_model_state is not None:
                model_path = os.path.join(self.save_path, f'model_graph_{graph_index}_env_{env_number}.pth')
                # Ensure the model state is saved on the CPU to avoid device-related issues
                cpu_model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in best_model_state.items()}
                cpu_optimizer_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in best_optimizer_state.items()}
                
                torch.save({
                    'model_state_dict': cpu_model_state,
                    'optimizer_state_dict': cpu_optimizer_state,
                    'epoch': best_epoch,
                    'loss': best_loss
                }, model_path)
                self.logger.info(f"Saved the best model for graph {graph_index} (epoch {best_epoch+1}, loss: {best_loss:.4f})")
            else:
                self.logger.warning(f"No improvement in training for graph {graph_index}, will use the model from the last epoch for evaluation")
                best_model_state = model.state_dict()
    
            # Restore the best model state for evaluation
            model.load_state_dict(best_model_state)
            model.eval()
            print_memory_usage(f"Graph {graph_index}: Preparing for final evaluation")
    
            final_rmse, outputs, labels = self._evaluate_rmse_denorm(
                model, test_loader, A_wave, test_meanstd, return_outputs=True
            )
            self.logger.info(f"Graph {graph_index}: RMSE after training (denorm) = {final_rmse:.6f}")
            with open(rmse_log_path, "a") as f:
                f.write(f"env={env_number}, graph={graph_index}, stage=after, rmse={final_rmse:.6f}\n")
                f.flush()
            print(f"[RMSE] env={env_number}, graph={graph_index}, stage=after, rmse={final_rmse:.6f}")
            # wandb upload: use graph index as step, plot two curves
            env_serial = self._get_env_serial(graph_index)
            run = get_wandb_run(self.dataname if hasattr(self, 'dataname') else 'unknown', env_number)
            payload = {
                "env_serial": env_serial,
                "graph_idx": int(graph_index[0]) if hasattr(graph_index, "__len__") else int(graph_index),
                "disturb_env": env_number,
                "rmse_after_train": final_rmse,
            }
            if rmse_pretrain is not None:
                payload["rmse_pretrain"] = rmse_pretrain
            step = env_serial if env_serial is not None else wandb_steps.get(env_number, 0)
            run.log(payload, step=step)
            if env_serial is None:
                wandb_steps[env_number] = step + 1

            return outputs, labels, final_rmse
    
    def evaluate_all(self, all_outputs, all_labels):
        """Evaluate the overall performance of all graphs"""
        outputs_np = [t.numpy() for t in all_outputs]
        labels_np = [t.numpy() for t in all_labels]
        outputs = np.concatenate(outputs_np, axis=2)
        labels = np.concatenate(labels_np, axis=2)
        metrics = metric_func(
            pred=outputs,
            y=labels,
            times=self.config.task_args['pred_num']
        )
        return metrics

if __name__ == '__main__':
    print("-"*15)
    #path appeared multiple times  
    # Parse arguments
    args = parse_arguments()
    dataname = args.dataname
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration file
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
    
    # Add parameters from the configuration file to args
    args.data_args = config['data']
    args.task_args = config['task']
    args.model_args = config['model']
    
    # Set up logging
    logger, filename = setup_logger(args.taskmode, args.test_dataset, args.logindex, args.model, args.aftername)
    
    # --- Add CUDA diagnostic log ---
    logger.info("--- CUDA Diagnostic Information ---")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device index: {torch.cuda.current_device()}")
        logger.info(f"Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    logger.info(f"Script selected device (args.device): {args.device}")
    logger.info("--------------------")
    
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ifWarp = ""
    if os.path.getsize(filename) != 0:        
        ifWarp = "\n\n"
    logger.info(ifWarp + str(current_time) + ": start training")
    logger.info("model: " + args.model)
    if args.taskmode == 'task4':
        # Initialize trainer
        trainer = Task4Trainer(args, logger, dataname)
        graph_set=np.load(f"./graph_generator/output/{dataname}/parameter_list.npy")
        
        # --- Parallel processing logic starts ---
        num_total_graphs = 400
        if len(graph_set) < num_total_graphs:
            logger.warning(f"Number of available graphs ({len(graph_set)}) is less than requested {num_total_graphs}. Will process all available graphs.")
            num_total_graphs = len(graph_set)
        
        # Select graphs to process based on the graph range parameter
        if args.graph_range == 'A':
            env_range=range(20)
            logger.info(f"Processing all")
        else:  # args.graph_range == 'all'
            print("No graph range selected")
            exit(0)
        ######################
        worker_graph_set = graph_set
        # Build environment serial number table (according to current processing order)
        env_order_map = {}
        for idx, graph_index in enumerate(worker_graph_set):
            key = tuple(int(x) for x in graph_index) if hasattr(graph_index, "__len__") else int(graph_index)
            env_order_map[key] = idx
        trainer.env_order_map = env_order_map
        if not use_pretrain:
            env_range=range(1)
            
        # --- Parallel processing logic ends ---

        # Train on each graph
        all_outputs = []
        all_labels = []
        rmse_list=[]
        for env_number in env_range:
            for graph_index in worker_graph_set:

                A_wave = np.load(f"./graph_generator/output/{dataname}/{dataname}_matrix.npy")
                logger.info(f"Training graph {graph_index}")
                if dataname == 'hill':
                    trajectory_file = os.path.join("./graph_generator/output/hill/",
                                    f"undisturbed_trajectories_a{graph_index[0]}_h{graph_index[1]}_B{graph_index[2]}.csv")
                elif dataname =='fhn2':
                    trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{graph_index[0]}_B_{graph_index[1]}.csv")
                    trajectory_file_x2 = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{graph_index[0]}_B_{graph_index[1]}_x2.csv")
                else:
                    trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/", f"undisturbed_trajectories_R_{graph_index[0]}_B_{graph_index[1]}.csv")
                X_dataset = pd.read_csv(trajectory_file)
                X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
                state_col = X_sorted.columns[2]
                X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
                X=X_pivot.values[:, :, np.newaxis]
                if dataname=='fhn2':
                    X_dataset_x2 = pd.read_csv(trajectory_file_x2)
                    X_sorted_x2 = X_dataset_x2.sort_values(by=[X_dataset_x2.columns[0], X_dataset_x2.columns[1]])
                    state_col_x2 = X_sorted_x2.columns[2]
                    X_pivot_x2 = X_sorted_x2.pivot(index=X_sorted_x2.columns[0], columns=X_sorted_x2.columns[1], values=state_col_x2)
                    X_x2 = X_pivot_x2.values[:, :, np.newaxis]
                    X=np.concatenate((X, X_x2), axis=2)
                # print(X.shape)
                metadatapoint=DynamicsDataset(X,A_wave,args.task_args)
                train_dataset,test_dataset,val_dataset,A_wave=metadatapoint.get_dataset()

                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.task_args['batch_size'],
                    shuffle=True,
                    num_workers=8,
                    pin_memory=False,
                    drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=args.task_args['test_batch_size'],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.task_args['test_batch_size'],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True
                )
                model = trainer.get_model()                # Get the mean and standard deviation of the graph
                train_meanstd = train_dataset.get_normalize_params()
                test_meanstd = test_dataset.get_normalize_params()
                model_ori=deepcopy(model)
                if use_pretrain:
                    if model_name == 'v_GWN':
                        model_ori.load_state_dict(torch.load(f"./Param/v_GWN_small/{dataname}/model_best_GWN.pth", map_location=args.device))
                    else:
                        model_ori.load_state_dict(torch.load(f"./Param/stgcn_small/{dataname}/model_best_{dataname}.pth", map_location=args.device))
                    params = get_params(model_ori)
                    std_params = calculate_std(params)
                    # Calculate RMSE of pre-trained weights (undisturbed) on current graph as baseline
                    rmse_pretrain, _, _ = trainer._evaluate_rmse_denorm(
                        model_ori, test_loader, A_wave, test_meanstd, return_outputs=False
                    )
                    # Add disturbance
                    model_ori = disturbe_params(model_ori, std_params, disturbe_rate=0.001, disturbe_number=env_number)
                    set_seed(env_number)
                else:
                    rmse_pretrain = None
                # Train a single graph
                node0_x1_series = X[:, 0, 0] if X.shape[2] >= 1 else None
                node0_x2_series = X[:, 0, 1] if X.shape[2] >= 2 else None

                outputs, labels, final_rmse = trainer.train_single_graph(
                    env_number,
                    model_ori,
                    A_wave,
                    graph_index,
                    train_loader,
                    test_loader,
                    val_loader,
                    train_meanstd,
                    test_meanstd,
                    node0_x1_series=node0_x1_series,
                    node0_x2_series=node0_x2_series,
                    rmse_pretrain=rmse_pretrain
                )

                rmse_list.append(final_rmse)
                print("rmse for graph", graph_index, "is", final_rmse)
                
        
        