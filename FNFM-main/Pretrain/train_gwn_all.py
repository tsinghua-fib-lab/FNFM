import argparse
import os
import sys
import time
from typing import Any
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
import psutil
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
###################
model_size='small'
dataname="fhn"
seed=47
if dataname=='trust' or dataname=='bitcoin':
    inner=100
    circle=121
    random.seed(42)
    trainid_circle=random.sample(range(inner),int(inner*0.5))
    testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
    outer=range(inner,circle)
elif dataname=='euroad':
    inner=350
    circle=400
    random.seed(42)
    trainid_circle=random.sample(range(inner),int(inner*0.5))
    testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
    outer=range(inner,circle)
elif dataname=='collab':
    random.seed(42)
    observe_circle=list(range(0,5))+list(range(15,40))
    trainid_circle=list(random.sample(observe_circle,int(len(observe_circle)*0.7)))
    testid=genid=list(random.sample(observe_circle,int(len(observe_circle)*0.3)))
    outer=range(5,15)
elif dataname=='hill':
    inner=350
    circle=400
    random.seed(42)
    trainid_circle=random.sample(range(inner),int(inner*0.5))
    testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
    outer=range(inner,circle)
elif dataname=='twitter':
    inner=100
    circle=121
    random.seed(42)
    trainid_circle=random.sample(range(inner),int(inner*0.5))
    testid=genid=list(np.setdiff1d(range(inner),trainid_circle))
    outer=range(inner,circle)
elif dataname=='fhn2' or dataname=='fhn':
    graph_set_path=os.path.join("./graph_generator/output/fhn2/parameter_list.npy")
    circle=400
    graph_set=np.load(graph_set_path)
    trainid=[]
    e = graph_set[:, 0]
    f = graph_set[:, 1]
    upper_mask = f > e + 0.2
    lower_mask = f < e - 0.2
    upper_ids = np.where(upper_mask)[0]
    lower_ids = np.where(lower_mask)[0]
    observe_circle=list(set(upper_ids) | set(lower_ids))
    random.seed(42)
    trainid_circle=list(random.sample(observe_circle,int(len(observe_circle)*0.7)))
    testid=genid=list(np.setdiff1d(observe_circle,trainid_circle))
    outer=np.setdiff1d(range(circle),observe_circle)
###################
def print_memory_usage(message=""):
    # GPU Memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU Memory] {message}: Allocated={allocated:.2f}MB, Cached={cached:.2f}MB")
    
    # System Memory
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    memory_percent = process.memory_percent()
    print(f"[System Memory] {message}: Used={memory_mb:.2f}MB ({memory_percent:.1f}%)")

def clear_memory():
    """Clear memory and GPU cache"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def train_epoch(model, A_wave,train_dataloader, optimizer, loss_criterion, device):
    """Train one epoch"""
    model.train()
    train_losses = []
    
    # Add data check
    print(f"Number of training batches: {len(train_dataloader)}")
    for batch_idx,data in enumerate(train_dataloader):
        # Print data information for the first batch
        if batch_idx == 0:
            print(f"Data shape: {data.x.shape}")
            print(f"Adjacency matrix shape: {A_wave.shape}")
            print(f"Label shape: {data.y.shape}")
            print(f"Data type: {data.x.dtype}")
            print(f"Data range: [{data.x.min():.4f}, {data.x.max():.4f}]")
            if data.y.ndim >= 4 and data.y.shape[-1] >= 2:
                channel_diff = torch.mean(torch.abs(data.y[..., 0] - data.y[..., 1])).item()
                channels_equal = torch.allclose(
                    data.y[..., 0], data.y[..., 1], atol=1e-6
                )
                print(
                    f"Label channel comparison: Channel 0 == Channel 1? {channels_equal} | "
                    f"mean|c0-c1|={channel_diff:.6f}"
                )
            else:
                print("Label channels < 2, cannot compare x1/x2.")
            
        optimizer.zero_grad()
        A_wave = A_wave.float()
        A_wave = A_wave.to(device=device)
        data.x = data.x.to(device=device)
        data.y = data.y.to(device=device)
        
        out = model(data, A_wave)
        loss_predict = loss_criterion(out, data.y)

        loss = loss_predict
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.detach().cpu().numpy())
        
    return sum(train_losses)/len(train_losses)

def test_epoch(model, test_dataloader, device):
    """Test model"""
    model.eval()
    with torch.no_grad():
        for step, (data, A_wave) in enumerate(test_dataloader):
            A_wave = A_wave.to(device=device)
            data.x = data.x.to(device=device)
            data.y = data.y.to(device=device)
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
    parser.add_argument('--config_filename', default='config.yaml', type=str,
                            help='Configuration filename for restoring the model.')
    parser.add_argument('--test_dataset', default='metr-la', type=str)
    parser.add_argument('--meta_dim', default=32, type=int)
    parser.add_argument('--target_days', default=15, type=int)
    parser.add_argument('--model', default='v_GWN', type=str)
    parser.add_argument('--loss_lambda', default=1.5, type=float)
    parser.add_argument('--memo', default='revise', type=str)
    parser.add_argument('--epochs', default=150, type=int)  
    parser.add_argument('--train_mode', action="store_true")
    
    parser.add_argument('--taskmode', default='task4', type = str)
    parser.add_argument('--nodeindex', default=0, type = int)
    parser.add_argument("--ifchosenode", action="store_true")
    parser.add_argument('--logindex', default='0', type = str)
    parser.add_argument('--ifspatial',default=1, type = int)  
    parser.add_argument('--ifnewname',default=0, type = int)
    parser.add_argument('--aftername',default='', type = str) 
    parser.add_argument('--datanum',default=0.7, type = float)  
    parser.add_argument('--data_args',default="Dynamics", type = str)  


    args = parser.parse_args()
    args.epochs=150
    print(time.strftime('%Y-%m-%d %H:%M:%S'), "meta_dim = ", args.meta_dim,"target_days = ", args.target_days)
    return args
class Task4Trainer:
    def __init__(self, config, logger):
        """Initialize Task4 Trainer
        Args:
            config: Configuration dictionary
            logger: Logger
        """
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model related configuration
        self.model_name = config.model
        self.target_days = config.target_days
        self.epochs = config.epochs
        
        # Set save path, choose different path based on model type
        if self.model_name == 'v_GWN':
            self.save_path = f'./Param/v_GWN_{model_size}/{dataname}/'
        else:
            self.save_path = f'./Param/{self.model_name.lower()}/{dataname}/'
        os.makedirs(self.save_path, exist_ok=True)
        self.loss_criterion = nn.MSELoss()
        self.model = None
        self.optimizer = None
    
    def init_model(self,load_best=False):
        """Initialize model and optimizer"""
        self.model = StgnnSet(self.config.data_args, 
                            self.config.task_args,
                            self.config.model_args,
                            model=self.model_name,model_size=model_size).to(self.device)
        if load_best:
            if os.path.exists(os.path.join(self.save_path, 'model_best.pth')):
                self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'model_best.pth'), map_location=self.device)["model_state_dict"])
                self.logger.info(f"successfully loaded best model: {os.path.join(self.save_path, 'model_best.pth')}")
            else:
                exit("cannot find best model")
        if not load_best:
            lr = self.config.model_args['meta_lr']
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self.logger.info(f"model and optimizer initialized with learning rate {lr}")

    def prepare_dataloaders(self, graphs, A_wave):
        """Load data for all graphs and merge them into a training and validation DataLoader."""
        self.logger.info("Start loading and merging data for all graphs...")
        
        all_train_datasets = []
        all_val_datasets = []
        all_train_data = []
        all_val_data = []
        all_test_data=[]
        for graph_params in tqdm(graphs, desc="Loading and preparing data",bar_format="{n_fmt}/{total_fmt} elapsed {elapsed} remaining {remaining}"):
            if dataname=='hill':
                a_val, h_val, b_val = graph_params[0], graph_params[1], graph_params[2]
                trajectory_file = os.path.join("./graph_generator/output/hill/",
                                        f"undisturbed_trajectories_a{a_val}_h{h_val}_B{b_val}.csv")
            elif dataname=='fhn2':
                r_val, b_val = graph_params[0], graph_params[1]
                trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                trajectory_file_x2 = os.path.join(f"./graph_generator/output/{dataname}/",
                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}_x2.csv")
            else:
                r_val, b_val = graph_params[0], graph_params[1]
                trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
            if not os.path.exists(trajectory_file):
                self.logger.warning(f"Data file does not exist, skipping: {trajectory_file}")
                continue

            try:
                X_dataset = pd.read_csv(trajectory_file)
                X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
                state_col = X_sorted.columns[2]
                X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
                X = X_pivot.values[:, :, np.newaxis]
                if dataname=='fhn2':
                    X_dataset_x2 = pd.read_csv(trajectory_file_x2)
                    X_sorted_x2 = X_dataset_x2.sort_values(by=[X_dataset_x2.columns[0], X_dataset_x2.columns[1]])
                    state_col_x2 = X_sorted_x2.columns[2]
                    X_pivot_x2 = X_sorted_x2.pivot(index=X_sorted_x2.columns[0], columns=X_sorted_x2.columns[1], values=state_col_x2)
                    X_x2 = X_pivot_x2.values[:, :, np.newaxis]
                    X=np.concatenate((X, X_x2), axis=2)
                metadatapoint = DynamicsDataset(X, A_wave, self.config.task_args)
                train_dataset, _, val_dataset, _ = metadatapoint.get_dataset()

                all_train_datasets.append(train_dataset)
                all_val_datasets.append(val_dataset)
                # all_train_data.append(X_train)
                # all_val_data.append(X_val)
                # all_test_data.append(X_test)
            except Exception as e:
                self.logger.error(f"Error loading graph data {graph_params}: {e}")
                continue

        if not all_train_datasets or not all_val_datasets:
            self.logger.error("Failed to load any data. Program terminating.")
            return None, None

        # Merge datasets
        combined_train_dataset = ConcatDataset(all_train_datasets)
        combined_val_dataset = ConcatDataset(all_val_datasets)
        # train_data=np.concatenate(all_train_data, axis=0)
        # val_data=np.concatenate(all_val_data, axis=0)
        # test_data=np.concatenate(all_test_data, axis=0)
        # print(train_data.shape)
        # print(val_data.shape)
        # print(test_data.shape)
        self.logger.info(f"Data loading completed. Total training samples: {len(combined_train_dataset)}, total validation samples: {len(combined_val_dataset)}")
        # Create DataLoader
        batch_size=self.config.task_args['batch_size']
        if dataname=='euroad':
            batch_size=4
        train_dataloader = DataLoader(
            combined_train_dataset,
            batch_size=batch_size,
            shuffle=True, # Enable data shuffling
            num_workers=4,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False
        )
        
        val_dataloader = DataLoader(
            combined_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False
        )

        return train_dataloader, val_dataloader

    def prepare_test_dataloaders(self, test_graphs, A_wave):
        """Load test graph data and create test DataLoader"""
        self.logger.info("Start loading test graph data...")
        
        all_test_datasets = []

        for graph_params in tqdm(test_graphs, desc="Loading test data",bar_format="{n_fmt}/{total_fmt} elapsed {elapsed} remaining {remaining}"):
            if dataname=='hill':
                a_val, h_val, b_val = graph_params[0], graph_params[1], graph_params[2]
                trajectory_file = os.path.join("./graph_generator/output/hill/",
                                        f"undisturbed_trajectories_a{a_val}_h{h_val}_B{b_val}.csv")
            else:
                r_val, b_val = graph_params[0], graph_params[1]
                trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
            
            if not os.path.exists(trajectory_file):
                self.logger.warning(f"Test data file does not exist, skipping: {trajectory_file}")
                continue

            try:
                X_dataset = pd.read_csv(trajectory_file)
                X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
                state_col = X_sorted.columns[2]
                X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
                X = X_pivot.values[:, :, np.newaxis]
                
                metadatapoint = DynamicsDataset(X, A_wave, self.config.task_args)
                _, test_dataset, _, _ = metadatapoint.get_dataset()
                
                all_test_datasets.append(test_dataset)

            except Exception as e:
                self.logger.error(f"Error loading test graph data {graph_params}: {e}")
                continue

        if not all_test_datasets:
            self.logger.error("Failed to load any test data.")
            return None

        # Merge test datasets
        combined_test_dataset = ConcatDataset(all_test_datasets)
        
        self.logger.info(f"Test data loading completed. Total test samples: {len(combined_test_dataset)}")

        # Create test DataLoader
        test_dataloader = DataLoader(
            combined_test_dataset,
            batch_size=self.config.task_args['test_batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False
        )

        return test_dataloader

    def train_epoch(self, train_dataloader, A_wave):
        """Train one epoch on the merged dataset"""
        self.model.train()
        total_loss = 0
        total_predict_loss = 0
        A_wave_gpu = torch.tensor(A_wave, dtype=torch.float).to(self.device)
        
        for data in tqdm(train_dataloader, desc="Epoch Training", leave=False,bar_format="{n_fmt}/{total_fmt} elapsed {elapsed} remaining {remaining}"):
            self.optimizer.zero_grad()
            data.x = data.x.to(self.device)
            data.y = data.y.to(self.device)
            self.model_name='v_GWN'
            # Adapt specific input for GWN model (bidirectional adjacency matrix)
            if self.model_name == 'v_GWN':
                adj_mx = [A_wave_gpu, A_wave_gpu.t()]
                out = self.model(data, adj_mx)
                meta_graph = A_wave
            else:
                meta_graph,out = self.model(data, A_wave_gpu)

            
            # Calculate prediction loss
            loss_predict = self.loss_criterion(out, data.y)
            
        
            
            loss = loss_predict
            #print_memory_usage("During process")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_predict_loss += loss_predict.item()
 
        avg_total_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_predict_loss = total_predict_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        return avg_total_loss, avg_predict_loss

    def evaluate(self, val_dataloader, A_wave,means=None,stds=None):
        """Evaluate model on the given dataset"""
        self.model.eval()
        all_losses = []
        A_wave_gpu = torch.tensor(A_wave, dtype=torch.float).to(self.device)

        with torch.no_grad():
            # Align shapes only once before the loop starts to avoid increasing dimensions inside the batch loop
            if means is not None and stds is not None:
                means_b = np.reshape(means, (1, -1, 1))
                stds_b = np.reshape(stds, (1, -1, 1))
            else:
                means_b = None
                stds_b = None
            for data in tqdm(val_dataloader, desc="Evaluating", leave=False,bar_format="{n_fmt}/{total_fmt} elapsed {elapsed} remaining {remaining}"):
                data = data.to(self.device)
                
                # Adapt specific input for GWN model (bidirectional adjacency matrix)
                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave_gpu, A_wave_gpu.t()]
                    out = self.model(data, adj_mx)
                    meta_graph = A_wave
                else:
                    out = self.model(data, A_wave_gpu)
                    meta_graph = A_wave
                out=out.detach().cpu().numpy()
                data.y=data.y.detach().cpu().numpy()
                if means_b is not None and stds_b is not None:
                    out = out * stds_b + means_b
                    data.y = data.y * stds_b + means_b
                loss = mean_squared_error(np.reshape(out,(out.shape[0],-1)), np.reshape(data.y,(data.y.shape[0],-1)))
                all_losses.append(loss)
        avg_loss = np.mean(all_losses)
        loss_std = np.std(all_losses)
        self.model.train()
        return avg_loss, loss_std

    def evaluate_rmse_evalparams(self, test_dataset, A_wave, train_meanstd):
        """Align with FNFM PredictionModel evalparams evaluation metric (per-graph RMSE)"""
        self.model.eval()
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        A_wave_gpu = torch.tensor(A_wave, dtype=torch.float).to(self.device)
        mean_tensor = torch.tensor(train_meanstd[0], device=self.device, dtype=torch.float32).reshape(1, -1, 1)
        std_tensor = torch.tensor(train_meanstd[1], device=self.device, dtype=torch.float32).reshape(1, -1, 1)
        outputs_list = []
        labels_list = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave_gpu, A_wave_gpu.t()]
                    out = self.model(data, adj_mx)
                else:
                    out = self.model(data, A_wave_gpu)
                out_denorm = out * std_tensor + mean_tensor
                label_denorm = data.y * std_tensor + mean_tensor
                out_denorm = out_denorm[:, :1, :]
                label_denorm = label_denorm[:, :1, :]
                outputs_list.append(out_denorm)
                labels_list.append(label_denorm)
        if not outputs_list:
            self.model.train()
            return 0.0, 0.0
        outputs = torch.cat(outputs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        mse = F.mse_loss(outputs, labels).item()
        rmse = float(np.sqrt(mse))
        self.model.train()
        return rmse, mse

    def get_predictions(self, dataloader, A_wave):
        """Get model predictions on the given dataset"""
        self.model.eval()
        all_outputs = []
        all_labels = []
        A_wave_gpu = torch.tensor(A_wave, dtype=torch.float).to(self.device)

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Getting predictions", leave=False,bar_format="{n_fmt}/{total_fmt} elapsed {elapsed} remaining {remaining}"):
                data = data.to(self.device)
                
                # Adapt specific input for GWN model (bidirectional adjacency matrix)
                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave_gpu, A_wave_gpu.t()]
                    out = self.model(data, adj_mx)
                    meta_graph = A_wave
                else:
                    out = self.model(data, A_wave_gpu)
                    meta_graph = A_wave
                
                all_outputs.append(out.detach().cpu())
                all_labels.append(data.y.detach().cpu())
                
        return all_outputs, all_labels
    
    def evaluate_all(self, all_outputs, all_labels):
        """Evaluate overall performance of all graphs"""
        # --- FIX: Ensure evaluation is done on NumPy arrays on the CPU ---
        # Convert lists of tensors to NumPy arrays for metric calculation
        outputs_np = [t.numpy() for t in all_outputs]
        labels_np = [t.numpy() for t in all_labels]

        # Merge all outputs and labels
        outputs = np.concatenate(outputs_np, axis=2)
        labels = np.concatenate(labels_np, axis=2)
        
        # Calculate overall metrics
        metrics = metric_func(
            pred=outputs,
            y=labels,
            times=self.config.task_args['pred_num']
        )
        
        return metrics

if __name__ == '__main__':  
    # Parse arguments
    args = parse_arguments()
    set_seed(seed)
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration file
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
    
    # Add parameters from configuration file to args
    args.data_args = config['data']
    args.task_args = config['task']
    args.model_args = config['model']
    
    # Set logger
    logger, filename = setup_logger(args.taskmode, args.test_dataset, args.logindex, args.model, args.aftername)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ifWarp = ""
    if os.path.getsize(filename) != 0:        
        ifWarp = "\n\n"
    logger.info(ifWarp + str(current_time) + ": start training")
    logger.info("target dataset: %s" % args.test_dataset)
    logger.info("model: " + args.model)
    logger.info("taskmode: " + args.taskmode)
    logger.info("ifchosenode: " + str(args.ifchosenode))
    logger.info("logindex: " + str(args.logindex))
    test_count=0# Record a test
    if args.taskmode == 'task4':
        # Initialize trainer
        trainer = Task4Trainer(args, logger)
        graph_set=np.load(f"./graph_generator/output/{dataname}/parameter_list.npy")
        
    # Randomly sample from available graphs
        # Set random seed to ensure reproducibility
        selected_indices = trainid_circle
        test_indices = testid
        # Get graph data based on selected indices
        processed_graph_set = graph_set[selected_indices]

        # Check if data files exist and create the final parameter list
        existing_graph_set = []
        for params in processed_graph_set:
            if dataname=='hill':
                a_val, h_val, b_val = params[0], params[1], params[2]
                trajectory_file = os.path.join("./graph_generator/output/hill/",
                                        f"undisturbed_trajectories_a{a_val}_h{h_val}_B{b_val}.csv")
            else:
                r_val, b_val = params[0], params[1]
                trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
            if os.path.exists(trajectory_file):
                existing_graph_set.append(params)
            else:
                print(trajectory_file)
                logger.warning(f"Data file does not exist, skipped parameters: R={params[0]}, B={params[1]}")
        
        logger.info(f"Among {len(processed_graph_set)} extracted graphs, {len(existing_graph_set)} has valid data files.")

        if not existing_graph_set:
            logger.error("No available training data, program terminating.")
            sys.exit()
        
        # Use the same set of graphs for training and validation, splitting data by different time periods
        train_graphs = np.array(existing_graph_set)
        val_graphs = train_graphs # Use the same graphs for training and validation
        logger.info(f"Will use all {len(train_graphs)} graphs, train on their training periods, and validate on validation periods.")


        # Load shared adjacency matrix
        A_wave = np.load(f"./graph_generator/output/{dataname}/{dataname}_matrix.npy")

        if args.train_mode:
            trainer.init_model()
        else:
            trainer.init_model(load_best=True)
        
        logger.info("Adopt unified data loading strategy, load all data and shuffle...")
        train_dataloader, val_dataloader= trainer.prepare_dataloaders(train_graphs, A_wave)

        if train_dataloader is None or val_dataloader is None:
            logger.error("Cannot create training or validation data loaders, program terminating.")
            sys.exit()
        if args.train_mode:
            # Training loop
            best_loss = float('inf')
            best_model_state = deepcopy(trainer.model.state_dict())

            best_epoch = -1
            bad_epochs = 0
            
            # History records for plotting
            train_rmse_history = []
            val_rmse_history = []
            epoch_history = []
            
            logger.info(f"Start training, total {args.epochs} epochs.")
            logger.info(f"Training config: batch_size={args.task_args['batch_size']}, test_batch_size={args.task_args['test_batch_size']}")
            logger.info(f"Learning rate: {args.model_args['meta_lr']}")
            logger.info(f"Each epoch will traverse all {len(train_dataloader.dataset)} training samples and {len(val_dataloader.dataset)} validation samples.")
            
            # Print memory status before training starts
            #print_memory_usage("Before training starts")
            model_path = os.path.join(trainer.save_path, f'model_best_GWN.pth')
            logger.info(f"Best model parameters saved to {model_path}")
            for epoch in tqdm(range(args.epochs), desc="Model training",bar_format="{n_fmt}/{total_fmt} elapsed {elapsed} remaining {remaining}"):
                # Clear memory
                clear_memory()
                
                # 1. Train one epoch on all training graphs
                train_loss, train_predict_loss = trainer.train_epoch(
                    train_dataloader, A_wave
                )
                
                # 2. Evaluate on all validation graphs
                avg_epoch_val_loss, val_loss_std = trainer.evaluate(val_dataloader, A_wave)
                
                # Calculate RMSE
                train_rmse = np.sqrt(train_predict_loss) if train_predict_loss >= 0 else 0
                val_rmse = np.sqrt(avg_epoch_val_loss) if avg_epoch_val_loss >= 0 else 0
                if test_count ==0:
                    test_count=1
                    # Prepare test set data
                    test_graphs = graph_set[test_indices]
                    outer_graphs = graph_set[outer]
                    existing_test_graph_set = []
                    mse_list=[]
                    rmse_list=[]
                    outer_mse_list=[]
                    outer_rmse_list=[]
                    with torch.no_grad():
                        for params in test_graphs:
                            if dataname=='hill':
                                a_val, h_val, b_val = params[0], params[1], params[2]
                                trajectory_file = os.path.join("./graph_generator/output/hill/",
                                                        f"undisturbed_trajectories_a{a_val}_h{h_val}_B{b_val}.csv")
                            elif dataname=='fhn2':
                                r_val, b_val = params[0], params[1]
                                trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                                trajectory_file_x2 = os.path.join(f"./graph_generator/output/{dataname}/",
                                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}_x2.csv")
                            else:
                                r_val, b_val = params[0], params[1]
                                trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                            X_dataset = pd.read_csv(trajectory_file)
                            X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
                            state_col = X_sorted.columns[2]
                            X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
                            X = X_pivot.values[:, :, np.newaxis]
                            if dataname=='fhn2':
                                X_dataset_x2 = pd.read_csv(trajectory_file_x2)
                                X_sorted_x2 = X_dataset_x2.sort_values(by=[X_dataset_x2.columns[0], X_dataset_x2.columns[1]])
                                state_col_x2 = X_sorted_x2.columns[2]
                                X_pivot_x2 = X_sorted_x2.pivot(index=X_sorted_x2.columns[0], columns=X_sorted_x2.columns[1], values=state_col_x2)
                                X_x2 = X_pivot_x2.values[:, :, np.newaxis]
                                X=np.concatenate((X, X_x2), axis=2)
                            metadatapoint = DynamicsDataset(X, A_wave, args.task_args)
                            train_dataset, test_dataset, _, _ = metadatapoint.get_dataset()
                            means,stds= metadatapoint.get_normalize_params()
                            test_dataloader= DataLoader(
                                test_dataset,
                                batch_size=args.task_args['test_batch_size'],
                                shuffle=False,
                                num_workers=0,
                                pin_memory=False,
                                drop_last=True,
                                persistent_workers=False
                            )
                            rmse, mse = trainer.evaluate_rmse_evalparams(test_dataset, A_wave, train_dataset.get_normalize_params())
                            rmse_list.append(rmse)
                            mse_list.append(mse)
                        for params in outer_graphs:
                            if dataname=='hill':
                                a_val, h_val, b_val = params[0], params[1], params[2]
                                trajectory_file = os.path.join("./graph_generator/output/hill/",
                                                        f"undisturbed_trajectories_a{a_val}_h{h_val}_B{b_val}.csv")
                            elif dataname=='fhn2':
                                r_val, b_val = params[0], params[1]
                                trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                                trajectory_file_x2 = os.path.join(f"./graph_generator/output/{dataname}/",
                                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}_x2.csv")
                            else:
                                r_val, b_val = params[0], params[1]
                                trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                            f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                            X_dataset = pd.read_csv(trajectory_file)
                            X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
                            state_col = X_sorted.columns[2]
                            X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
                            X = X_pivot.values[:, :, np.newaxis]    
                            if dataname=='fhn2':
                                X_dataset_x2 = pd.read_csv(trajectory_file_x2)
                                X_sorted_x2 = X_dataset_x2.sort_values(by=[X_dataset_x2.columns[0], X_dataset_x2.columns[1]])
                                state_col_x2 = X_sorted_x2.columns[2]
                                X_pivot_x2 = X_sorted_x2.pivot(index=X_sorted_x2.columns[0], columns=X_sorted_x2.columns[1], values=state_col_x2)
                                X_x2 = X_pivot_x2.values[:, :, np.newaxis]
                                X=np.concatenate((X, X_x2), axis=2)
                            metadatapoint = DynamicsDataset(X, A_wave, args.task_args)
                            train_dataset, test_dataset, _, _ = metadatapoint.get_dataset()
                            means,stds= metadatapoint.get_normalize_params()
                            test_dataloader= DataLoader(
                                test_dataset,
                                batch_size=args.task_args['test_batch_size'],
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False,
                                drop_last=True,
                                persistent_workers=False
                            )
                            rmse, mse = trainer.evaluate_rmse_evalparams(test_dataset, A_wave, train_dataset.get_normalize_params())
                            outer_rmse_list.append(rmse)
                            outer_mse_list.append(mse)
                        mse=np.mean(mse_list)
                        outer_mse=np.mean(outer_mse_list)
                        print(f"Average MSE on test set: {mse:.6f}")
                        print(f"Average MSE on external test set: {outer_mse:.6f}")
                        rmse=np.mean(rmse_list) if rmse_list else 0.0
                        outer_rmse=np.mean(outer_rmse_list) if outer_rmse_list else 0.0
                        print(f"Average RMSE on external test set: {outer_rmse:.6f}")
                        print(f"Average RMSE on test set: {rmse:.6f}")
                # 3. Check if it is the best model
                if avg_epoch_val_loss < best_loss:
                    best_loss = avg_epoch_val_loss
                    best_model_state = deepcopy(trainer.model.state_dict())
                    best_epoch = epoch
                    bad_epochs = 0
                    improvement_info = "✓ New best model"
                    np.save(os.path.join(trainer.save_path, f'best_model_state'+str(epoch)+'.npy'), best_model_state)
                else:
                    bad_epochs += 1
                    improvement_info = f"No improvement for {bad_epochs} consecutive epochs"

                # Record history data for plotting
                train_rmse_history.append(train_rmse)
                val_rmse_history.append(val_rmse)
                epoch_history.append(epoch + 1)
                
                logger.info(f"Epoch [{epoch+1}/{args.epochs}]")
                logger.info(f"  Train Loss: {train_loss:.6f} | Train RMSE: {train_rmse:.6f}")
                logger.info(f"  Val Loss: {avg_epoch_val_loss:.6f} | Val RMSE: {val_rmse:.6f} (std ±{val_loss_std:.6f})")
                logger.info(f"  Best Loss: {best_loss:.6f} (Epoch {best_epoch+1})")
                logger.info(f"  Status: {improvement_info}")
                #print_memory_usage(f"Epoch {epoch+1} ended")

                # 5. Early stopping check
                if bad_epochs >= 1:  
                    logger.info(f"Performance has not improved for {bad_epochs} consecutive epochs, early stopping at epoch {epoch+1}.")
                    break
            
            # Training ended, plot RMSE trend
            if epoch_history:
                plt.figure(figsize=(12, 7))
                plt.plot(epoch_history, train_rmse_history, marker='^', linestyle='--', label='Train RMSE')
                plt.plot(epoch_history, val_rmse_history, marker='s', linestyle='-', label='Validation RMSE')
                plt.title('Train vs. Validation RMSE Trend (Multi-Graph)')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                plt.grid(True)
                plt.legend()
                rmse_trend_path = os.path.join(trainer.save_path, 'rmse_trend_multi_graph.png')
                plt.savefig(rmse_trend_path)
                plt.close()
                logger.info(f"Saved RMSE trend plot: {rmse_trend_path}")
                
        # Training ended, process results
        all_outputs = []
        all_labels = []
        
        if best_model_state:
            logger.info(f"Training completed. Best model found at epoch {best_epoch+1}, validation loss: {best_loss:.4f}")
            
            # Load best model
            trainer.model.load_state_dict(best_model_state)

            # Save best model parameters
            model_path = os.path.join(trainer.save_path, f'model_best_GWN_two_features.pth')
            torch.save(best_model_state, model_path)
            logger.info(f"Best model parameters saved to {model_path}")

        # Evaluate on test set
        logger.info("Start final evaluation on the test set...")
        
        # Prepare test set data
        test_graphs = graph_set[test_indices]
        outer_graphs = graph_set[outer]
        existing_test_graph_set = []
        mse_list=[]
        rmse_list=[]
        outer_mse_list=[]
        outer_rmse_list=[]
        with torch.no_grad():
            for params in test_graphs:
                if dataname=='hill':
                    a_val, h_val, b_val = params[0], params[1], params[2]
                    trajectory_file = os.path.join("./graph_generator/output/hill/",
                                            f"undisturbed_trajectories_a{a_val}_h{h_val}_B{b_val}.csv")
                elif dataname=='fhn2':
                    r_val, b_val = params[0], params[1]
                    trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                    trajectory_file_x2 = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{r_val}_B_{b_val}_x2.csv")
                else:
                    r_val, b_val = params[0], params[1]
                    trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                X_dataset = pd.read_csv(trajectory_file)
                X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
                state_col = X_sorted.columns[2]
                X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
                X = X_pivot.values[:, :, np.newaxis]    
                metadatapoint = DynamicsDataset(X, A_wave, args.task_args)
                train_dataset, test_dataset, _, _ = metadatapoint.get_dataset()
                means,stds= metadatapoint.get_normalize_params()
                test_dataloader= DataLoader(
                    test_dataset,
                    batch_size=args.task_args['test_batch_size'],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True,
                    persistent_workers=False
                )
                rmse, mse = trainer.evaluate_rmse_evalparams(test_dataset, A_wave, train_dataset.get_normalize_params())
                rmse_list.append(rmse)
                mse_list.append(mse)
            for params in outer_graphs:
                if dataname=='hill':
                    a_val, h_val, b_val = params[0], params[1], params[2]
                    trajectory_file = os.path.join("./graph_generator/output/hill/",
                                            f"undisturbed_trajectories_a{a_val}_h{h_val}_B{b_val}.csv")
                elif dataname=='fhn2':
                    r_val, b_val = params[0], params[1]
                    trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                    trajectory_file_x2 = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{r_val}_B_{b_val}_x2.csv")
                else:
                    r_val, b_val = params[0], params[1]
                    trajectory_file = os.path.join(f"./graph_generator/output/{dataname}/",
                                                f"undisturbed_trajectories_R_{r_val}_B_{b_val}.csv")
                X_dataset = pd.read_csv(trajectory_file)
                X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
                state_col = X_sorted.columns[2]
                X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
                X = X_pivot.values[:, :, np.newaxis]
                if dataname=='fhn2':
                    X_dataset_x2 = pd.read_csv(trajectory_file_x2)
                    X_sorted_x2 = X_dataset_x2.sort_values(by=[X_dataset_x2.columns[0], X_dataset_x2.columns[1]])
                    state_col_x2 = X_sorted_x2.columns[2]
                    X_pivot_x2 = X_sorted_x2.pivot(index=X_sorted_x2.columns[0], columns=X_sorted_x2.columns[1], values=state_col_x2)
                    X_x2 = X_pivot_x2.values[:, :, np.newaxis]
                    X=np.concatenate((X, X_x2), axis=2)
                metadatapoint = DynamicsDataset(X, A_wave, args.task_args)
                train_dataset, test_dataset, _, _ = metadatapoint.get_dataset()
                means,stds= metadatapoint.get_normalize_params()
                test_dataloader= DataLoader(
                    test_dataset,
                    batch_size=args.task_args['test_batch_size'],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,
                    drop_last=True,
                    persistent_workers=False
                )
                rmse, mse = trainer.evaluate_rmse_evalparams(test_dataset, A_wave, train_dataset.get_normalize_params())
                outer_rmse_list.append(rmse)
                outer_mse_list.append(mse)
                    
        mse=np.mean(mse_list)
        outer_mse=np.mean(outer_mse_list)
        print(f"Average MSE on test set: {mse:.6f}")
        print(f"Average MSE on external test set: {outer_mse:.6f}")
        rmse=np.mean(rmse_list) if rmse_list else 0.0
        outer_rmse=np.mean(outer_rmse_list) if outer_rmse_list else 0.0
        print(f"Average RMSE on external test set: {outer_rmse:.6f}")
        print(f"Average RMSE on test set: {rmse:.6f}")