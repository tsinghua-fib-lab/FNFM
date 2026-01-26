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
import psutil
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
###################
model_size='small'
seed=37
def get_data_split(dataname):
    if dataname == 'trust' or dataname == 'bitcoin':
        inner = 100
        circle = 121
        random.seed(42)
        trainid_circle = random.sample(range(inner), int(inner * 0.5))
        testid = list(np.setdiff1d(range(inner), trainid_circle))
        outer = range(inner, circle)
    elif dataname == 'euroad':
        inner = 350
        circle = 400
        random.seed(42)
        trainid_circle = random.sample(range(inner), int(inner * 0.5))
        testid = list(np.setdiff1d(range(inner), trainid_circle))
        outer = range(inner, circle)
    elif dataname == 'hill':
        inner = 350
        circle = 400
        random.seed(42)
        trainid_circle = random.sample(range(inner), int(inner * 0.5))
        testid = list(np.setdiff1d(range(inner), trainid_circle))
        outer = range(inner, circle)
    elif dataname == 'collab':
        random.seed(42)
        observe_circle = list(range(0, 5)) + list(range(15, 40))
        trainid_circle = list(random.sample(observe_circle, int(len(observe_circle) * 0.7)))
        testid = list(random.sample(observe_circle, int(len(observe_circle) * 0.3)))
        outer = range(5, 15)
    elif dataname == 'twitter':
        inner = 100
        circle = 121
        random.seed(42)
        trainid_circle = random.sample(range(inner), int(inner * 0.5))
        testid = list(np.setdiff1d(range(inner), trainid_circle))
        outer = range(inner, circle)
    elif dataname == 'fhn':
        graph_set_path = os.path.join("./graph_generator/output/fhn/parameter_list.npy")
        circle = 400
        graph_set = np.load(graph_set_path)
        e = graph_set[:, 0]
        f = graph_set[:, 1]
        upper_mask = f > e + 0.2
        lower_mask = f < e - 0.2
        upper_ids = np.where(upper_mask)[0]
        lower_ids = np.where(lower_mask)[0]
        observe_circle = list(set(upper_ids) | set(lower_ids))
        random.seed(42)
        trainid_circle = list(random.sample(observe_circle, int(len(observe_circle) * 0.7)))
        testid = list(np.setdiff1d(observe_circle, trainid_circle))
        outer = np.setdiff1d(range(circle), observe_circle)
    else:
        raise ValueError(f"Unsupported dataname: {dataname}")
    return trainid_circle, testid, outer
def print_memory_usage(message=""):
    # GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU Memory] {message}: Allocated={allocated:.2f}MB, Cached={cached:.2f}MB")
    
    # System memory
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    memory_percent = process.memory_percent()
    print(f"[System Memory] {message}: Used={memory_mb:.2f}MB ({memory_percent:.1f}%)")

def clear_memory():
    """Clean up memory and GPU cache"""
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
        loss_reconsturct = loss_criterion(meta_graph, A_wave)
        loss = loss_predict + loss_reconsturct
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.detach().cpu().numpy())
        
    return sum(train_losses)/len(train_losses)

def test_epoch(model, test_dataloader, device):
    """Test the model"""
    model.eval()
    with torch.no_grad():
        for step, (data, A_wave) in enumerate(test_dataloader):
            A_wave = A_wave.to(device=device)
            data = data.to(device=device)
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
    parser.add_argument('--model', default='v_STGCN5', type=str)
    parser.add_argument('--loss_lambda', default=1.5, type=float)
    parser.add_argument('--memo', default='revise', type=str)
    parser.add_argument('--epochs', default=150, type=int)  
    parser.add_argument('--lr',default=0.0001,type=float) 
    parser.add_argument('--dataname', default='collab', type=str)
    parser.add_argument('--taskmode', default='task4', type = str)
    parser.add_argument('--nodeindex', default=0, type = int)
    parser.add_argument('--test_only', default=False, type = bool)
    parser.add_argument('--no_load_model', action='store_true',
                        help='Skip loading any saved/best model weights.')
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
        """Initialize Task4 trainer
        Args:
            config: Configuration parameter dictionary
            logger: Logger
        """
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model-related configuration
        self.model_name = config.model
        self.target_days = config.target_days
        self.epochs = config.epochs
        dataname = config.dataname
        # Set save path
        self.save_path = f'./Param/stgcn_{model_size}/{dataname}/'
        os.makedirs(self.save_path, exist_ok=True)
        self.loss_criterion = nn.MSELoss()
        self.model = None
        self.optimizer = None
       
    def init_model(self):
        """Initialize model and optimizer"""
        self.model = StgnnSet(self.config.data_args, 
                            self.config.task_args,
                            self.config.model_args,
                            model_size=model_size,
                            model=self.model_name).to(self.device)
        
        lr = self.config.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.logger.info(f"Model and optimizer initialized with learning rate {lr}")

    def prepare_dataloaders(self, graphs, A_wave):
        """Load data for all graphs and merge them into a single training and validation DataLoader."""
        self.logger.info("Starting to load and merge data for all graphs...")
        
        all_train_datasets = []
        all_val_datasets = []

        for graph_params in tqdm(graphs, desc="Loading and preparing data",bar_format="{n_fmt}/{total_fmt} used {elapsed} remaining {remaining}"):
            if dataname=='hill':
                a_val, h_val, b_val = graph_params[0], graph_params[1], graph_params[2]
                trajectory_file = os.path.join("./graph_generator/output/hill/",
                                        f"undisturbed_trajectories_a{a_val}_h{h_val}_B{b_val}.csv")
            elif dataname=='fhn':
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
                self.logger.warning(f"Data file not found, skipping: {trajectory_file}")
                continue

            try:
                X_dataset = pd.read_csv(trajectory_file)
                X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
                state_col = X_sorted.columns[2]
                X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
                X = X_pivot.values[:, :, np.newaxis]
                if dataname=='fhn':
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

            except Exception as e:
                self.logger.error(f"Error loading graph data {graph_params}: {e}")
                continue

        if not all_train_datasets or not all_val_datasets:
            self.logger.error("Failed to load any data. Program terminated.")
            return None, None

        # Merge datasets
        combined_train_dataset = ConcatDataset(all_train_datasets)
        combined_val_dataset = ConcatDataset(all_val_datasets)
        
        self.logger.info(f"Data loading complete. Total training samples: {len(combined_train_dataset)}, Total validation samples: {len(combined_val_dataset)}")

        # Create DataLoader
        train_dataloader = DataLoader(
            combined_train_dataset,
            batch_size=self.config.task_args['batch_size'],
            shuffle=True, # Enable data shuffling
            num_workers=4,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False
        )
        
        val_dataloader = DataLoader(
            combined_val_dataset,
            batch_size=self.config.task_args['test_batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False
        )

        return train_dataloader, val_dataloader

    def train_epoch(self, train_dataloader, A_wave):
        """Train for one epoch on the merged dataset"""
        self.model.train()
        total_loss = 0
        total_predict_loss = 0
        A_wave_gpu = torch.tensor(A_wave, dtype=torch.float).to(self.device)
        
        for data in tqdm(train_dataloader, desc="Epoch Training", leave=False,bar_format="{n_fmt}/{total_fmt} used {elapsed} remaining {remaining}"):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            
            out, meta_graph = self.model(data, A_wave_gpu)
            loss_predict = self.loss_criterion(out, data.y)
            loss_reconstruct = self.loss_criterion(meta_graph, A_wave_gpu)
            loss = loss_predict + loss_reconstruct
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_predict_loss += loss_predict.item()

        avg_total_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_predict_loss = total_predict_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        return avg_total_loss, avg_predict_loss

    def evaluate(self, val_dataloader, A_wave,means=None,stds=None):
        """Evaluate the model on a given dataset"""
        self.model.eval()
        all_losses = []
        A_wave_gpu = torch.tensor(A_wave, dtype=torch.float).to(self.device)
        means_expanded, stds_expanded = None, None
        if means is not None and stds is not None:
            # Precompute expanded statistics once to avoid excessive dimensions
            means_expanded = np.expand_dims(np.expand_dims(means, axis=0), axis=2)
            stds_expanded = np.expand_dims(np.expand_dims(stds, axis=0), axis=2)

        with torch.no_grad():
            for data in tqdm(val_dataloader, desc="Evaluating", leave=False,bar_format="{n_fmt}/{total_fmt} used {elapsed} remaining {remaining}"):
                data = data.to(self.device)
                out, _ = self.model(data, A_wave_gpu)
                loss = self.loss_criterion(out, data.y)
                all_losses.append(loss.item())
                
                out=out.detach().cpu().numpy()
                data.y=data.y.detach().cpu().numpy()
                if means_expanded is not None and stds_expanded is not None:
                    out= out*stds_expanded+means_expanded
                    data.y=data.y*stds_expanded+means_expanded
                if out.ndim >= 3 and out.shape[1] > 1:
                    out = out[:, :1, ...]
                    data.y = data.y[:, :1, ...]
                loss = mean_squared_error(np.reshape(out,(out.shape[0],-1)), np.reshape(data.y,(data.y.shape[0],-1)))
                all_losses.append(loss)
        avg_loss = np.mean(all_losses)
        loss_std = np.std(all_losses)
        self.model.train()
        return avg_loss, loss_std

    def get_predictions(self, dataloader, A_wave):
        """Get model predictions on a given dataset"""
        self.model.eval()
        all_outputs = []
        all_labels = []
        A_wave_gpu = torch.tensor(A_wave, dtype=torch.float).to(self.device)

        with torch.no_grad():
            for data in tqdm(dataloader, desc="Getting predictions", leave=False,bar_format="{n_fmt}/{total_fmt} used {elapsed} remaining {remaining}"):
                data = data.to(self.device)
                out, _ = self.model(data, A_wave_gpu)
                all_outputs.append(out.detach().cpu())
                all_labels.append(data.y.detach().cpu())
                
        return all_outputs, all_labels
    
    def evaluate_all(self, all_outputs, all_labels):
        """Evaluate the overall performance of all graphs"""
        # --- FIX: Ensure evaluation is done on NumPy arrays on the CPU ---
        # Convert lists of tensors to NumPy arrays for metric calculation
        outputs_np = [t.numpy() for t in all_outputs]
        labels_np = [t.numpy() for t in all_labels]

        # Concatenate all outputs and labels
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
    set_seed(seed)
    # Parse arguments
    args = parse_arguments()
    dataname = args.dataname
    trainid_circle, testid, outer = get_data_split(dataname)
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

    if args.taskmode == 'task4':
        # Initialize trainer
        trainer = Task4Trainer(args, logger)
        graph_set=np.load(f"./graph_generator/output/{dataname}/parameter_list.npy")

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
            if os.path.exists(trajectory_file):
                existing_graph_set.append(params)
            else:
                print(trajectory_file)
                logger.warning(f"Data file does not exist, skipped parameters: R={params[0]}, B={params[1]}")
        
        logger.info(f"Among the {len(processed_graph_set)} sampled graphs, {len(existing_graph_set)} have valid data files.")

        if not existing_graph_set:
            logger.error("No training data available, program terminated.")
            sys.exit()
        
        # Use the same set of graphs for training and validation, splitting data by different time periods
        train_graphs = np.array(existing_graph_set)
        val_graphs = train_graphs # Use the same graphs for training and validation
        logger.info(f"Will use all {len(train_graphs)} graphs, training on their training periods and validating on their validation periods.")

        # Load the shared adjacency matrix
        A_wave = np.load(f"./graph_generator/output/{dataname}/{dataname}_matrix.npy")
        
        # Initialize model and optimizer
        trainer.init_model()
        model_path = os.path.join(trainer.save_path, f'model_best_{dataname}.pth')
        if os.path.exists(model_path):
            logger.info(f"Found existing weights at {model_path}, skip training.")
            args.test_only = True
        if not args.test_only:
            # --- New data loading logic ---
            logger.info("Using a unified data loading strategy, loading and shuffling all data...")
            train_dataloader, val_dataloader = trainer.prepare_dataloaders(train_graphs, A_wave)

            if train_dataloader is None or val_dataloader is None:
                logger.error("Failed to create training or validation data loaders, program terminated.")
                sys.exit()

            # Training loop
            best_loss = float('inf')
            best_model_state = deepcopy(trainer.model.state_dict())

            best_epoch = -1
            bad_epochs = 0
            
            # History for plotting
            train_rmse_history = []
            val_rmse_history = []
            epoch_history = []
            
            logger.info(f"Starting training for a total of {args.epochs} epochs.")
            logger.info(f"Training configuration: batch_size={args.task_args['batch_size']}, test_batch_size={args.task_args['test_batch_size']}")
            logger.info(f"Each epoch will iterate through all {len(train_dataloader.dataset)} training samples and {len(val_dataloader.dataset)} validation samples.")
            
            # Print memory status before training starts
            print_memory_usage("Before training starts")
            breakflag=0
            for epoch in tqdm(range(args.epochs), desc="Model training in progress",bar_format="{n_fmt}/{total_fmt} used {elapsed} remaining {remaining}"):
                # Clean up memory
                clear_memory()
                
                # 1. Train for one epoch on all training graphs
                train_loss, train_predict_loss = trainer.train_epoch(
                    train_dataloader, A_wave
                )
                
                # 2. Evaluate on all validation graphs
                avg_epoch_val_loss, val_loss_std = trainer.evaluate(val_dataloader, A_wave)
                
                # Calculate RMSE
                train_rmse = np.sqrt(train_predict_loss) if train_predict_loss >= 0 else 0
                val_rmse = np.sqrt(avg_epoch_val_loss) if avg_epoch_val_loss >= 0 else 0
                
                # 3. Check if it is the best model
                if avg_epoch_val_loss < best_loss and breakflag==0:
                    best_loss = avg_epoch_val_loss
                    best_model_state = deepcopy(trainer.model.state_dict())
                    best_epoch = epoch
                    bad_epochs = 0
                    improvement_info = "✓ New best model"
                else:
                    bad_epochs += 1
                    improvement_info = f"No improvement for {bad_epochs} consecutive epochs"
                if bad_epochs >= 2:
                    breakflag=1
                # Record historical data for plotting
                train_rmse_history.append(train_rmse)
                val_rmse_history.append(val_rmse)
                epoch_history.append(epoch + 1)
                
                logger.info(f"Epoch [{epoch+1}/{args.epochs}]")
                logger.info(f"  Training Loss: {train_loss:.6f} | Training RMSE: {train_rmse:.6f}")
                logger.info(f"  Validation Loss: {avg_epoch_val_loss:.6f} | Validation RMSE: {val_rmse:.6f} (Standard Deviation ±{val_loss_std:.6f})")
                logger.info(f"  Best Loss: {best_loss:.6f} (Epoch {best_epoch+1})")
                logger.info(f"  Status: {improvement_info}")
                print_memory_usage(f"End of epoch {epoch+1}")

                # 5. Early stopping check
                if bad_epochs >= 10:  # Set patience to 10
                    logger.info(f"No improvement for {bad_epochs} consecutive epochs, early stopping at epoch {epoch+1}.")
                    break
                if best_model_state:
                    logger.info(f"Saving the best model parameters at epoch {best_epoch+1}, with validation loss: {best_loss:.4f}")
                    model_path = os.path.join(trainer.save_path, f'model_best_{dataname}.pth')
                    torch.save(best_model_state, model_path)
                    logger.info(f"Best model parameters have been saved to {model_path}")
            # After training, plot the RMSE trend graph
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
                logger.info(f"RMSE trend graph has been saved: {rmse_trend_path}")
                
            # After training, process the results
            all_outputs = []
            all_labels = []
            
            if best_model_state:
                logger.info(f"Training complete. Best model found at epoch {best_epoch+1}, with validation loss: {best_loss:.4f}")
                if args.no_load_model:
                    logger.info("no_load_model=True: skip loading best model weights.")
                else:
                    # Load the best model
                    trainer.model.load_state_dict(best_model_state)
                # Save the best model parameters
                model_path = os.path.join(trainer.save_path, f'model_best_{dataname}.pth')
                torch.save(best_model_state, model_path)
                logger.info(f"Best model parameters have been saved to {model_path}")
            # Evaluate on the test set
        else:
            logger.info("test_only=True: load model_best and run evaluation.")
            trainer.model.load_state_dict(torch.load(model_path, map_location=args.device))
            logger.info(f"loaded model from {model_path}")
        logger.info("Starting final evaluation on the test set...")
        
        # Prepare test set data
        test_graphs = graph_set[test_indices]
        outer_graphs = graph_set[outer]
        existing_test_graph_set = []
        mse_list=[]
        outer_mse_list=[]
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
                _, test_dataset, _, _ = metadatapoint.get_dataset()
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
                test_mse, test_loss_std = trainer.evaluate(test_dataloader, A_wave,means,stds)
                mse_list.append(test_mse)
            for parameter in outer_graphs:
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
                    X_dataset = pd.read_csv(trajectory_file)
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
                _, test_dataset, _, _ = metadatapoint.get_dataset()
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
                test_mse, test_loss_std = trainer.evaluate(test_dataloader, A_wave,means,stds)
                outer_mse_list.append(test_mse)
        mse=np.mean(mse_list)
        outer_mse=np.mean(outer_mse_list)
        print(f"Average MSE on the test set: {mse:.6f}")
        print(f"Average MSE on the outer test set: {outer_mse:.6f}")
        rmse=np.sqrt(mse)
        outer_rmse=np.sqrt(outer_mse)
        print(f"Average RMSE on the test set: {rmse:.6f}")
        print(f"Average RMSE on the outer test set: {outer_rmse:.6f}")
