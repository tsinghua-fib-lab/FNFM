#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder training module based on LayerVAE
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import yaml
import os
import math
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from .layer_tokenizer_transformer import LayerVAE
from typing import Dict, List, Tuple, Optional # Import necessary for type hints


class SimpleParameterDataset(Dataset):
    """Dataset that directly handles flattened parameters"""
    def __init__(self, parameters, kgEmbs=None, timeEmbs=None):
        self.parameters = parameters
        # Keep kgEmbs and timeEmbs for interface compatibility
        self.kgEmbs = kgEmbs if kgEmbs is not None else torch.zeros((len(parameters), 1))
        self.timeEmbs = timeEmbs if timeEmbs is not None else torch.zeros((len(parameters), 1))
        
    def __len__(self):
        return len(self.parameters)
    
    def __getitem__(self, idx):
        param = self.parameters[idx]
        # Convert numpy array to torch tensor
        param = torch.tensor(param, dtype=torch.float32)
        if param.dim() > 1:
            param = param.flatten()
        return idx, param, self.kgEmbs[idx], self.timeEmbs[idx]


class LayerAE_Trainer: # Name can be changed to LayerVAE_Trainer
    """LayerVAE trainer"""
    def __init__(self, config, dataloader, exp_index=None, results_folder=None, path_config=None,genid=None):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.scale = config.get("scale")
        self.path_config = path_config
        self.exp_index = exp_index if exp_index is not None else config.get("exp_index", "default")
        self.results_folder = Path(results_folder) if results_folder else Path(f'./ModelSave/layer_vae_exp{self.exp_index}')
        self.results_folder.mkdir(parents=True, exist_ok=True)
        print(f"Layer VAE models will be saved to: {self.results_folder}")
        self.genid=genid
        self.writer = config.get("writer",None)

            
        
        # Instantiate the LayerVAE model
        self.model = LayerVAE(
            scale=self.scale,
            d_model=config.get("vae:d_model", 256),
            nhead=config.get("transformer:n_head", 8),
            num_layers=config.get("transformer:n_layer", 6),
            latent_dim=config.get("vae:latent_dim", 32), # VAE-specific parameter
            kl_weight=config.get("training:kl_beta",None),
            model_name=config.get("vae:model_name", "v_STGCN5")).to(self.device)
        
        self.dataloader = dataloader
        self.optimizer = Adam(self.model.parameters(), lr=config.get("optim:lr", 1e-4), weight_decay=config.get("optim:wd", 1e-6))
        
        self._setup_scheduler()
        
        self.eval_downstream = config.get("training:eval_downstream", True)
        self.raw_params_path = config.get("data:raw_params_path")
        self.basemodel = config.get("basemodel", "stgcn")
        
        if self.eval_downstream:
            self._setup_downstream_model()

    def _setup_scheduler(self):
        # Simplified scheduler setup
        epochs = self.config.get("training:epochs_train")
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
#Should the eval sequence here use genid?
    def _setup_downstream_model(self):
        """Set up the downstream task model and evaluation data"""

        config_path = 'PredictionModel/config.yaml'
        with open(config_path) as f:
            self.downstream_config = yaml.full_load(f)
        
        from PredictionModel.NetSet import StgnnSet
        self.stgnn_model = StgnnSet(self.downstream_config['data'], self.downstream_config['task'], self.downstream_config['model'], self.basemodel).to(self.device)
        self.stgnn_model.eval()
        for param in self.stgnn_model.parameters():
            param.requires_grad = False
        print("Downstream task model initialized successfully (for LayerVAE_Trainer)")
        self.raw_params_all_for_eval = np.load(self.raw_params_path)
        self.eval_indices = self.genid
        self.lat_indices=range(len(self.raw_params_all_for_eval ))

    def train(self):
        epochs = self.config.get("training:epochs_train")
        print(f"Starting LayerVAE training, total epochs: {epochs}")
        global_step = 0

        # Get all data from the full dataloader
        # (Assuming the data size is small enough to be loaded into memory at once)
        all_data = []
        for batch in self.dataloader:
            all_data.append(batch)
        
        original_indices = torch.cat([b[0] for b in all_data])
        original_flat_weights = torch.cat([b[1] for b in all_data])
        
        # Determine the number of samples to train per epoch
        epoch_sample_size = len(self.dataloader.dataset)
        print(f"Number of samples per epoch: {epoch_sample_size}")
        
        for epoch in range(epochs):
            # Periodically evaluate and record downstream task performance
            #if self.eval_downstream and (epoch%100 == 0 or epoch==0):
            if self.eval_downstream and epoch% 500== 0 and epoch!=0:

                self.eval_parameters(epoch)
            self.model.train()
            total_recon_loss_epoch = 0
            total_kl_loss_epoch = 0
            
            # Shuffle the order for training
            perm = torch.randperm(original_flat_weights.size(0))
            idx = perm[:epoch_sample_size]
            
            # Create a temporary pbar using the subset
            pbar_data = original_flat_weights[idx]
            
            pbar = tqdm(pbar_data.split(self.dataloader.batch_size), desc=f"Epoch {epoch+1}/{epochs}")

            for flat_weights in pbar:
                flat_weights = flat_weights.to(self.device)
                
                self.optimizer.zero_grad()
                
                # VAE-specific loss calculation
                total_loss, recon_loss, kl_div_loss = self.model.training_losses(flat_weights)
                
                total_loss.backward()
                self.optimizer.step()
                
                total_recon_loss_epoch += recon_loss.item()
                total_kl_loss_epoch += kl_div_loss.item()
                
                if self.writer is not None:
                    self.writer.add_scalar('AE/BatchTotalLoss', total_loss.item(), global_step)
                    self.writer.add_scalar('AE/BatchReconLoss', recon_loss.item(), global_step)
                    self.writer.add_scalar('AE/BatchKLDLoss', kl_div_loss.item(), global_step)
                    
                    # Record latent space statistics (for each node dimension)
                    if epoch % 50 == 0 and 'pbar_data' in locals() and pbar_data is not None:
                        with torch.no_grad():
                            _, mu, log_var,_ = self.model.forward(flat_weights) # Rerun forward to get mu, log_var
                            latent_mean = mu.mean().item()
                            latent_std = torch.exp(0.5 * log_var).mean().item() # Average standard deviation
                            self.writer.add_scalar('AE/Latent_Mean', latent_mean, global_step)
                            self.writer.add_scalar('AE/Latent_Std', latent_std, global_step)
                            # Record reconstruction error statistics (RMSE)
                            reconstruction, _, _,_ = self.model.forward(flat_weights)
                            recon_rmse = torch.sqrt(F.mse_loss(reconstruction, flat_weights))
                            self.writer.add_scalar('AE/Recon_RMSE', recon_rmse.item(), global_step)

                pbar.set_postfix({'total_loss': total_loss.item(), 'recon_loss': recon_loss.item(), 'kl_loss': kl_div_loss.item()})
                global_step += 1
            
            avg_recon_loss_epoch = total_recon_loss_epoch / len(pbar)
            avg_kl_loss_epoch = total_kl_loss_epoch / len(pbar)
            avg_total_loss_epoch = avg_recon_loss_epoch + self.model.kl_weight * avg_kl_loss_epoch # Theoretically should be consistent with the average of all batch total_loss
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.writer is not None:
                self.writer.add_scalar('AE/EpochTotalLoss', avg_total_loss_epoch, epoch)
                self.writer.add_scalar('AE/EpochReconLoss', avg_recon_loss_epoch, epoch)
                self.writer.add_scalar('AE/EpochKLDLoss', avg_kl_loss_epoch, epoch)
                self.writer.add_scalar('AE/LearningRate', current_lr, epoch)
            else:
                exit("TensorBoard writer is empty, cannot record loss")
            self.scheduler.step()
            
            


        # Save the final model
        final_model_path = self.results_folder / 'layer_vae_model_final.pt'
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Final LayerVAE model saved: {final_model_path}")
        
        # Ensure all data is written and close the TensorBoard writer
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
                print(f"TensorBoard logs saved successfully")
            except Exception as e:
                print(f"Error closing TensorBoard writer: {e}")
        else:
            print("TensorBoard writer is empty, no need to close")
    def save_latents(self,latent_path=None):

        self.model.eval()
        all_latents = []
        all_indices = []  
        all_reconstructions = []
        print("\n--- Starting downstream task evaluation ---")
        with torch.no_grad():
            for idx in self.lat_indices:
                raw_params = torch.tensor(self.raw_params_all_for_eval[idx], dtype=torch.float32, device=self.device)
                # VAE reconstruction
                # VAE's forward returns reconstruction, mu, log_var
                reconstruction, _, _ ,z= self.model(raw_params.unsqueeze(0))
                all_latents.append(z.cpu().numpy())
                all_reconstructions.append(reconstruction.cpu().numpy())
                all_indices.append([idx])
                
        # Concatenate all batches
        all_latents = np.concatenate(all_latents, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        all_reconstructions = np.concatenate(all_reconstructions, axis=0)
        # Sort by index, in case the dataloader is shuffled
        sorted_indices_map = np.argsort(all_indices)
        all_latents = all_latents[sorted_indices_map]
        all_indices_sorted = all_indices[sorted_indices_map]
        all_reconstructions = all_reconstructions[sorted_indices_map]
        # Save as .npz file
        np.savez(
            latent_path,
            latents=all_latents,
            indices=all_indices_sorted
        )
        reconstruction_path=self.results_folder / 'reconstructions.npz'
        np.savez(
            reconstruction_path,
            latents=all_reconstructions,
            indices=all_indices_sorted,
        )
        print(f"✅ Latent vectors and embeddings saved successfully to: {latent_path}")
        print("="*50 + "\n")
    


    def eval_parameters(self, epoch=None):

        
        self.model.eval()
        rmses_vae = [] # Change ae to vae
        rmses_raw = []
        reconstruction_losses = []
        
        print("\n--- Starting downstream task evaluation ---")
        with torch.no_grad():
            for idx in self.eval_indices:
                raw_params = torch.tensor(self.raw_params_all_for_eval[idx], dtype=torch.float32, device=self.device)
                
                # VAE reconstruction
                # VAE's forward returns reconstruction, mu, log_var
                reconstructed_params, _, _ ,_= self.model(raw_params.unsqueeze(0))
                reconstructed_params = reconstructed_params.squeeze(0)
                
                # Calculate reconstruction loss
                rmse_loss = torch.sqrt(F.mse_loss(reconstructed_params, raw_params)).item()
                reconstruction_losses.append(rmse_loss)
                
                # Evaluation
                outputs_vae, y_label = self.stgnn_model.evalparams(reconstructed_params, self.downstream_config, idx, self.basemodel, if_predict=False, path_config=self.path_config)
                outputs_raw, _ = self.stgnn_model.evalparams(raw_params, self.downstream_config, idx, self.basemodel,if_predict=False, path_config=self.path_config)
                
                rmse_vae = torch.sqrt(F.mse_loss(outputs_vae, y_label)).item()
                rmse_raw = torch.sqrt(F.mse_loss(outputs_raw, y_label)).item()
                
                rmses_vae.append(rmse_vae)
                rmses_raw.append(rmse_raw)
                print(f"  Graph {idx}: VAE RMSE={rmse_vae:.4f}, Raw RMSE={rmse_raw:.4f}, Recon Loss={rmse_loss:.6f}")

        avg_rmse_vae = np.mean(rmses_vae)
        avg_rmse_raw = np.mean(rmses_raw)
        avg_recon_loss = np.mean(rmse_loss)
        
        print(f"--- Evaluation complete: Avg VAE RMSE={avg_rmse_vae:.4f}, Avg Raw RMSE={avg_rmse_raw:.4f}, Avg Recon Loss={avg_recon_loss:.6f} ---")
        
        # Record evaluation metrics to TensorBoard
        if epoch is not None and self.writer is not None:
            try:
                self.writer.add_scalar('AE/VAE_RMSE_Mean', avg_rmse_vae, epoch)
                self.writer.add_scalar('AE/Raw_RMSE_Mean', avg_rmse_raw, epoch)
                self.writer.add_scalar('AE/Reconstruction_Loss_Mean', avg_recon_loss, epoch)
                self.writer.add_scalar('AE/VAE_RMSE_Std', np.std(rmses_vae), epoch)
                self.writer.add_scalar('AE/Raw_RMSE_Std', np.std(rmses_raw), epoch)
                
                # Record RMSE ratio (VAE performance relative to Raw)
                rmse_ratio = avg_rmse_vae / avg_rmse_raw if avg_rmse_raw > 0 else float('inf')
                self.writer.add_scalar('AE/RMSE_Ratio_VAE_vs_Raw', rmse_ratio, epoch)
                
                # # Record for each graph individually
                # for i, (vae_rmse, raw_rmse, recon_loss) in enumerate(zip(rmses_vae, rmses_raw, reconstruction_losses)):
                #     graph_idx = self.eval_indices[i]
                #     self.writer.add_scalar(f'Eval_PerGraph/Graph_{graph_idx}_VAE_RMSE', vae_rmse, epoch)
                #     self.writer.add_scalar(f'Eval_PerGraph/Graph_{graph_idx}_Raw_RMSE', raw_rmse, epoch)
                #     self.writer.add_scalar(f'Eval_PerGraph/Graph_{graph_idx}_Recon_Loss', recon_loss, epoch)
                
                # Ensure evaluation data is written
                self.writer.flush()
                print(f"Evaluation metrics recorded to TensorBoard (epoch {epoch})")
                
            except Exception as e:
                print(f"Error recording evaluation metrics to TensorBoard: {e}")
        elif epoch is not None and self.writer is None:
            print(f"Skipping TensorBoard recording (epoch {epoch}) - writer not available")
        
        self.model.train() # Resume training mode 
    def load_model(self, model_path):
        """Load a pre-trained LayerVAE model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}") 
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"Successfully loaded LayerVAE model: {model_path}")

