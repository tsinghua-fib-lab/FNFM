import argparse
import os
import sys
import time
from typing import Any
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import importlib
from datapreparing import datapreparing
from torchdiff_cfm import TorchDiffCfm
from numpy import random
from TimeTransformer import cfmTransformer,Transformer3,Transformer5
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import matplotlib.pyplot as plt
import math
import gc  # Add garbage collection module
import json
from vae.layer_ae_trainer import LayerAE_Trainer
from vae.layer_tokenizer_transformer import LayerVAE
torch.set_num_threads(20)
###########
# model_name='v_STGCN5'
###########
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class WandbWriter:
    """Minimal WandB wrapper that mimics TensorBoard's add_scalar/close API."""
    def __init__(self, project: str, run_name: str = None, config: dict = None):
        spec = importlib.util.find_spec("wandb")
        if spec is None:
            raise ImportError("uninstall wandb,please pip install wandb")
        self.wandb = importlib.import_module("wandb")
        self.project = project
        self.run_name = run_name
        self.config = config
        self.run = None
        self._ensure_init()

    def _ensure_init(self):
        if getattr(self.wandb, "run", None) is None:
            self.run = self.wandb.init(project=self.project, name=self.run_name, config=self.config)
        else:
            self.run = self.wandb.run

    @staticmethod
    def _to_python_number(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        if isinstance(value, np.ndarray):
            return value.item()
        return value

    def add_scalar(self, tag, value, step=None):
        self._ensure_init()
        self.wandb.log({tag: self._to_python_number(value)}, step=step)

    def add_scalars(self, tag, scalar_dict, step=None):
        # Maintain TensorBoard-like API; flatten into a single log call
        self._ensure_init()
        payload = {f"{tag}/{k}": self._to_python_number(v) for k, v in scalar_dict.items()}
        self.wandb.log(payload, step=step)

    def flush(self):
        pass

    def close(self):
        try:
            self.wandb.finish()
        except Exception:
            pass

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class LatentDataset(Dataset): 
    """Dataset for loading latent vectors and conditional embeddings from .npz file"""
    def __init__(self, train_indices,indices_used,npz_path, time_emb_path,graph_emb_path,normalization_method='zscore'):
        """
        Args:
            npz_path: Path to .npz file
            normalization_method: Normalization method ('zscore' or 'abs_max')
        """
        try:
            print(f"Loading data from {npz_path}...")
            with np.load(npz_path) as data:
                self.latents = torch.from_numpy(data['latents']).to(torch.float32)#[N, Seq, Chan]
                self.latents = self.latents.permute(0, 2, 1).contiguous()
                self.indices = torch.from_numpy(data['indices']).int()
                self.graphEmb=np.load(graph_emb_path).astype(np.float32)
                self.timeEmb=np.load(time_emb_path).astype(np.float32)
                self.time_emb_path=time_emb_path
                self.graph_emb_path=graph_emb_path
                self.npz_path=npz_path
                self.normalization_method=normalization_method
            print("Data loading complete.")
            self.unconditional = False

            # Select normalization method
            self.normalization_method = normalization_method
            self.train_indices = train_indices
            self.indices_used = indices_used
            if normalization_method == 'zscore':
                # Z-score standardization
                self.latent_mean = self.latents[self.train_indices].mean(dim=0, keepdim=True)
                self.latent_std = self.latents[self.train_indices].std(dim=0, keepdim=True)
                # Avoid division by zero
                self.latent_std = torch.where(self.latent_std < 1e-8, torch.tensor(1.0), self.latent_std)
                
                # Standardize to mean=0, std=1
                self.latents = (self.latents - self.latent_mean) / self.latent_std
                
                # Ensure mean and std have the same dimensions as latents (B, D, 1) for later computation
                if len(self.latents.shape) == 2:
                    self.latent_mean = self.latent_mean.unsqueeze(-1)
                    self.latent_std = self.latent_std.unsqueeze(-1)

            # Ensure latent vector shape is (N, latent_dim, 1)
            if len(self.latents.shape) == 2:
                self.latents = self.latents.unsqueeze(-1)
            
            num_samples = len(self.indices_used)

            print(f"  - Total samples: {num_samples}")
            print(f"  - Latent shape: {self.latents.shape}")#[N, Seq, Chan]
            print(f"  - Normalization method: {normalization_method}")

        except Exception as e:
            print(f"Error loading data from npz file: {e}")
            raise

    def denormalize(self, normalized_latents):
        """Denormalize the normalized latent vectors back to original scale"""
        target_device = normalized_latents.device
        
        if self.normalization_method == 'zscore':
            # Z-score denormalization
            latent_mean = self.latent_mean.to(target_device)
            latent_std = self.latent_std.to(target_device)
            #print(latent_mean.shape,latent_std.shape,normalized_latents.shape)
            if normalized_latents.shape[1:-1] == latent_std.shape[1:-1]:
                denormalized = normalized_latents * latent_std + latent_mean
            else:
                exit("latent_std.shape != normalized_latents.shape")
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        return denormalized

    def __len__(self):
        return len(self.indices_used)

    def __getitem__(self, idx):
        # Data is already sorted by index, can directly use idx
        current_index_in_used = self.indices_used[idx]
        
        graph_emb = self.graphEmb[current_index_in_used]
        time_emb = self.timeEmb[current_index_in_used]

        if self.unconditional:
            graph_emb = np.ones_like(graph_emb)
            time_emb = np.ones_like(time_emb)

        return (
            self.indices[current_index_in_used],
            self.latents[current_index_in_used],
            graph_emb,
            time_emb
        )
# ============ Simplified Diffusion Trainer ============
def get_cosine_schedule_with_warmup(optimizer, total_training_steps,warmup_steps=100, num_cycles=0.5, last_epoch=-1):
    """
    Create a learning rate scheduler:
    During warmup phase (warmup_steps), learning rate increases linearly from 0 to initial lr.
    After warmup phase, learning rate decreases following a cosine curve.
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        # Core: math.cos()
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class SimpleMLP(nn.Module):
    """
    A simple MLP denoising network
    """
    def __init__(self,
                channels: int,       # Number of channels in latent vector (latent_dim)
                seq_length: int,     # Sequence length of latent vector
                d_model: int,        # Hidden dimension of MLP
                d_kgEmb: int,        # Graph embedding dimension
                d_timeEmb: int,      # Time embedding dimension
                ):
        super().__init__()
        self.channels = channels
        self.seq_length = seq_length
        
        # Calculate flattened input dimension
        flat_dim = channels * seq_length

        # 1. Encoder for timestep t
        time_mlp_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, time_mlp_dim),
            nn.GELU(),
            nn.Linear(time_mlp_dim, d_model)
        )
        
        # 2. Encoder for conditional embeddings (graphEmb, timeEmb)
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_kgEmb + d_timeEmb, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # 3. Main MLP network
        # Input dimension = flattened x + time encoding + conditional encoding
        input_dim = flat_dim + d_model + d_model 
        
        self.main_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, flat_dim) # Output dimension must match flattened x
        )
        print("SimpleMLP model created for debugging.")
        print(f"   - MLP input dimension: {input_dim}")
        print(f"   - MLP output dimension: {flat_dim}")

    def forward(self,x: torch.Tensor, t: torch.Tensor, graphEmb: torch.Tensor, timeEmb: torch.Tensor) -> torch.Tensor:
        """
        t: timestep [B]
        x: noisy data [B, C, S]
        graphEmb: graph condition [B, d_kgEmb]
        timeEmb: time condition [B, d_timeEmb]
        """
        # 0. Get batch size
        batch_size = x.shape[0]

        # 1. Encode timestep t -> [B, d_model]
        t_emb = self.time_mlp(t.clone().detach().float())
        #        t_emb = self.time_mlp(torch.tensor(t,device=x.device).float())  # Ensure t is float type
        # 2. Encode conditions -> [B, d_model]
        cond_input = torch.cat((graphEmb, timeEmb), dim=1)
        cond_emb = self.cond_mlp(cond_input)

        # 3. Flatten input x -> [B, C * S]
        x_flat = x.view(batch_size, -1)

        # 4. Concatenate all inputs
        mlp_input = torch.cat((x_flat, t_emb, cond_emb), dim=1)
        
        # 5. Pass through main MLP network
        output_flat = self.main_mlp(mlp_input)

        # 6. Reshape output back to original shape [B, C, S]
        output = output_flat.view(batch_size, self.channels, self.seq_length)

        return output
class SimpleLatentTrainer:
    
    def __init__(self, config, cfm_model, train_dataset, test_dataset, **kwargs):
        genid = kwargs.get('genid', None)
        if genid is not None and len(genid) > 0:
            np.random.seed(47)
            self.eval_indices = genid
            print(f"Received genid")

        self.config = config
        self.train_dataset = train_dataset
        self.raw_params_all = np.load(rawdata_path).astype(np.float32)
        # Basic parameters
        self.batch_size = config.batchsize
        self.epochs = config.cfm_epochs
        self.lr = config.cfm_lr

        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.cfm_model = cfm_model.to(self.device)
        # Data loader
        self.dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            drop_last=True
        )
        self.test_dataset = test_dataset
        self.outer_dataset =outer_dataset
        # Optimizer
        self.optimizer = Adam(cfm_model.parameters(), lr=self.lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, total_training_steps=self.epochs, warmup_steps=100)
        # Other parameters
        self.step = 0
        self.logger = kwargs.get('logger')
        self.writer = kwargs.get('writer')
        self.results_folder = Path(kwargs.get('results_folder', './results'))
        self.results_folder.mkdir(exist_ok=True)
        
        # Evaluation related
        self.latent_dim = kwargs.get('latent_dim', 64)
        self.eval_batch_size = kwargs.get('eval_batch_size', 64)
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
    

    
    def _clear_memory(self):
        """Clear memory"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error clearing memory: {e}")
    
    def train(self):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        self.cfm_model.train()
        print("Starting diffusion model training...")
        # Periodic evaluation and saving
        epoch=0

        # Calculate update frequency
        update_freq = 50  
        for epoch in range(self.epochs):
            if epoch!=0 and epoch%1000==0:
                #self.evaluate()
                self.save_checkpoint()
                # check_latent_space_collapse(
                # ae_trainer=ae_trainer,
                # test_dataset=self.test_dataset,
                # raw_params_all=self.raw_params_all,
                # stgnn_config_path=stgnn_config_path,
                # path_config=eval_path_config,
                # basemodel='stgcn',  
                # device=device
                # )
                evaluate_post_diffusion(
                    epoch=epoch,
                    writer=self.writer,
                    eval_index=self.eval_indices,
                    diffusion_model=self.cfm_model,
                    ae_trainer=ae_trainer,
                    outer_dataset=self.outer_dataset,
                    test_dataset=self.test_dataset,
                    raw_params_all=self.raw_params_all,
                    stgnn_config_path=stgnn_config_path,
                    path_config=eval_path_config,
                    basemodel=self.config.basemodel,  # Assume downstream task model is stgcn
                    device=device
                )
                print(f"  Checkpoint saved: checkpoint_{self.step}.pt")
            epoch_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", ncols=120)
            total_epoch_loss = 0.0
            for i, batch in enumerate(epoch_bar):
                # Get training data
                indices, latent_vectors, graphEmb, timeEmb = batch
                # [B, Chan, Seq] to match diffusion model input
                #print(latent_vectors.shape, graphEmb.shape, timeEmb.shape)
                latent_vectors = latent_vectors.to(self.device)
                graphEmb = graphEmb.to(self.device) 
                timeEmb = timeEmb.to(self.device)  
                loss = self.cfm_model(latent_vectors, graphEmb, timeEmb)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                #print(loss.item(), get_lr(self.optimizer))
                self.optimizer.step()
                
                total_epoch_loss += loss.item()
                if self.writer:
                    self.writer.add_scalar('cfm_Loss/step', loss.item(), self.step)

                # Update progress bar, showing current loss and epoch average loss
                running_avg_loss = total_epoch_loss / (i + 1)
                epoch_bar.set_description(f'Epoch {epoch+1}/{self.epochs} [Avg Loss: {running_avg_loss:.4f}]')
                
                self.step += 1
            
            # Record average loss for the entire epoch
            avg_epoch_loss = total_epoch_loss / len(self.dataloader)
            if self.writer:
                self.writer.add_scalar('cfm_Loss/epoch_avg', avg_epoch_loss, epoch)

            epoch_bar.close()
            self.lr_scheduler.step()
        
        # Save final model
        self.save_final_model()
        print(f"\nDiffusion training complete! Total steps: {self.epochs}")
            
    
    def save_checkpoint(self):
        try:
            checkpoint = {
                'step': self.step,
                'model_state_dict': self.cfm_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(checkpoint, self.results_folder / f'checkpoint_{self.step}.pt')
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def save_final_model(self):
        try:
            model_path = self.results_folder / 'final_diffusion_model.pt'
            torch.save(self.cfm_model.state_dict(), model_path)
            print(f"Diffusion model saved: {model_path.name}")
            if self.logger:
                self.logger.info(f'Final diffusion model saved: {model_path}')
                self.logger.info(f'Training completed with simplified workflow')
        except Exception as e:
            print(f"Error saving final model: {e}")
    def load_model(self):
        model_path = self.results_folder / 'final_diffusion_model.pt'
        try:
            self.cfm_model.load_state_dict(torch.load(model_path))
            print(f"Diffusion model loaded: {model_path.name}")
        except Exception as e:
            print(f"Error loading diffusion model: {e}")
          
def plot_two_distributions(arr1, arr2, label1="real", label2="generated", title="variance distribution", kde_bandwidth='scott'):
    """
    Plot distribution curves of two float arrays (using histogram and KDE).

    Parameters:
    arr1 (np.array or list): First float array.
    arr2 (np.array or list): Second float array.
    label1 (str): Legend label for first distribution.
    label2 (str): Legend label for second distribution.
    title (str): Chart title.
    kde_bandwidth (str or float): Bandwidth selection for KDE. Can be 'scott', 'silverman',
                                  or a float. Higher values make the curve smoother.
                                  See seaborn.kdeplot documentation for details.
    """

    # Convert inputs to numpy arrays for compatibility
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    fig, ax = plt.subplots(figsize=(8, 6))

    print(f"{label1} data statistics: mean={np.mean(arr1):.2f}, std={np.std(arr1):.2f}, min={np.min(arr1):.2f}, max={np.max(arr1):.2f}")
    print(f"{label2} data statistics: mean={np.mean(arr2):.2f}, std={np.std(arr2):.2f}, min={np.min(arr2):.2f}, max={np.max(arr2):.2f}")

    # Use histplot to draw histogram (density)
    sns.histplot(arr1, stat="density", label=label1, color="skyblue", alpha=0.6, ax=ax, bins=50)
    sns.histplot(arr2, stat="density", label=label2, color="salmon", alpha=0.6, ax=ax, bins=50)

    # Overlay KDE curves separately
    sns.kdeplot(arr1, color="skyblue", linewidth=2, ax=ax)
    sns.kdeplot(arr2, color="salmon", linewidth=2, ax=ax)   
    plt.title(title, fontsize=16)
    plt.xlabel("value", fontsize=14)
    plt.ylabel("density", fontsize=14)
    plt.legend(fontsize=12) # Show legend
    plt.grid(True, linestyle='--', alpha=0.7) # Add grid
    plt.tight_layout() # Automatically adjust chart layout to prevent label overlap
    plt.savefig("variance_distribution.png")
    print(f"Variance distribution plot saved to: variance_distribution.png")

def save_trajectory(
    epoch,
    writer,
    eval_index,
    diffusion_model,
    ae_trainer,
    train_dataset,
    test_dataset,
    outer_dataset,
    raw_params_all,
    stgnn_config_path,
    path_config,
    basemodel,
    device,
    args,

):
    gen_list=[]
    indices_list=[]
    results_folder = Path(f'./ModelSave/exp{args.expIndex}')
    final_model_path = results_folder / 'layer_vae_model_final.pt'
    base_data_dir = "../Pretrain"
    diffusion_model.eval()
    ae_trainer.model.eval()
    
    from PredictionModel.NetSet import StgnnSet
    with open(stgnn_config_path) as f:
        downstream_config = yaml.full_load(f)
    data_args, task_args, model_args = downstream_config['data'], downstream_config['task'], downstream_config['model']
    stgnn_model = StgnnSet(data_args, task_args, model_args, basemodel).to(device)
    stgnn_model.eval()
    for param in stgnn_model.parameters():
        param.requires_grad = False
    set_seed(42)
    select_indices=np.random.choice(len(train_dataset.train_indices), size=50)
    val_dataset = Subset(train_dataset, select_indices)
    val_dataloader = DataLoader[Any](val_dataset, batch_size=32, shuffle=False)
    raw_param=raw_params_all[0]
    _,_,_,metadata =ae_trainer.model.encode(torch.from_numpy(raw_param.flatten()).float().unsqueeze(0).to(device))
    
    from torch.utils.data import ConcatDataset
    print(f"Combining test_dataset ({len(test_dataset)}) and outer_dataset ({len(outer_dataset)}).")
    combined_eval_dataset = ConcatDataset([test_dataset, outer_dataset])
    print(f"Total evaluation dataset size: {len(combined_eval_dataset)}")
    test_dataloader = DataLoader(combined_eval_dataset, batch_size=32, shuffle=False)
    indices_list=[]
    trajectory={}
    
    # 4.1 获取AE模型的元数据
    raw_param=raw_params_all[0]
    _,_,_,metadata =ae_trainer.model.encode(torch.from_numpy(raw_param.flatten()).float().unsqueeze(0).to(device))
    
    print("Generating parameters for PCA...")
    all_generated_params = []
    all_generated_indices = []
    all_generated_coefficients=[]
    pca_gen_dataloader = DataLoader(combined_eval_dataset, batch_size=10, shuffle=False, drop_last=True)
    with torch.no_grad():
        for batch_data in tqdm(pca_gen_dataloader, desc="Generating params for PCA"):
            indices, _, kg_embs, time_embs = batch_data
            indices, kg_embs, time_embs = indices.to(device), kg_embs.to(device).to(float), time_embs.to(device).to(float)
            coefficients=kg_embs.cpu().numpy()
            all_generated_coefficients.append(coefficients)
            # a. Generate latents from diffusion model
            diff_latents_normalized = diffusion_model.sample(
                batch_size=len(indices), graphEmb=kg_embs, timeEmb=time_embs
            )
            # b. Denormalize latents
            diff_latents_ae_input = test_dataset.denormalize(diff_latents_normalized)
            # c. Permute for AE decoder
            diff_latents_ae_input = diff_latents_ae_input.permute(0, 2, 1).contiguous()
            # d. Decode to get high-dimensional parameters
            diff_params = ae_trainer.model.decode(diff_latents_ae_input, metadata=metadata)
            all_generated_params.append(diff_params.cpu().numpy())
            all_generated_indices.append(indices.cpu().numpy())
    all_generated_params = np.concatenate(all_generated_params, axis=0)
    all_generated_coefficients = np.concatenate(all_generated_coefficients, axis=0)
    print(f"Generated {all_generated_params.shape[0]} parameters")

    # 2. Save the generated weights
    results_folder = Path(f'./ModelSave/exp{args.expIndex}')
    generated_weights_path = results_folder / (args.targetDataset + args.model_name+"generated_weights.npz")
    np.savez(generated_weights_path, indices=all_generated_indices,coefficients= all_generated_coefficients,weights=all_generated_params)
    print(f"Saved generated weights to {generated_weights_path}")
    diffusion_model.train()

def evaluate_post_diffusion(
    epoch,
    writer,
    eval_index,
    diffusion_model,
    ae_trainer,
    outer_dataset,
    test_dataset,
    raw_params_all,
    stgnn_config_path,
    path_config,
    basemodel,
    device
):
    """
    After diffusion model training, evaluate using AE decoder.
    1. Generate latent vectors
    2. Reconstruct high-dimensional parameters using AE decoder
    3. Calculate reconstruction RMSE
    4. Evaluate downstream task RMSE
    """
    gen_list=[]
    indices_list=[]
    print("\n" + "=" + "="*48 + "=")
    print("Stage 3: Evaluate generation quality (AE decoding + downstream task)")
    print("="*50)
    results_folder = Path(f'./ModelSave/exp{args.expIndex}')
    final_model_path = results_folder / 'layer_vae_model_final.pt'
    base_data_dir = "./Pretrain"
    diffusion_model.eval()
    ae_trainer.model.eval()
    mse_entries = []

    def record_mse(section, sample_idx, raw_rmse, generated_rmse, environment_params):
        if sample_idx is None:
            return
        if isinstance(environment_params, torch.Tensor):
            env_params_serializable = environment_params.detach().cpu().tolist()
        elif isinstance(environment_params, np.ndarray):
            env_params_serializable = environment_params.tolist()
        else:
            env_params_serializable = environment_params
        mse_entries.append({
            'section': section,
            'index': int(sample_idx),
            'environment_params': env_params_serializable,
            'raw_rmse': float(raw_rmse),
            'ae_cfm_rmse': float(generated_rmse)
        })

    # 3. Load downstream task model
    print(f"Loading downstream task model configuration: {stgnn_config_path}")
    from PredictionModel.NetSet import StgnnSet
    with open(stgnn_config_path) as f:
        downstream_config = yaml.full_load(f)
    data_args, task_args, model_args = downstream_config['data'], downstream_config['task'], downstream_config['model']
    stgnn_model = StgnnSet(data_args, task_args, model_args, basemodel).to(device)
    stgnn_model.eval()
    for param in stgnn_model.parameters():
        param.requires_grad = False
    set_seed(42)
    outer_dataloader = DataLoader(outer_dataset, batch_size=32, shuffle=False)
    
    all_recon_rmses = []
    all_downstream_rmses_diff = []
    all_downstream_rmses_raw = []
    all_downstream_rmses_ae = []  # Diagnostic: Store RMSE reconstructed from real latent vectors
    # 4.1 Get AE model metadata
    raw_param=raw_params_all[0]
    _,_,_,metadata =ae_trainer.model.encode(torch.from_numpy(raw_param.flatten()).float().unsqueeze(0).to(device))
    
    with torch.no_grad():
        batch_count = 0
        for batch_data in tqdm(outer_dataloader, desc="Evaluating"):
            
            indices, gt_latents_normalized, kg_embs, time_embs = batch_data
            # This ensures testing only on test set, but will this cause issues? Testing one sample at a time
            if torch.is_tensor(kg_embs):
                environment_params_all = kg_embs.detach().cpu().numpy()
            else:
                environment_params_all = np.asarray(kg_embs)
            indices, gt_latents_normalized, kg_embs, time_embs = indices.to(device), gt_latents_normalized.to(device).to(float), kg_embs.to(device).to(float), time_embs.to(device).to(float)

            # a. Use diffusion model to generate normalized latent vectors (output: B, Chan, Seq)
            diff_latents_normalized = diffusion_model.sample(
                batch_size=len(indices), graphEmb=kg_embs, timeEmb=time_embs
            )
            # b. Denormalize latent vectors (z-score or abs_max) to feed into AE
            diff_latents_ae_input = test_dataset.denormalize(diff_latents_normalized)
            # [B, Chan, Seq] -> [B, Seq, Chan] for AE decoder
            diff_latents_ae_input = diff_latents_ae_input.permute(0, 2, 1).contiguous()
            # c. Use AE decoder to reconstruct high-dimensional parameters (scaled)
            diff_params = ae_trainer.model.decode(diff_latents_ae_input,metadata=metadata)
            # d. Get corresponding original high-dimensional parameters
            raw_params_unscaled_np = raw_params_all[indices.cpu().numpy()]
            raw_params_unscaled = torch.from_numpy(raw_params_unscaled_np).float().to(device).view(len(indices), -1)
            gen_list.append(diff_params)
            indices_list.append(indices)
            # e. Calculate reconstruction RMSE
            recon_rmse = F.mse_loss(diff_params, raw_params_unscaled).item()
            all_recon_rmses.append(recon_rmse)

            ae_latents = test_dataset.denormalize(gt_latents_normalized).to(torch.float32)
            ae_latents = ae_latents.permute(0, 2, 1).contiguous()
            # 2. Use AE decoder to reconstruct parameters from real latent vectors
            ae_params = ae_trainer.model.decode(ae_latents,metadata)
            
            # f. Evaluate downstream task performance for each sample
            for i in range(len(indices)):
                idx = indices[i].item()
                environment_params = environment_params_all[i] if environment_params_all is not None else None
                outputs_gen, y_label = stgnn_model.evalparams(diff_params[i], downstream_config, idx, path_config=path_config, basemodel=basemodel, if_adjust=False, name='DIFF')
                downstream_rmse_gen = F.mse_loss(outputs_gen, y_label).item()
                all_downstream_rmses_diff.append(downstream_rmse_gen)

                outputs_gt, _ = stgnn_model.evalparams(raw_params_unscaled[i], downstream_config, idx, basemodel=basemodel, path_config=path_config, if_adjust=False, name='REAL')
                downstream_rmse_gt = F.mse_loss(outputs_gt, y_label).item()
                all_downstream_rmses_raw.append(downstream_rmse_gt)

                outputs_recon, _ = stgnn_model.evalparams(ae_params[i], downstream_config, idx, basemodel, path_config=path_config,if_adjust=False,name='AE')
                downstream_rmse_recon = F.mse_loss(outputs_recon, y_label).item()
                all_downstream_rmses_ae.append(downstream_rmse_recon)
                record_mse('outer', idx, downstream_rmse_gt, downstream_rmse_gen, environment_params)

            # --- Diagnostic end ---
            batch_count += 1
    #gen_list=torch.cat(gen_list,axis=0).detach().cpu().numpy()
    #indices_list=torch.cat(indices_list,axis=0).detach().cpu().numpy()
    #gen_data_path=f"outer_data_{epoch}.npz"
    #np.savez(results_folder/gen_data_path,latents=gen_list,indices=indices_list)
    # 5. Report results
    outer_recon_rmse = np.sqrt(np.mean(all_recon_rmses)) if all_recon_rmses else 0
    outer_downstream_rmse_gen = np.sqrt(np.mean(all_downstream_rmses_diff)) if all_downstream_rmses_diff else 0
    outer_downstream_rmse_gt = np.sqrt(np.mean(all_downstream_rmses_raw)) if all_downstream_rmses_raw else 0
    outer_downstream_rmse_recon_from_gt = np.sqrt(np.mean(all_downstream_rmses_ae)) if all_downstream_rmses_ae else 0
    writer.add_scalar('Evaluation/Outer/Avg_Param_Recon_RMSE', outer_recon_rmse, epoch)
    writer.add_scalar('Evaluation/Outer/Avg_Downstream_RMSE_GroundTruth', outer_downstream_rmse_gt, epoch)
    writer.add_scalar('Evaluation/Outer/Avg_Downstream_RMSE_AE_Ceiling', outer_downstream_rmse_recon_from_gt, epoch)
    writer.add_scalar('Evaluation/Outer/Avg_Downstream_RMSE_Generated', outer_downstream_rmse_gen, epoch)
    
    # 4. Prepare data loader, can we filter out validation set directly from data loader?
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    indices_list=[]
    gen_list=[]
    all_recon_rmses = []
    all_downstream_rmses_diff = []
    all_downstream_rmses_raw = []
    all_downstream_rmses_ae = []  # Diagnostic: Store RMSE reconstructed from real latent vectors
    # 4.1 Get AE model metadata
    raw_param=raw_params_all[0]
    _,_,_,metadata =ae_trainer.model.encode(torch.from_numpy(raw_param.flatten()).float().unsqueeze(0).to(device))
    with torch.no_grad():
        batch_count = 0
        for batch_data in tqdm(test_dataloader, desc="Evaluating"):
            
            indices, gt_latents_normalized, kg_embs, time_embs = batch_data
            # This ensures testing only on test set, but will this cause issues? Testing one sample at a time
            if torch.is_tensor(kg_embs):
                environment_params_all = kg_embs.detach().cpu().numpy()
            else:
                environment_params_all = np.asarray(kg_embs)
            indices, gt_latents_normalized, kg_embs, time_embs = indices.to(device), gt_latents_normalized.to(device).to(float), kg_embs.to(device).to(float), time_embs.to(device).to(float)

            # a. Use diffusion model to generate normalized latent vectors (output: B, Chan, Seq)
            diff_latents_normalized = diffusion_model.sample(
                batch_size=len(indices), graphEmb=kg_embs, timeEmb=time_embs
            )
            # b. Denormalize latent vectors (z-score or abs_max) to feed into AE
            diff_latents_ae_input = test_dataset.denormalize(diff_latents_normalized)
            # [B, Chan, Seq] -> [B, Seq, Chan] for AE decoder
            diff_latents_ae_input = diff_latents_ae_input.permute(0, 2, 1).contiguous()
            # c. Use AE decoder to reconstruct high-dimensional parameters (scaled)
            diff_params = ae_trainer.model.decode(diff_latents_ae_input,metadata=metadata)
            # d. Get corresponding original high-dimensional parameters
            raw_params_unscaled_np = raw_params_all[indices.cpu().numpy()]
            raw_params_unscaled = torch.from_numpy(raw_params_unscaled_np).float().to(device).view(len(indices), -1)
            gen_list.append(diff_params)
            indices_list.append(indices)
            # e. Calculate reconstruction RMSE
            recon_rmse = F.mse_loss(diff_params, raw_params_unscaled).item()
            all_recon_rmses.append(recon_rmse)

            ae_latents = test_dataset.denormalize(gt_latents_normalized).to(torch.float32)
            ae_latents = ae_latents.permute(0, 2, 1).contiguous()
            # 2. Use AE decoder to reconstruct parameters from real latent vectors
            ae_params = ae_trainer.model.decode(ae_latents,metadata)
            
            # f. Evaluate downstream task performance for each sample
            for i in range(len(indices)):
                idx = indices[i].item()
                environment_params = environment_params_all[i] if environment_params_all is not None else None
                outputs_gen, y_label = stgnn_model.evalparams(diff_params[i], downstream_config, idx, path_config=path_config, basemodel=basemodel, if_adjust=False, name='DIFF')
                downstream_rmse_gen = F.mse_loss(outputs_gen, y_label).item()
                all_downstream_rmses_diff.append(downstream_rmse_gen)

                outputs_gt, _ = stgnn_model.evalparams(raw_params_unscaled[i], downstream_config, idx, basemodel=basemodel, path_config=path_config, if_adjust=False, name='REAL')
                downstream_rmse_gt = F.mse_loss(outputs_gt, y_label).item()
                all_downstream_rmses_raw.append(downstream_rmse_gt)

                outputs_recon, _ = stgnn_model.evalparams(ae_params[i], downstream_config, idx, basemodel, path_config=path_config,if_adjust=False,name='AE')
                downstream_rmse_recon = F.mse_loss(outputs_recon, y_label).item()
                all_downstream_rmses_ae.append(downstream_rmse_recon)
                record_mse('test', idx, downstream_rmse_gt, downstream_rmse_gen, environment_params)

            # --- Diagnostic end ---
            batch_count += 1
    gen_list=torch.cat(gen_list,axis=0).detach().cpu().numpy()
    indices_list=torch.cat(indices_list,axis=0).detach().cpu().numpy()
    gen_data_path=f"gen_data_{epoch}.npz"
    np.savez(results_folder/gen_data_path,latents=gen_list,indices=indices_list)
    # 5. Report results
    avg_recon_rmse = np.sqrt(np.mean(all_recon_rmses)) if all_recon_rmses else 0
    avg_downstream_rmse_gen = np.sqrt(np.mean(all_downstream_rmses_diff)) if all_downstream_rmses_diff else 0
    avg_downstream_rmse_gt = np.sqrt(np.mean(all_downstream_rmses_raw)) if all_downstream_rmses_raw else 0
    avg_downstream_rmse_recon_from_gt = np.sqrt(np.mean(all_downstream_rmses_ae)) if all_downstream_rmses_ae else 0
    writer.add_scalar('Evaluation/Summary/Avg_Param_Recon_RMSE', avg_recon_rmse, epoch)
    writer.add_scalar('Evaluation/Summary/Avg_Downstream_RMSE_GroundTruth', avg_downstream_rmse_gt, epoch)
    writer.add_scalar('Evaluation/Summary/Avg_Downstream_RMSE_AE_Ceiling', avg_downstream_rmse_recon_from_gt, epoch)
    writer.add_scalar('Evaluation/Summary/Avg_Downstream_RMSE_Generated', avg_downstream_rmse_gen, epoch)
    print("\n" + "="*50)
    print("Final evaluation results")
    print(f"  - AE+CFM parameter RMSE: {avg_recon_rmse:.4f}")
    print(f"  - Real parameter RMSE: {avg_downstream_rmse_gt:.4f}")
    print(f"  - AE reconstruction RMSE: {avg_downstream_rmse_recon_from_gt:.4f}")
    print(f"  - AE+CFM RMSE: {avg_downstream_rmse_gen:.4f} ")
    print(f"  - Outer recon RMSE: {outer_recon_rmse:.4f}")
    print(f"  - Outer GroundTruth Downstream RMSE: {outer_downstream_rmse_gt:.4f}")
    print(f"  - Outer AEDownstream RMSE: {outer_downstream_rmse_recon_from_gt:.4f}")
    print(f"  - Outer Generated Downstream RMSE: {outer_downstream_rmse_gen:.4f}")
    print("="*50)

    if mse_entries:
        mse_results_path = results_folder / 'mse_results.json'
        with open(mse_results_path, 'w') as f:
            json.dump(mse_entries, f, indent=2)
        print(f"Saved per-index RMSE to {mse_results_path}")

    diffusion_model.train()

if __name__ == '__main__':

        
    parser = argparse.ArgumentParser(description="Latent space diffusion model training script (training only, no AE)")
    # Core parameters
    parser.add_argument("--expIndex", type=int, default=889, help="Experiment index for naming output folder")
    parser.add_argument("--cfm_epochs", type=int, default=300, help="Total training steps for diffusion model")
    parser.add_argument("--ae_epochs", type=int, default=500, help="Total training steps for autoencoder")
    parser.add_argument("--denoise", type=str, default='cfmTransformer', help="Denoising model")
    parser.add_argument("--model_name", type=str, default='v_STGCN5', help="Name of the model architecture")
    parser.add_argument("--modeldim", type=int, default=128, help="Model dimension")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size")
    parser.add_argument("--log_backend", type=str, default="wandb", choices=["tensorboard", "wandb", "none"], help="tensorboard/wandb/none")
    parser.add_argument("--wandb_project", type=str, default="crosscity", help="wandb title")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name, auto-generated if not provided")

    # Diffusion process parameters
    parser.add_argument("--cfmstep", type=int, default=1000, help="Diffusion steps")
    # Data processing parameters
    parser.add_argument("--normalization", type=str, default='zscore', choices=['zscore', 'abs_max'],
                        help="Normalization method: zscore (Z-score standardization, recommended) or abs_max (absolute max normalization)")
    # Evaluation related parameters
    parser.add_argument("--evaluate_post_train", action='store_false', help="Evaluate using AE decoder after training")
    parser.add_argument("--targetDataset", type=str, default="Epidemic", help="Dataset for datapreparing")
    parser.add_argument("--mode", type=str, default='AE', help="Training mode: AE, CFM, or AE_CFM")
    parser.add_argument("--ae_arch", type=str, default='transformer', help="AE architecture to use: 'mlp' or 'transformer' or 'layer_transformer'")
    parser.add_argument("--basemodel", type=str, default='v_GWN')
    parser.add_argument("--cfm_lr", type=float, default=1E-4, help="Learning rate")
    parser.add_argument("--num_environments", type=int, default=3, help="Total number of noisy environment datasets to load")
    parser.add_argument("--train_objective", type=str, default='cfm', choices=['cfm','diff'], help="Training objective: cfm or diff")
    parser.add_argument("--unconditional", action="store_true", help="set condition to 1")

    args = parser.parse_args()        

    ae_model_path=f"./ModelSave/exp{args.expIndex}/layer_vae_model_final.pt"
    latent_data_path=f"./ModelSave/exp{args.expIndex}/latents_data.npz"
    stgnn_config_path=f"PredictionModel/config.yaml"
    "hyper params"
    
    batchsize=args.batchsize
    outputpath = './Output/exp{}'.format(args.expIndex)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    modelsavepath = './ModelSave/exp{}'.format(args.expIndex)
    if not os.path.exists(modelsavepath):
        os.makedirs(modelsavepath)
    '''
    data preparing
    '''
    if args.basemodel == 'v_STGCN5':
        flag='stgcn'
    elif args.basemodel == 'v_GWN':
        flag='gwn'
    else:
        print("model_name is not supported")
        exit()
    if args.targetDataset in ['twitter',"hill","euroad","collab","fhn","fhn2"]:
        base_data_dir = "../Pretrain"
        graph_set_path = os.path.join(base_data_dir, f'PrepareParams/TaskParams_{flag}_{args.targetDataset}_final.npy')
        task_params_path = os.path.join(base_data_dir, f'PrepareParams/TaskParams_{flag}_{args.targetDataset}_final.npy')
        rawdata_path = os.path.join(base_data_dir, f'PrepareParams/ModelParams_{flag}_{args.targetDataset}_final.npy')
        kg_emb_path = os.path.join(base_data_dir, f'PrepareParams/TaskParams_{flag}_{args.targetDataset}_final.npy')
        time_emb_path = os.path.join(base_data_dir, f'PrepareParams/TaskParams_{flag}_{args.targetDataset}_final.npy')
        
        # Prepare path configuration for evalparams
        eval_path_config = {
            'pathname': args.targetDataset,
            'graph_set_path': os.path.join(base_data_dir, f'PrepareParams/TaskParams_{flag}_{args.targetDataset}_final.npy'),
            'adjacency_matrix_path': os.path.join(base_data_dir, f'graph_generator/output/{args.targetDataset}/{args.targetDataset}_matrix.npy'),
            'trajectories_dir': os.path.join(base_data_dir, f'graph_generator/output/{args.targetDataset}/')
        }
    elif args.targetDataset=='HillDisturbe':
        base_data_dir="./Pretrain"
        graph_set_path = os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_hill_disturbe.npy")
        task_params_path=os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_hill_disturbe.npy")
        rawdata_path = os.path.join(base_data_dir, f'PrepareParams/ModelParams_{flag}_hill_disturbe.npy')
        kg_emb_path = os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_hill_disturbe.npy")
        time_emb_path = os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_hill_disturbe.npy")
        eval_path_config = {
            'pathname': 'hill',
            'graph_set_path': os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_hill_disturbe.npy"),
            'adjacency_matrix_path': os.path.join(base_data_dir, "graph_generator/output/hill/hill_matrix.npy"),
            'trajectories_dir': os.path.join(base_data_dir, "graph_generator/output/hill/")
        }    
        
    elif args.targetDataset=='fhn2':
        base_data_dir="../Pretrain"
        graph_set_path = os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_fhn2_final.npy")
        task_params_path=os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_fhn2_final.npy")
        rawdata_path = os.path.join(base_data_dir, f'PrepareParams/ModelParams_{flag}_fhn2_final.npy')
        kg_emb_path = os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_fhn2_final.npy")
        time_emb_path = os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_fhn2_final.npy")

        eval_path_config = {
            'pathname': 'fhn2',
            'graph_set_path': os.path.join(base_data_dir, f"PrepareParams/TaskParams_{flag}_fhn2_final.npy"),
            'adjacency_matrix_path': os.path.join(base_data_dir, "graph_generator/output/fhn2/fhn2_matrix.npy"),
            'trajectories_dir': os.path.join(base_data_dir, "graph_generator/output/fhn2/")
        }
    else:
        # If there are other datasets, add them here
        print(f"Unsupported dataset: {args.targetDataset}")
        sys.exit(1)
    training_seq, scale,trainid,genid,outerid,kgtrainEmb, kggenEmb, timetrainEmb,timegenEmb, genTarget =datapreparing (
        targetDataset=args.targetDataset, 
        basemodel=args.basemodel,
        rawdata_path=rawdata_path,
        kg_emb_path=kg_emb_path,
        time_emb_path=time_emb_path,
        graph_set_path=graph_set_path,
        task_params_path=task_params_path
        )  
    
   
    print('training_seq.shape', training_seq.shape)
    print('kgEmbedding.shape', kgtrainEmb.shape)  
    print('timeEmbedding.shape', timetrainEmb.shape)
    print('gen_seq.shape', genTarget.shape) 
    # Set up logging
    writer = None
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    if args.log_backend == "tensorboard":
        os.makedirs('TensorBoardLogs', exist_ok=True)
        writer = SummaryWriter(log_dir=f'TensorBoardLogs/exp{args.expIndex}/{current_time}')
        print(f"TensorBoard log directory: TensorBoardLogs/exp{args.expIndex}")
    elif args.log_backend == "wandb":
        run_name = args.wandb_run_name or f"exp{args.expIndex}_{current_time}"
        try:
            writer = WandbWriter(project=args.wandb_project, run_name=run_name, config=vars(args))
            run_url = getattr(getattr(writer, "run", None), "url", None)
            if run_url:
                print(f"wandb log: {run_url}")
            else:
                print(f"wandb log started, project {args.wandb_project}, run {run_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize wandb, please check installation and key: {e}")
    else:
        raise ValueError("unkonwn log_backend, please choose tensorboard or wandb")
    
    "AE_config"
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE_config = {
    "seed": 42,
    "device": device,
    "scale": scale,
    "writer": writer,
    "basemodel": args.basemodel,
    #model config
    "vae:d_model": 128,
    "vae:latent_dim": 32,
    "vae:model_name": args.model_name,
    "transformer:n_head": 8,
    "transformer:n_layer": 2,
    # Data Paths to be passed to the trainer
    "data:raw_params_path": rawdata_path,
    "data:parameter_list_path": graph_set_path,
    "data:kg_emb_path": kg_emb_path,
    "data:time_emb_path": time_emb_path,
    "data:task_params_path": task_params_path,
    
    # Training Set Configuration
    "trainset:batchsize": 32,
    "trainset:step_for_one_epoch": len(training_seq),

    # Optimizer Configuration
    "optim:lr": 1e-4,
    "optim:wd": 3e-9,
    "optim:scheduler": "OneCycleLR",

    # Training and Task Configuration
    "training:kl_beta": 1e-6,  # Added KL divergence regularization
    "training:epochs_train": args.ae_epochs,
    "training:eval_downstream": True,    
    }
    # Create directories
    if args.ae_arch == 'layer_transformer':
        from vae.layer_ae_trainer import LayerAE_Trainer as AE_Trainer
        from vae.layer_ae_trainer import SimpleParameterDataset
        print("Using Layer-wise Transformer Autoencoder.")
        # This AE directly processes flat weights using SimpleParameterDataset
        ae_dataset = SimpleParameterDataset(training_seq, kgtrainEmb, timetrainEmb)
        ae_dataloader = DataLoader(ae_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=0)
    else:
        raise ValueError(f"Unsupported --ae-arch: {args.ae_arch}. Choose 'mlp' or 'transformer' or 'layer_transformer'.")
    ae_trainer = AE_Trainer(
        config=AE_config, 
        dataloader=ae_dataloader,
        exp_index=args.expIndex, 
        results_folder=modelsavepath,
        path_config=eval_path_config,
        genid=genid
    )
    if args.mode =='AE' or args.mode == 'AE_CFM':
        ae_trainer.train()
        if args.targetDataset== 'Graph':
            graphEmb=np.load(kg_emb_path)
            graphEmb=graphEmb.astype(float)
            timeEmb=np.load(time_emb_path).astype(float)
        else:
            emb_path=time_emb_path
            graphEmb=np.load(emb_path).astype(np.float32)
            timeEmb=np.load(time_emb_path).astype(np.float32)
        # 4. Automatic evaluation after training
        print("--- AE Training finished, starting evaluation ---")
        ae_trainer.eval_parameters()
        ae_trainer.save_latents(latent_data_path)
        print("--- AE Evaluation finished ---")
    elif args.mode not in ['AE','AE_CFM', 'CFM', 'TEST_CFM']:
        raise ValueError(f"Unsupported mode: {args.mode}. Choose 'AE', 'CFM', or 'AE_CFM'.")
    if args.mode == 'CFM':
        graphEmb=np.load(kg_emb_path).astype(float)
        timeEmb=np.load(time_emb_path).astype(float)
        ae_trainer.load_model(ae_model_path)        
    elif args.mode == 'AE_CFM':
        pass
    elif args.mode == 'TEST_CFM':
        graphEmb=np.load(kg_emb_path).astype(float)
        timeEmb=np.load(time_emb_path).astype(float)
        ae_trainer.load_model(ae_model_path) 
    else: 
        print("AE training complete")
        sys.exit(0)

    # CFM model training
    normalization_method = getattr(args, 'normalization', 'zscore')

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    outputpath = Path(f'./Output/exp{args.expIndex}')
    outputpath.mkdir(parents=True, exist_ok=True)
    modelsavepath = Path(f'./ModelSave/exp{args.expIndex}')
    modelsavepath.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data and create datasets
    print("\n" + "="*48 + "=")
    print("Stage 1: Load data from .npz file")
    print("="*50)
    # Note: Normalize training and test sets separately
    train_dataset = LatentDataset(trainid,trainid,latent_data_path,time_emb_path,kg_emb_path,normalization_method=normalization_method)
    test_dataset = LatentDataset(trainid,genid,latent_data_path,time_emb_path,kg_emb_path,normalization_method=normalization_method)
    outer_dataset = LatentDataset(trainid,outerid,latent_data_path,time_emb_path,kg_emb_path,normalization_method=normalization_method)

    if args.unconditional:
        print("Running in unconditional mode, all conditions will be set to 1.")
        train_dataset.unconditional = True
        test_dataset.unconditional = True
        outer_dataset.unconditional = True
    # Infer latent_dim and seq_len from dataset
    # Latent vector shape: (N, channel, seq_len)
    _, latent_dim,seq_len = train_dataset.latents.shape
    print(f"Successfully loaded latent vector data: {train_dataset.latents.shape}")
    print(f"Using latent space dimension (channels): {latent_dim}, sequence length: {seq_len}")
    # Step 2: Train diffusion model in latent space
    print("Stage 2: Latent space diffusion model training")
    print("="*50)
    # Create denoising network
    print(f"Using denoising network: {args.denoise}")
    if args.denoise == 'cfmTransformer':
        d_model = args.modeldim
        q, v, h = 2, 2, 2
        N = 4
        d_kgEmb = train_dataset.graphEmb.shape[1]
        d_timeEmb = train_dataset.timeEmb.shape[1]
        dropout = 0.1
        d_input = latent_dim # Feature dimension
        d_output = latent_dim
        layernum = 3
        # generateModel=SimpleMLP(d_input,seq_len,d_model,d_kgEmb,d_timeEmb)
        generateModel = cfmTransformer(
            d_input, d_model, d_output, d_kgEmb, d_timeEmb,h, N,
            dropout=dropout, pe='regular'
        ).to(device)
    # Support for other Transformer types can be added here
    elif args.denoise == 'Trans3':
        d_model = args.modeldim
        N = 4
        # Get conditional embedding dimension from dataset
        d_kgEmb = train_dataset.graphEmb.shape[1]
        d_timeEmb = train_dataset.timeEmb.shape[1]
        dropout = 0.1
        d_input = latent_dim # Feature dimension
        d_output = latent_dim
        layernum = 1
        generateModel = Transformer3(
            d_input, d_model, d_output, d_kgEmb, d_timeEmb, N,
            layernum=layernum, dropout=dropout, pe='original'
        ).to(device)
    elif args.denoise == 'Trans5':
        d_model = args.modeldim
        q, v, h = 10, 10, 10
        N = 4
        d_kgEmb = train_dataset.graphEmb.shape[1]
        d_timeEmb = train_dataset.timeEmb.shape[1]
        dropout = 0.1
        d_input = latent_dim # Feature dimension
        d_output = latent_dim
        layernum = 1
        generateModel = Transformer5(
            d_input, d_model, d_output, d_kgEmb, d_timeEmb, q, v, h, N,
            layernum=layernum, dropout=dropout, pe='original'
        ).to(device)
    else:
        raise ValueError(f"Unsupported denoising network: {args.denoise}, supported options: Trans3, Trans5")
    # Create diffusion model
    cfm_model=TorchDiffCfm(
        generateModel,
        latent_dim,
        seq_len,
        train_steps=1000,
        gen_steps=1000,
    )
    # Create diffusion trainer
    trainer = SimpleLatentTrainer(
        config=args,
        cfm_model=cfm_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batchsize,
        num_steps=args.cfm_epochs,
        lr=1e-4,
        device=device,
        logger=None,
        writer=writer,
        results_folder=modelsavepath,
        latent_dim=latent_dim,
        genid=genid
    )
    if args.mode == 'TEST_CFM':
        trainer.load_model()
        trainer.cfm_model.eval()
        raw_params_all = np.load(rawdata_path)
        # save_trajectory(
        #         args=args,
        #         train_dataset=train_dataset,
        #         epoch=args.cfm_epochs,
        #         writer=writer,
        #         eval_index=genid,
        #         diffusion_model=trainer.cfm_model,
        #         ae_trainer=ae_trainer,
        #         outer_dataset=outer_dataset,
        #         test_dataset=test_dataset,
        #         raw_params_all=raw_params_all,
        #         stgnn_config_path=stgnn_config_path,
        #         path_config=eval_path_config,
        #         basemodel=args.basemodel,
        #         device=device
        #     )
        # print("test finished")
        # sys.exit(0)
    else:
        # Start diffusion training
        trainer.train()
    # Post-training evaluation
    if args.evaluate_post_train:
        if not ae_model_path:
            print("AE model path not provided (--ae_model_path), cannot perform decoding evaluation.")
        elif not os.path.exists(rawdata_path):
            print(f"Error: Cannot find original parameter file {rawdata_path}")
        else:
            print(f"Loading original parameters for evaluation: {rawdata_path}")
            raw_params_all = np.load(rawdata_path)
            evaluate_post_diffusion(
                epoch=args.cfm_epochs,
                writer=writer,
                eval_index=genid,
                diffusion_model=trainer.cfm_model,
                ae_trainer=ae_trainer,
                outer_dataset=outer_dataset,
                test_dataset=test_dataset,
                raw_params_all=raw_params_all,
                stgnn_config_path=stgnn_config_path,
                path_config=eval_path_config,
                basemodel=args.basemodel,  # Assume downstream task model is stgcn
                device=device
            )
    print(f"Diffusion model: ModelSave/exp{args.expIndex}/final_diffusion_model.pt")
    print(f"Log: Logs/exp{args.expIndex}.log")
    if args.log_backend == "tensorboard":
        print(f" TensorBoard: TensorBoardLogs/exp{args.expIndex}/")
    elif args.log_backend == "wandb" and writer and getattr(writer, 'run', None):
        run_url = getattr(writer.run, "url", None)
        if run_url:
            print(f" wandb run: {run_url}")
    if writer:
        writer.close() 
