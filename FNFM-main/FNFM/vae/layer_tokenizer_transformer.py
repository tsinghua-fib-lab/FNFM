import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# WeightGraphOrganizer class from FNO_GraphVAE - independent functional module
class WeightGraphOrganizer:
    """
    Weight Graph Organizer - organizes weights as output-centered graph nodes according to the paper.
    This implementation aggregates weights belonging to the same logical block (e.g., TimeBlock) into single node features.
    Supports layer-specific normalization operations.
    """
    def __init__(self, use_layer_norm=True, learnable_norm=False,model_name='v_STGCN5'):
        self.use_layer_norm = use_layer_norm
        self.learnable_norm = learnable_norm
        
        # 1. Define individual weight tensors and their order in the flattened vector
        # Note: This order must strictly match the way the flattened weight vector is created.
        # Here we follow the data flow order from meta_stgcn.py.
        if model_name == 'v_STGCN5':
            self.param_configs_ordered = [# Corrected based on meta_stgcn.py
                ('block1.Theta1', {'shape': (32, 8)}), 
                ('block1.temporal1.conv1', {'shape': (32, 2, 1, 4)}),
                ('block1.temporal1.conv2', {'shape': (32, 2, 1, 4)}),
                ('block1.temporal1.conv3', {'shape': (32, 2, 1, 4)}),
                ('block1.temporal2.conv1', {'shape': (32, 8, 1, 4)}),
                ('block1.temporal2.conv2', {'shape': (32, 8, 1, 4)}),
                ('block1.temporal2.conv3', {'shape': (32, 8, 1, 4)}),
                ('last_temporal.conv1', {'shape': (32, 32, 1, 4)}),
                ('last_temporal.conv2', {'shape': (32, 32, 1, 4)}),
                ('last_temporal.conv3', {'shape': (32, 32, 1, 4)}),
                ('fully', {'shape': (6, 96)})
            ]
            self.param_configs = dict(self.param_configs_ordered)

            # 2. Define logical blocks to group weights into nodes
            self.logical_configs_ordered = [
                ('theta', {
                    'type': 'Linear',
                    'params': ['block1.Theta1'],
                    'category': 'theta',
                }),
                ('temporal1', {
                    'type': 'TimeBlock',
                    'params': ['block1.temporal1.conv1', 'block1.temporal1.conv2', 'block1.temporal1.conv3'],
                    'category': 'temporal1',
                }),
                ('temporal2', {
                    'type': 'TimeBlock',
                    'params': ['block1.temporal2.conv1', 'block1.temporal2.conv2', 'block1.temporal2.conv3'],
                    'category': 'temporal2',
                }),
                ('last_temporal', {
                    'type': 'TimeBlock',
                    'params': ['last_temporal.conv1', 'last_temporal.conv2', 'last_temporal.conv3'],
                    'category': 'last_temporal',
                }),
                ('fully', {
                    'type': 'Linear',
                    'params': ['fully'],
                    'category': 'fully',
                })
            ]
            self.logical_configs = dict(self.logical_configs_ordered)
            self.category_mapping = {'temporal1': 0, 'theta': 1, 'temporal2': 2, 'last_temporal': 3, 'fully': 4}
        elif model_name == 'v_GWN':
            # 1. Define individual weight tensors for v_GWN based on meta_gwn.py
            # Default parameters: residual_channels=32, dilation_channels=32, skip_channels=32, 
            # end_channels=32, blocks=4, layers=2, kernel_size=2, in_dim=2, out_dim=6
            # gcn parameters: order=2, support_len=2 -> gcn_in_channels = (2*2+1)*32 = 160
            
            self.param_configs_ordered = [
                ('start_conv', {'shape': (32, 2, 1, 1)})
            ]
            
            # 8 WaveNet blocks (4 blocks * 2 layers)
            for i in range(8):
                self.param_configs_ordered.extend([
                    (f'filter_convs.{i}', {'shape': (32, 32, 1, 2)}),
                    (f'gate_convs.{i}', {'shape': (32, 32, 1, 2)}),
                    (f'residual_convs.{i}', {'shape': (32, 32, 1, 1)}),
                    (f'skip_convs.{i}', {'shape': (32, 32, 1, 1)}),
                    (f'bn.{i}.weight', {'shape': (32,)}),
                    (f'bn.{i}.bias', {'shape': (32,)}),
                    # Note: The gconv module contains a linear layer which is a Conv2d
                    (f'gconv.{i}.mlp.mlp', {'shape': (32, 160, 1, 1)}) 
                ])

            self.param_configs_ordered.extend([
                ('end_conv_1', {'shape': (32, 32, 1, 1)}),
                ('end_conv_2', {'shape': (6, 32, 1, 1)})
            ])
            
            self.param_configs = dict(self.param_configs_ordered)

            # 2. Define logical blocks to group weights into nodes
            self.logical_configs_ordered = [
                ('start_conv', {
                    'type': 'TimeBlock', # Using TimeBlock as it handles Conv2d layers
                    'params': ['start_conv'],
                    'category': 'start_conv',
                })
            ]

            for i in range(8):
                self.logical_configs_ordered.append(
                    (f'wavenet_block_{i}', {
                        'type': 'TimeBlock', # All params are Conv2d, so TimeBlock logic applies
                        'params': [
                            f'filter_convs.{i}',
                            f'gate_convs.{i}',
                            f'residual_convs.{i}',
                            f'skip_convs.{i}',
                            f'bn.{i}.weight',
                            f'bn.{i}.bias',
                            f'gconv.{i}.mlp.mlp'
                        ],
                        'category': f'wavenet_block_{i}',
                    })
                )

            self.logical_configs_ordered.extend([
                ('end_conv_1', {
                    'type': 'TimeBlock',
                    'params': ['end_conv_1'],
                    'category': 'end_conv_1',
                }),
                ('end_conv_2', {
                    'type': 'TimeBlock',
                    'params': ['end_conv_2'],
                    'category': 'end_conv_2',
                })
            ])

            self.logical_configs = dict(self.logical_configs_ordered)

            # 3. Create category mapping
            categories = [name for name, _ in self.logical_configs_ordered]
            self.category_mapping = {name: i for i, name in enumerate(categories)}
            
        self._calculate_param_indices()
        self._create_node_mapping()
        
    def _calculate_param_indices(self):
        """Calculate the start and end indices of each weight tensor in the flattened vector."""
        self.param_indices = {}
        current_idx = 0
        for name, config in self.param_configs_ordered:
            param_count = np.prod(config['shape'])
            self.param_indices[name] = {
                'start': current_idx,
                'end': current_idx + param_count,
                'shape': config['shape']
            }
            current_idx += param_count
        self.total_params = current_idx

    def _create_node_mapping(self,model_name='v_STGCN5'):
        """Create node information based on logical blocks and calculate maximum feature dimension."""
        self.node_info = []
        self.logical_layer_info = {}
        max_feature_dim = 0
        
        node_start_idx = 0
        for name, config in self.logical_configs_ordered:
            node_feature_dim = 0
            num_nodes = 0
            
            # Determine number of nodes (output units) and node feature dimension
            first_param_name = config['params'][0]
            first_param_shape = self.param_configs[first_param_name]['shape']
            if config['type'] == 'TimeBlock':
                num_nodes = first_param_shape[0]  # Cout
                for param_name in config['params']:
                    shape = self.param_configs[param_name]['shape']
                    node_feature_dim += int(np.prod(shape[1:])) # Cin * h * w
            elif config['type'] == 'Linear':
                shape = first_param_shape
                if name == 'theta':
                    # Theta1 is (in_features=32, out_features=8)
                    # Node is output neuron, so num_nodes = out_features = 8
                    num_nodes = shape[1]
                    # Feature is weights connecting to that output neuron, so node_feature_dim = in_features = 32
                    node_feature_dim = shape[0]
                else: # fully, shape is (out, in)
                    num_nodes = shape[0]
                    node_feature_dim = shape[1]
            # if model_name == 'v_GWN':
            #     # For GWN, all defined logical blocks are collections of Conv2d layers.
            #     # We can reuse the 'TimeBlock' logic where each output channel is a node.
            #     if config['type'] == 'TimeBlock':
            #         num_nodes = first_param_shape[0]  # Cout
            #         for param_name in config['params']:
            #             shape = self.param_configs[param_name]['shape']
            #             node_feature_dim += np.prod(shape[1:]) # Cin * h * w
 
            self.logical_layer_info[name] = {
                'num_nodes': num_nodes,
                'node_feature_dim': node_feature_dim,
                'node_start_idx': node_start_idx,
                'node_end_idx': node_start_idx + num_nodes
            }
            
            if node_feature_dim > max_feature_dim:
                max_feature_dim = node_feature_dim
            
            for i in range(num_nodes):
                self.node_info.append({
                    'node_idx': node_start_idx + i,
                    'logical_layer': name,
                    'category_id': self.category_mapping[config['category']],
                    'intra_layer_idx': i
                })
            
            node_start_idx += num_nodes

        self.total_nodes = node_start_idx
        self.max_feature_dim = max_feature_dim
    def flat_to_weight_graph(self, flat_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert flattened weights to weight graph representation."""
        if flat_weights.dim() == 1:
            flat_weights = flat_weights.unsqueeze(0)
        
        batch_size = flat_weights.shape[0]
        device = flat_weights.device
        
        weight_graph = torch.zeros(batch_size, self.total_nodes, self.max_feature_dim, device=device)
        category_ids = torch.zeros(self.total_nodes, dtype=torch.long, device=device)
        node_positions = torch.zeros(self.total_nodes, dtype=torch.long, device=device)

        for name, config in self.logical_configs.items():
            info = self.logical_layer_info[name]
            num_nodes, feature_dim = info['num_nodes'], info['node_feature_dim']
            
            # 1. Extract and reshape all weights belonging to this logical block
            block_weights = []
            for param_name in config['params']:
                indices = self.param_indices[param_name]
                shape = indices['shape']
                w = flat_weights[:, indices['start']:indices['end']].view(batch_size, *shape)
                block_weights.append(w)
            # 2. Build features for each output unit (node)
            for i in range(num_nodes):
                node_feature_parts = []
                if config['type'] == 'TimeBlock':
                    for w in block_weights: # w shape: [B, Cout, Cin, h, w]
                        feature_part = w[:, i, ...].reshape(batch_size, -1)
                        node_feature_parts.append(feature_part)
                elif config['type'] == 'Linear':
                    w = block_weights[0] # The only weight tensor
                    if name == 'theta': # shape: [B, Din, Dout]
                        # Each output node's feature is the column vector
                        feature_part = w[:, :, i]
                    else: # fully, shape: [B, Dout, Din]
                        feature_part = w[:, i, :]
                    node_feature_parts.append(feature_part)
                # 3. Concatenate, normalize, and pad
                node_feature = torch.cat(node_feature_parts, dim=1)
                
                node_idx = info['node_start_idx'] + i
                weight_graph[:, node_idx, :feature_dim] = node_feature
                category_ids[node_idx] = self.category_mapping[config['category']]
                node_positions[node_idx] = i

        return weight_graph, category_ids, node_positions

    def weight_graph_to_flat(self, weight_graph: torch.Tensor, category_ids: torch.Tensor, node_positions: torch.Tensor) -> torch.Tensor:
        """Convert weight graph back to flattened weights."""
        batch_size = weight_graph.shape[0]
        device = weight_graph.device
        reconstructed_params = {}

        for name, config in self.logical_configs.items():
            info = self.logical_layer_info[name]
            num_nodes, feature_dim = info['num_nodes'], info['node_feature_dim']
            
            # Initialize weight tensors for this logical block
            block_reconstructed_weights = {p_name: torch.zeros(batch_size, *self.param_configs[p_name]['shape'], device=device) for p_name in config['params']}

            for i in range(num_nodes):
                node_idx = info['node_start_idx'] + i
                node_feature = weight_graph[:, node_idx, :feature_dim]
                
                # Split features and place back into corresponding weight tensors
                current_offset = 0
                if config['type'] == 'TimeBlock':
                    for param_name in config['params']:
                        shape = self.param_configs[param_name]['shape']
                        part_dim = int(np.prod(shape[1:]))
                        feature_part = node_feature[:, current_offset : current_offset + part_dim]
                        block_reconstructed_weights[param_name][:, i, ...] = feature_part.view(batch_size, *shape[1:])
                        current_offset += part_dim
                elif config['type'] == 'Linear':
                    param_name = config['params'][0]
                    if name == 'theta': # [B, Din, Dout]
                        block_reconstructed_weights[param_name][:, :, i] = node_feature
                    else: # fully [B, Dout, Din]
                        block_reconstructed_weights[param_name][:, i, :] = node_feature
            
            reconstructed_params.update(block_reconstructed_weights)
            
        # Concatenate into flattened vector
        flat_weights = torch.zeros(batch_size, self.total_params, device=device)
        for name, indices in self.param_indices.items():
            flat_weights[:, indices['start']:indices['end']] = reconstructed_params[name].reshape(batch_size, -1)
            
        return flat_weights

class LayerTokenizer(nn.Module):
    """
    Layer Tokenizer based on weight graph nodes
    Supports layer normalization and uses independent linear projections for each logical layer.
    """
    def __init__(self, d_model: int = 256, max_positions: int = 400, 
                 use_layer_norm: bool = True, learnable_norm: bool = False,model_name='v_STGCN5'):
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        
        # Weight graph organizer (supports layer normalization)
        self.weight_organizer = WeightGraphOrganizer(use_layer_norm, learnable_norm,model_name)

        # Create independent node feature projections for each logical layer
        self.node_projections = nn.ModuleDict()
        for name, info in self.weight_organizer.logical_layer_info.items():
            self.node_projections[name] = nn.Linear(info['node_feature_dim'], d_model)
        
    def forward(self, flat_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass
        Args:
            flat_weights: [batch_size, 16960] flattened weights
        Returns:
            tokens: [batch_size, num_nodes, d_model] weight graph tokens
            metadata: dictionary containing weight graph metadata
        """
        # Convert to weight graph
        weight_graph, category_ids, node_positions = self.weight_organizer.flat_to_weight_graph(flat_weights)
        batch_size, num_nodes, _ = weight_graph.shape
        
        # Perform node feature projection layer by layer
        projected_tokens_list = []
        for name, info in self.weight_organizer.logical_layer_info.items():
            start_idx, end_idx = info['node_start_idx'], info['node_end_idx']
            feature_dim = info['node_feature_dim']
            
            # Extract features of current layer (unpadded part)
            layer_features = weight_graph[:, start_idx:end_idx, :feature_dim]
            
            # Apply independent linear projection for this layer
            projected = self.node_projections[name](layer_features)
            projected_tokens_list.append(projected)

        # Concatenate all projected tokens
        tokens = torch.cat(projected_tokens_list, dim=1)
        
        # Prepare metadata
        metadata = {
            'weight_graph': weight_graph, # Note: this is the padded graph before projection
            'category_ids': category_ids,
            'node_positions': node_positions,
            'num_nodes': num_nodes # Total number of nodes (e.g., 110)
        }
        
        return tokens, metadata
    
    def decode(self, tokens: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Decode tokens back to flattened weights
        Args:
            tokens: [batch_size, num_nodes, d_model]
            metadata: metadata returned during tokenization
        Returns:
            flat_weights: [batch_size, 16960]
        """
        batch_size = tokens.shape[0]
        # Create an empty, padded weight graph tensor to receive back-projection results
        reconstructed_weight_graph = torch.zeros(
            batch_size, 
            self.weight_organizer.total_nodes, 
            self.weight_organizer.max_feature_dim, 
            device=tokens.device,
            dtype=tokens.dtype
        )

        # Perform back-projection layer by layer
        for name, info in self.weight_organizer.logical_layer_info.items():
            start_idx, end_idx = info['node_start_idx'], info['node_end_idx']
            feature_dim = info['node_feature_dim']
            
            # Extract tokens of current layer
            layer_tokens = tokens[:, start_idx:end_idx, :]
            
            # Use transposed weights of corresponding linear layer for back-projection
            projection_layer = self.node_projections[name]
            # [B, num_layer_nodes, d_model] -> [B, num_layer_nodes, feature_dim]
            reconstructed_features = F.linear(layer_tokens, projection_layer.weight.t())
            
            # Place back-projection results back to correct positions in weight graph tensor
            reconstructed_weight_graph[:, start_idx:end_idx, :feature_dim] = reconstructed_features
        
        # Convert back to flattened weights
        flat_weights = self.weight_organizer.weight_graph_to_flat(
            reconstructed_weight_graph, metadata['category_ids'], metadata['node_positions']
        )
        
        return flat_weights

class LayerTransformer(nn.Module):
    """
    Shared Transformer encoder based on layer-specific processing
    """
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 num_categories: int = 5,
                 max_intra_layer_nodes: int = 32):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Positional Embeddings
        self.category_embedding = nn.Embedding(num_categories, d_model)
        self.position_embedding = nn.Embedding(max_intra_layer_nodes, d_model)

        # --- Encoder Attention Blocks ---
        self.encoder_attn_layers = nn.ModuleList([])
        self.encoder_ffn_layers = nn.ModuleList([])
        self.encoder_norm1_layers = nn.ModuleList([])
        self.encoder_norm2_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.encoder_norm1_layers.append(nn.LayerNorm(d_model))
            self.encoder_attn_layers.append(
                nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                        dropout=dropout, batch_first=True)
            )
            self.encoder_norm2_layers.append(nn.LayerNorm(d_model))
            # FFN hidden dim often 4*d_model in Transformers
            self.encoder_ffn_layers.append(FeedForward(d_model, d_model * 4, dropout))
        # --- Decoder Attention Blocks ---
        self.decoder_attn_layers = nn.ModuleList([])
        self.decoder_ffn_layers = nn.ModuleList([])
        self.decoder_norm1_layers = nn.ModuleList([])
        self.decoder_norm2_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.decoder_norm1_layers.append(nn.LayerNorm(d_model))
            self.decoder_attn_layers.append(
                nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                        dropout=dropout, batch_first=True)
            )
            self.decoder_norm2_layers.append(nn.LayerNorm(d_model))
            # FFN hidden dim often 4*d_model in Transformers
            self.decoder_ffn_layers.append(FeedForward(d_model, d_model * 4, dropout))

    def init_parameters(self, mean=0.0, std=0.001): # Adjusted std based on typical transformer init

        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def init_layernorm(m):
             if isinstance(m, nn.LayerNorm):
                  nn.init.constant_(m.bias, 0)
                  nn.init.constant_(m.weight, 1.0)

        # Encoder / Decoder Attention Block Layers
        for i in range(self.num_layers):
            # Attention weights (handled by nn.MultiheadAttention internal init)
            # FFN layers
            self.encoder_ffn_layers[i].apply(init_linear)
            self.decoder_ffn_layers[i].apply(init_linear)
            # LayerNorm
            init_layernorm(self.encoder_norm1_layers[i])
            init_layernorm(self.encoder_norm2_layers[i])
            init_layernorm(self.decoder_norm1_layers[i])
            init_layernorm(self.decoder_norm2_layers[i])
        
    def encode(self, tokens: torch.Tensor, category_ids: torch.Tensor, node_positions: torch.Tensor):
        """
        Forward pass
        Args:
            tokens: [batch_size, num_nodes, d_model] input tokens
            category_ids: [num_nodes]
            node_positions: [num_nodes]
        Returns:
            output_tokens: [batch_size, num_nodes, d_model] output tokens (after layer-specific processing)
        """
        # Add positional embeddings
        cat_embed = self.category_embedding(category_ids)
        pos_embed = self.position_embedding(node_positions)
        tokens = tokens + cat_embed.unsqueeze(0) + pos_embed.unsqueeze(0)

        for i in range(self.num_layers):
            # --- Attention Block ---
            # 1. Layer Norm 1 -> Multi-Head Self-Attention -> Residual 1
            x_norm1 = self.encoder_norm1_layers[i](tokens)
            attn_output, _ = self.encoder_attn_layers[i](x_norm1, x_norm1, x_norm1)
            tokens = tokens + attn_output # Residual connection 1

            # 2. Layer Norm 2 -> Feed Forward -> Residual 2
            x_norm2 = self.encoder_norm2_layers[i](tokens)
            ffn_output = self.encoder_ffn_layers[i](x_norm2)
            tokens = tokens + ffn_output # Residual connection 2
        return tokens
    def decode(self,tokens: torch.Tensor, category_ids: torch.Tensor, node_positions: torch.Tensor):
        """
        Forward pass
        Args:
            tokens: [batch_size, num_nodes, d_model] input tokens
            category_ids: [num_nodes]
            node_positions: [num_nodes]
        Returns:
            output_tokens: [batch_size, num_nodes, d_model] output tokens (after layer-specific processing)
        """
        # Add positional embeddings
        cat_embed = self.category_embedding(category_ids)
        pos_embed = self.position_embedding(node_positions)
        tokens = tokens + cat_embed.unsqueeze(0) + pos_embed.unsqueeze(0)

        for i in range(self.num_layers):
            # --- Attention Block ---
            # 1. Layer Norm 1 -> Multi-Head Self-Attention -> Residual 1
            x_norm1 = self.decoder_norm1_layers[i](tokens)
            attn_output, _ = self.decoder_attn_layers[i](x_norm1, x_norm1, x_norm1)
            tokens = tokens + attn_output # Residual 1
            # 2. Layer Norm 2 -> Feed Forward -> Residual 2
            x_norm2 = self.decoder_norm2_layers[i](tokens)
            ffn_output = self.decoder_ffn_layers[i](x_norm2)
            tokens = tokens + ffn_output # Residual 2
        return tokens

class LayerVAE(nn.Module):
    """
    Variational Autoencoder (VAE) based on LayerTokenizer and LayerTransformer
    Following the design philosophy of FNO_GraphVAE.
    """
    def __init__(self,
                 scale, 
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 2, # number of transformer layers
                 latent_dim: int = 64, # dimension of each node's latent vector
                 kl_weight: float = 1.0,
                 dropout: float = 0.1,
                 model_name='v_STGCN5'):
        super().__init__()
        print("latent_dim",latent_dim)
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.nhead=nhead
        self.num_layers=num_layers
        self.scale = scale
        # Layer Tokenizer
        self.model_name=model_name
        self.tokenizer = LayerTokenizer(d_model=d_model,model_name=model_name)
        # Get total number of nodes, which will be the second dimension of each batch item in latent space
        organizer = self.tokenizer.weight_organizer
        self.num_nodes = organizer.total_nodes # e.g., 110 nodes

        num_categories = len(organizer.category_mapping)
        max_intra_layer_nodes = max(info['num_nodes'] for info in organizer.logical_layer_info.values()) if organizer.logical_layer_info else 0

        # Layer Transformer (Encoder)
        self.transformer = LayerTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            num_categories=num_categories,
            max_intra_layer_nodes=max_intra_layer_nodes
        )
        self.transformer.init_parameters() # Initialize parameters
        # Encoder: Projects the final d_model-dimensional encoded tokens
        # to mu and log_var for each node.
        # Input: (batch_size, num_nodes, d_model)
        # Output: (batch_size, num_nodes, latent_dim * 2)
        self.encoder_mu_logvar_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim * 2)
        )
        
        # Decoder: Projects the sampled latent vector (z) back to d_model space
        # Input: (batch_size, num_nodes, latent_dim)
        # Output: (batch_size, num_nodes, d_model)
        self.decoder_latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model) # FNO_GraphVAE uses an additional linear layer here
        )
        
        # Layer Transformer (Decoder)
        self.decoder_transformer = LayerTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            num_categories=num_categories,
            max_intra_layer_nodes=max_intra_layer_nodes
        )

        # Precompute static metadata needed for decoding from tokenizer's organizer
        category_ids = torch.tensor([info['category_id'] for info in organizer.node_info], dtype=torch.long)
        node_positions = torch.tensor([info['intra_layer_idx'] for info in organizer.node_info], dtype=torch.long)
        self.register_buffer('category_ids', category_ids, persistent=False)
        self.register_buffer('node_positions', node_positions, persistent=False)
    
    def encode(self, flat_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor]:
        """
        Encoder part: encode flattened weights into mean and log variance in latent space.
        Args:
            flat_weights: [batch_size, total_params]
        Returns:
            mu: [batch_size, num_nodes, latent_dim] mean
            log_var: [batch_size, num_nodes, latent_dim] log variance
            tokens: [batch_size, num_nodes, d_model] input tokens of encoder Transformer (for residual connection or debugging)
            metadata: dictionary containing weight graph metadata (for decoding)
        """
        # Tokenize input flattened weights
        flat_weights=(flat_weights-self.scale[0])/self.scale[1] # Normalization
        tokens, metadata = self.tokenizer(flat_weights) # tokens: (B, num_nodes, d_model)
        
        # Encoder Transformer processes tokens
        # `encoded_tokens` shape is (batch_size, num_nodes, d_model)
        encoded_tokens= self.transformer.encode(tokens, metadata['category_ids'], metadata['node_positions']) 
        
        # Project encoder Transformer's output to mu and log_var
        # Shape is (batch_size, num_nodes, latent_dim * 2)
        mu_logvar = self.encoder_mu_logvar_proj(encoded_tokens)
        
        # Split into mu and log_var
        mu, log_var = mu_logvar.chunk(2, dim=-1) # (B, num_nodes, latent_dim) each
        
        return mu, log_var, tokens, metadata # Return tokens and metadata for decoder use

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from latent distribution.
        Args:
            mu: [batch_size, num_nodes, latent_dim] mean
            log_var: [batch_size, num_nodes, latent_dim] log variance
        Returns:
            z: [batch_size, num_nodes, latent_dim] sampled latent vector
        """
        std = torch.exp(0.5 * log_var) # Standard deviation
        eps = torch.randn_like(std)    # Sample noise from standard normal distribution
        z = mu + eps * std             # Sample latent vector
        return z

    def decode(self, z: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Decoder part: decode latent vector back to flattened weights.
        Args:
            z: [batch_size, num_nodes, latent_dim] sampled latent vector
            metadata: dictionary containing weight graph metadata (obtained from encode stage)
        Returns:
            reconstructed_weights: [batch_size, total_params] reconstructed flattened weights
        """
        B,N_total,H_dim = z.shape
        # Project latent vector z back to d_model dimension as input to decoder Transformer
        # Shape is (batch_size, num_nodes, d_model)
        decoded_internal_dim = self.decoder_latent_proj(z)
        
        # Decoder Transformer processing
        # Shape is (batch_size, num_nodes, d_model)
        decoded_tokens= self.transformer.decode(decoded_internal_dim, metadata['category_ids'], metadata['node_positions'])
        
        # Decode decoder Transformer's output tokens back to flattened weights
        reconstructed_weights_scaled = self.tokenizer.decode(decoded_tokens, metadata)
        reconstructed_weights=reconstructed_weights_scaled * self.scale[1] + self.scale[0] # Denormalization
        return reconstructed_weights
    
    def forward(self, flat_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete VAE forward pass.
        Args:
            flat_weights: [batch_size, total_params] original flattened weights
        Returns:
            reconstruction: [batch_size, total_params] reconstructed flattened weights
            mu: [batch_size, num_nodes, latent_dim] mean
            log_var: [batch_size, num_nodes, latent_dim] log variance
            z: [batch_size, num_nodes, latent_dim] sampled latent vector
        """
        mu, log_var, tokens, metadata = self.encode(flat_weights)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, metadata)
        
        return reconstruction, mu, log_var, z

    def training_losses(self, flat_weights_orig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate VAE training losses (reconstruction loss + KL divergence loss).
        Args:
            flat_weights_orig: [batch_size, total_params] original flattened weights
        Returns:
            total_loss: total loss
            recon_loss: reconstruction loss (MSE)
            kl_div_loss: KL divergence loss
        """
        # Forward pass to get reconstruction results and latent variable parameters
        reconstruction, mu, log_var, z = self.forward(flat_weights_orig)

        # 1. Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(reconstruction, flat_weights_orig, reduction='mean')

        # 2. KL divergence loss (following FNO_GraphVAE's calculation method)
        # KL = 0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kld_element = 1 + log_var - mu.pow(2) - log_var.exp()
        kl_div_loss = -0.5 * torch.mean(kld_element) # Average over all nodes and batches

        # 3. Total VAE loss
        total_loss = recon_loss + self.kl_weight * kl_div_loss

        return total_loss, recon_loss, kl_div_loss

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent space without relying on original input tokens.
        Args:
            z: [batch_size, num_nodes, latent_dim] sampled latent vector
        Returns:
            reconstructed_weights: [batch_size, total_params] reconstructed flattened weights
        """
        # Prepare static metadata needed for decoding (from `self.register_buffer`)
        metadata = {
            'category_ids': self.category_ids,
            'node_positions': self.node_positions,
            'num_nodes': self.num_nodes # Total number of nodes
        }
        
        # Call decode method
        reconstructed_weights = self.decode(z, metadata)
        return reconstructed_weights


def demo_layer_vae(model_name='v_STGCN5'):
    """Demonstrate LayerVAE usage and dimension changes"""
    print("=" * 80)
    print("LayerVAE (Variational AutoEncoder) Dimension Demonstration")
    print("=" * 80)
    
    # 1. Initialize components
    batch_size = 4
    d_model = 256
    latent_dim = 64
    num_layers = 3 # Simplified number of Transformer layers
    scale=[0,1]
    # Assumed total flattened weights (obtained from WeightGraphOrganizer)
    dummy_organizer = WeightGraphOrganizer(model_name=model_name)
    total_flat_params = dummy_organizer.total_params # 16960
    total_nodes = dummy_organizer.total_nodes # 110
    
    flat_weights = torch.randn(batch_size, total_flat_params)
    
    vae = LayerVAE(d_model=d_model, latent_dim=latent_dim, num_layers=num_layers,model_name=model_name,scale=scale)

    print(f"Input flattened weights shape: {flat_weights.shape}")
    print(f"Transformer model dimension (d_model): {d_model}")
    print(f"Latent space dimension (latent_dim): {latent_dim} (per node)")
    print(f"Total number of nodes (num_nodes): {vae.num_nodes}")
    print("-" * 80)

    # 2. Encoding process
    print("--- Encoding Process ---")
    mu, log_var, tokens, metadata = vae.encode(flat_weights)
    print(f"Tokenizer output tokens shape: {tokens.shape} (Batch, NumNodes, d_model)")
    print(f"After encoder Transformer, mu shape: {mu.shape} (Batch, NumNodes, latent_dim)")
    print(f"After encoder Transformer, log_var shape: {log_var.shape} (Batch, NumNodes, latent_dim)")
    print("-" * 80)

    # 3. Reparameterization
    print("--- Reparameterization ---")
    z = vae.reparameterize(mu, log_var)
    print(f"Sampled latent vector z shape: {z.shape} (Batch, NumNodes, latent_dim)")
    print("-" * 80)

    # 4. Decoding process
    print("--- Decoding Process ---")
    reconstruction = vae.decode(z, metadata)
    print(f"Decoder input (after z projection): {vae.decoder_latent_proj(z).shape} (Batch, NumNodes, d_model)")
    # Corrected demo call to use the right method and arguments
    demo_decoded_tokens = vae.transformer.decode(vae.decoder_latent_proj(z), metadata['category_ids'], metadata['node_positions'])
    print(f"After decoder Transformer, tokens shape before tokenizer decode: {demo_decoded_tokens.shape} (Batch, NumNodes, d_model)")
    print(f"Final reconstructed flattened weights shape: {reconstruction.shape} (Batch, TotalFlatParams)")
    print("-" * 80)

    # 5. Complete forward and loss calculation
    print("--- Complete Forward & Loss Calculation ---")
    recon_output, mu_fwd, log_var_fwd, z_fwd = vae.forward(flat_weights)
    print(f"Forward pass reconstruction output shape: {recon_output.shape}")
    print(f"Forward pass mu shape: {mu_fwd.shape}")
    print(f"Forward pass log_var shape: {log_var_fwd.shape}")
    print(f"Forward pass z shape: {z_fwd.shape}")

    total_loss, recon_loss, kl_div_loss = vae.training_losses(flat_weights)
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence loss: {kl_div_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    print("-" * 80)

    # 6. Decode from latent space (without relying on original input)
    print("--- Decode from Latent Space ---")
    # Create a random latent vector Z
    # Ensure Z shape matches VAE's latent space definition: (batch_size, num_nodes, latent_dim)
    random_z = torch.randn(batch_size, vae.num_nodes, vae.latent_dim)
    reconstructed_from_random_z = vae.decode_from_latent(random_z)
    print(f"Reconstructed flattened weights shape from random latent vector Z ({random_z.shape}): {reconstructed_from_random_z.shape}")
    print("=" * 80)


if __name__ == "__main__":
    demo_layer_vae(model_name='v_GWN')
