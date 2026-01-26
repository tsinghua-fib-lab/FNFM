import torch
import torch.nn as nn
import math
from einops import rearrange
import sys
import torch.distributions as dist

from TimeTransformer.encoder import Encoder, CrossAttention_Encoder, AdaIN_Encoder
from TimeTransformer.decoder import Decoder
from TimeTransformer.utils import generate_original_PE, generate_regular_PE
import TimeTransformer.causal_convolution_layer as causal_convolution_layer
import torch.nn.functional as F

# sinusoidal positional embeds

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

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class KgEmbedder(nn.Module):
    """Simple two-layer embedder to process kgEmb to 128 dimensions"""
    def __init__(self, input_dim=9, output_dim=128, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Process through two linear layers
        output_emb = self.encoder(x)
        return output_emb # Return the new 128-dim embedding

class Transformer1(nn.Module):
    '''
    good transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 spatialloc: list,
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.1,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 ifkg: bool = True,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        self.ifkg = ifkg
        self.spatialloc = spatialloc
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_mlp = nn.Sequential(
            nn.Linear(self.kgEmb_dim, d_model),
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )
        
        self.timeEmb_mlp = nn.Sequential(
            nn.Linear(self.timeEmb_dim, d_model),
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )
        
        self.timelinear = nn.Linear(self.timeEmb_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(d_model, d_model) 
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape)
        x2 = x.permute(0,2,1) 

        
        timeEmb = self.timelinear(timeEmb)
        timeEmb = timeEmb.unsqueeze(1)
        timeEmb = torch.repeat_interleave(timeEmb, self.layernum, dim=1)

        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)
        encoding.add_(timeEmb)
        if self.ifkg:
            kgEmb = self.kgEmb_mlp(kgEmb)
            kgEmb = kgEmb.unsqueeze(1)        
            kgEmb = torch.repeat_interleave(kgEmb, 160, dim=1)
            encoding[:, self.spatialloc[0]:self.spatialloc[1], :].add_(kgEmb)
        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)
        output = self._linear(encoding)

        return output.permute(0,2,1)

   
class Transformer2(nn.Module):
    '''
    Conditions are added to each layer of transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 N: int,
                 layernum: int = 0,
                 dropout: float = 0.1,
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,  
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout) 
            for _ in range(N)])
        
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_mlp = nn.Sequential(  
            nn.Linear(self.kgEmb_dim, d_model),
        )
        
        self.timeEmb_mlp = nn.Sequential( 
            nn.Linear(self.timeEmb_dim, d_model), 
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:

        # print(x.shape)
        x2 = x.permute(0,2,1)

        kgEmb = self.kgEmb_mlp(kgEmb)
        kgEmb = kgEmb.unsqueeze(1)        
        kgEmb = torch.repeat_interleave(kgEmb, self.layernum, dim=1)
        
        timeEmb = self.timeEmb_mlp(timeEmb)
        timeEmb = timeEmb.unsqueeze(1)
        timeEmb = torch.repeat_interleave(timeEmb, self.layernum, dim=1)

        step = self.step_mlp(t)  
        step = step.unsqueeze(1) 
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)
        
        condition = kgEmb + timeEmb

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)  

        # Encoding stack
        for layer in self.layers_encoding:
            encoding.add_(condition)  # each layer
            encoding = layer(encoding)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)
    
class Transformer3(nn.Module):
    '''
    After the conditions are aggregated, they are added to each layer of transformer
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 N: int,
                 layernum: int = 0,
                 dropout: float = 0.1,
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout,batch_first=True) 
            for _ in range(N)])
        
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_linear = nn.Linear(self.kgEmb_dim, d_model)
        
        self.timeEmb_linear = nn.Linear(self.timeEmb_dim, d_model)
        
        self.forQueryFunc = nn.Sequential(  
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )
        self.condition_scale=nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond=True) -> torch.Tensor:
        B,C,L=x.shape
        # x.shape:  [64, 64, 265]
        xEmb = self._embedding(x.permute(0,2,1))
        kgEmb = kgEmb.to(torch.float32)
        timeEmb = timeEmb.to(torch.float32)
        kgEmb = self.kgEmb_linear(kgEmb)
        kgEmb = kgEmb.unsqueeze(2)
        
        timeEmb = self.timeEmb_linear(timeEmb)
        timeEmb = timeEmb.unsqueeze(2)
        
        kgtEmb = torch.cat((kgEmb, timeEmb), 2)
        
        xQuery = self.forQueryFunc(xEmb)
        
        score = torch.bmm(xQuery, kgtEmb)
        score = F.softmax(score, dim = 2)
        
        condition = torch.bmm(score, torch.transpose(kgtEmb, 1, 2))


        step = self.step_mlp(t)
        step = step.unsqueeze(1).expand(B, L, -1)

        # prepare embedding into encoder
        encoding = xEmb+step
        

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(L, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device).reshape(1,L,self._d_model)
            encoding=encoding+positional_encoding  
            
        # Encoder stack
        for layer in self.layers_encoding: 
            encoding = encoding + self.condition_scale * condition
            encoding = layer(encoding)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)
    
    
class Transformer4(nn.Module):
    '''
        Cross attention with two-layer preprocessing for kgEmb.
    '''
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_timeEmb: int, # Moved up
                 q: int,         # Moved up
                 v: int,         # Moved up
                 h: int,         # Moved up
                 N: int,         # Moved up
                 d_original_kgEmb: int = 9, # Original kgEmb dim (e.g., 9)
                 d_processed_kgEmb: int = 128, # Target kgEmb dim after processing (e.g., 128)
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,
                 kg_hidden_dim: int = 64, # Hidden dimension for KgEmbedder
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.processed_kgEmb_dim = d_processed_kgEmb # Processed output dim
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        # --- KgEmbedder ---
        self.kg_embedder = KgEmbedder(
            input_dim=d_original_kgEmb,
            output_dim=d_processed_kgEmb,
            hidden_dim=kg_hidden_dim
        )
        # --- KgEmbedder End ---

        self.layers_encoding = nn.ModuleList([CrossAttention_Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        # Decoder layers might not be used in diffusion models, confirm if needed
        # self.layers_decoding = nn.ModuleList([...])

        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }
        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model
        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        # These linear layers now process the processed output (128-dim) and timeEmb
        self.kgEmb_linear = nn.Linear(self.processed_kgEmb_dim, d_model) # Input is 128
        self.timeEmb_linear = nn.Linear(self.timeEmb_dim, d_model)

        # self.last_mlp seems unused, maybe remove or check usage
        # self.last_mlp = nn.Sequential(...)


    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond: bool = None) -> torch.Tensor:
        # kgEmb is expected to be the original 9-dimensional embedding here
        # x: (B, C, L), t: (B,), kgEmb: (B, d_original_kgEmb), timeEmb: (B, d_timeEmb)

        x2 = x.permute(0,2,1) # (B, L, C)

        # --- KgEmb Processing ---
        # Pass the original kgEmb through two linear layers to get the target dimension embedding
        kgEmb_processed = self.kg_embedder(kgEmb) # Output shape: (B, d_processed_kgEmb) (e.g., B, 128)
        # --- KgEmb Processing End ---


        # Now use kgEmb_processed (128-dim) for further steps
        kgEmb_mapped = self.kgEmb_linear(kgEmb_processed) # (B, d_model)
        kgEmb_mapped = kgEmb_mapped.unsqueeze(2)          # (B, d_model, 1)

        timeEmb_mapped = self.timeEmb_linear(timeEmb)      # (B, d_model)
        timeEmb_mapped = timeEmb_mapped.unsqueeze(2)      # (B, d_model, 1)

        # Combine the processed kgEmb and timeEmb
        kgtEmb = torch.cat((kgEmb_mapped, timeEmb_mapped), 2) # (B, d_model, 2)

        step = self.step_mlp(t) # (B, d_model)
        step = step.unsqueeze(1) # (B, 1, d_model)
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1) # (B, L, d_model)

        # Embedding module
        encoding = self._embedding(x2) # (B, L, M)
        encoding = encoding + step_emb # Add timestep embedding

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding = encoding + positional_encoding # Add positional encoding

        # Encoding stack (Cross Attention)
        for layer in self.layers_encoding:
            # Pass sequence encoding and the combined condition embedding (kgtEmb)
            encoding = layer(encoding, kgtEmb) # kgtEmb acts as Key/Value context

        output = self._linear(encoding) # (B, L, d_output)

        return output.permute(0,2,1) # (B, d_output, L)

class Transformer5(nn.Module):
    '''
        Adaptive LayerNorm
    '''

    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb:int,
                 d_timeEmb:int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 attention_size: int = None,
                 layernum: int = 0,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk',
                 pe: str = None,
                 pe_period: int = None,
                 learned_sinusoidal_cond: bool = False,   
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16,                 
                 ):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        self.timeEmb_dim = d_timeEmb
        self.channels = d_input
        step_dim = d_model

        self.layernum = layernum

        self.self_condition = False

        self.layers_encoding = nn.ModuleList([AdaIN_Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode,
                                                      kgt_emb_dim=d_kgEmb+d_timeEmb) for _ in range(N)])
        
        self.layers_decoding = nn.ModuleList([Decoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      attention_size=attention_size,
                                                      dropout=dropout,
                                                      chunk_mode=chunk_mode) for _ in range(N)])
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)

        pe_functions = {
            'original': generate_original_PE,  
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        self.kgEmb_linear = nn.Linear(self.kgEmb_dim, d_model)
        
        self.timeEmb_linear = nn.Linear(self.timeEmb_dim, d_model)
        
        self.last_mlp = nn.Sequential(
            nn.ReLU(),  
            nn.Linear(d_model, d_model)  
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond=True) -> torch.Tensor:

        # print(x.shape)
        x2 = x.permute(0,2,1)
        
        kgtEmb = torch.cat((kgEmb, timeEmb), 1)            

        step = self.step_mlp(t) 
        step = step.unsqueeze(1)  
        step_emb = torch.repeat_interleave(step, self.layernum, dim=1)

        # Embedding module
        encoding = self._embedding(x2)
        encoding.add_(step_emb)

        K = self.layernum

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)  

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding, kgtEmb)

        output = self._linear(encoding)
        
        return output.permute(0,2,1)
    
class Transformer6(nn.Module):
    '''
    Standard DiT structure with AdaLN-Zero conditioning.
    Modified to use only kgEmb for AdaLN conditioning.
    '''
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb: int,
                 d_timeEmb: int, # d_timeEmb is now unused for AdaLN conditioning 'c'
                 N: int,  # Number of DiT blocks
                 num_heads: int = 8, # Number of attention heads
                 layernum: int = 0, # Sequence length, required for positional embedding
                 dropout: float = 0.1,
                 mlp_ratio: float = 4.0,
                 learned_sinusoidal_cond: bool = False,
                 random_fourier_features: bool = False,
                 learned_sinusoidal_dim: int = 16
                 ):
        super().__init__()
        self._d_model = d_model
        self.kgEmb_dim = d_kgEmb
        # self.timeEmb_dim = d_timeEmb # No longer strictly needed for conditioning 'c'
        self.channels = d_input
        self.layernum = layernum # Sequence length
        self.self_condition = True # Keep self-conditioning if needed by trainer

        # 1. Input Embedding
        self._embedding = nn.Linear(d_input, d_model)

        # 2. Positional Embedding (Fixed Sinusoidal)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.layernum, d_model), requires_grad=False)
        self.initialize_positional_embeddings()

        # 3. Timestep Embedding (Still used for adding to input sequence)
        step_dim = d_model
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model
        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        # 4. Conditional Embedding MLP (Only kgEmb used for 'c')
        self.kgEmb_mlp = nn.Linear(self.kgEmb_dim, d_model)
        # self.timeEmb_mlp = nn.Linear(self.timeEmb_dim, d_model) # Not needed for 'c'

        # Conditioning MLP to process kg_embed into the final signal 'c'
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), # Input is now just d_model from kg_embed
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # 5. Transformer Blocks (DiT Blocks)
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(N)
        ])

        # 6. Final Layer
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear_final = nn.Linear(d_model, d_output)
        # Initialize the final linear layer weights to zero
        nn.init.zeros_(self.linear_final.weight)
        nn.init.zeros_(self.linear_final.bias)
        self.adaLN_modulation_final = nn.Sequential( # AdaLN for the final layer norm
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True) # Input is d_model from 'c'
        )
        nn.init.zeros_(self.adaLN_modulation_final[-1].weight)
        nn.init.zeros_(self.adaLN_modulation_final[-1].bias)


    def initialize_positional_embeddings(self):
        pos_embed = generate_original_PE(self.layernum, self._d_model) # (SeqLen, Dim)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

    def forward(self, x: torch.Tensor, t: torch.Tensor, kgEmb: torch.Tensor, timeEmb: torch.Tensor, x_self_cond = None) -> torch.Tensor:
        # x: (Batch, Channel, SeqLen)
        # t: (Batch,)
        # kgEmb: (Batch, d_kgEmb)
        # timeEmb: (Batch, d_timeEmb) - Received but not used for AdaLN conditioning 'c'
   
        x = x.permute(0, 2, 1) # (Batch, SeqLen, Channel)

        # Prepare embeddings
        x_embed = self._embedding(x)          # (Batch, SeqLen, d_model)
        t_embed = self.step_mlp(t)            # (Batch, d_model) - Still calculated for adding to input
        kg_embed = self.kgEmb_mlp(kgEmb)      # (Batch, d_model) - This will form the basis of 'c'
        # time_e = self.timeEmb_mlp(timeEmb)    # (Batch, d_model) - No longer needed for 'c'

        # --- Modification Start ---
        # Generate the conditioning signal 'c' solely based on kg_embed
        # The original combined 'cond' is removed.
        c = self.cond_mlp(kg_embed)           # (Batch, d_model) - Final conditioning signal from kg_embed
        # --- Modification End ---

        # Add positional embedding and timestep embedding to the input sequence
        # Note: t_embed is still added here as per standard Transformer/DiT practice
        x_processed = x_embed + self.pos_embed + t_embed.unsqueeze(1) # (Batch, SeqLen, d_model)

        # Apply DiT blocks
        for block in self.blocks:
            x_processed = block(x_processed, c) # Pass conditioning signal `c` (derived from kgEmb) to each block

        # Final Layer Norm with AdaLN
        # Generate scale/shift for final norm based on 'c'
        shift_final, scale_final = self.adaLN_modulation_final(c).chunk(2, dim=1) # (Batch, d_model)
        x_norm_final = self.norm_final(x_processed)
        x_modulated_final = scale_final.unsqueeze(1) * x_norm_final + shift_final.unsqueeze(1)

        # Final linear projection
        output = self.linear_final(x_modulated_final) # (Batch, SeqLen, d_output)

        return output.permute(0, 2, 1) # (Batch, d_output, SeqLen)

class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN).
    Generate scale and shift parameters for LayerNorm through an external condition embedding.
    """
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        # Use LayerNorm without learnable affine parameters, because scale and shift will be provided externally
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # A simple linear layer to map the condition embedding to scale and shift
        self.cond_proj = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        x: main input tensor, shape (B, seq_len, d_model)
        cond_emb: condition embedding, shape (B, cond_dim)
        """
        # 1. generate scale and shift
        scale_shift = self.cond_proj(cond_emb)  # (B, 2 * d_model)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)  # 2 x (B, d_model)

        # 2. adjust shape to broadcast
        # from (B, d_model) to (B, 1, d_model), so that it can act on each token in the sequence
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        # 3. apply AdaLN
        x_normed = self.layer_norm(x)
        # then apply the generated scale and shift. adding 1 is to make the scale approximately 1 during initialization, increasing stability.
        return x_normed * (1 + scale) + shift

class AdaptiveEncoderLayer(nn.Module):
    """
    A standard Transformer Encoder layer using AdaLN.
    """
    def __init__(self, d_model: int, h: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h

        # two adaptive layer normalization modules
        self.ada_ln_1 = AdaptiveLayerNorm(d_model, cond_dim)
        self.ada_ln_2 = AdaptiveLayerNorm(d_model, cond_dim)

        # standard multi-head self-attention and feedforward network
        self.self_attn = nn.MultiheadAttention(d_model, h, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        # pre-LN structure
        # 1. self-attention block
        x_normed = self.ada_ln_1(x, cond_emb)
        attn_output, _ = self.self_attn(x_normed, x_normed, x_normed)
        x = x + self.dropout1(attn_output)

        # 2. feedforward network block
        x_normed = self.ada_ln_2(x, cond_emb)
        ffn_output = self.ffn(x_normed)
        x = x + self.dropout2(ffn_output)
        
        return x


# --- main Transformer model ---

class cfmTransformer(nn.Module):
    """
    Fixed and reconstructed Transformer, using Adaptive LayerNorm.
    """
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_output: int,
                 d_kgEmb: int,
                 d_timeEmb: int,
                 h: int,
                 N: int,
                 dropout: float = 0.1,
                 pe: str = 'regular',
                 ):
        super().__init__()
        self._d_model = d_model

        # 1. condition embedding dimension
        # we project t, graphEmb, timeEmb to d_model, then add them
        # so the final condition dimension is d_model
        cond_dim = d_model

        # 2. define Transformer Encoder layer
        self.layers_encoding = nn.ModuleList([
            AdaptiveEncoderLayer(d_model, h, cond_dim, dropout) for _ in range(N)
        ])

        # 3. input/output linear layer
        self._embedding = nn.Linear(d_input, d_model)
        self._linear = nn.Linear(d_model, d_output)
        self.cond_fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        # 4. position encoding (PE)
        # note: pe_period etc. parameters need to match the generate_regular_PE function
        if pe == 'regular':
            self._generate_PE = self.generate_regular_PE
        else:
            self._generate_PE = None

        # 5. condition information processing module
        # time step t processing network (from DiT)
        self.step_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # project graphEmb and timeEmb to d_model
        self.kgEmb_linear = nn.Linear(d_kgEmb, d_model)
        self.timeEmb_linear = nn.Linear(d_timeEmb, d_model)
        
        # initialize parameters, zero initialize the projection layer of AdaLN, 
        # so that it approximates a standard LN in the early training
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.layers_encoding:
            nn.init.normal_(layer.ada_ln_1.cond_proj.weight, std=1e-4)
            nn.init.zeros_(layer.ada_ln_1.cond_proj.bias)
            nn.init.normal_(layer.ada_ln_2.cond_proj.weight, std=1e-4)
            nn.init.zeros_(layer.ada_ln_2.cond_proj.bias)
        if isinstance(self.cond_fusion_mlp[-1], nn.Linear):
            nn.init.xavier_uniform_(self.cond_fusion_mlp[-1].weight)
            nn.init.zeros_(self.cond_fusion_mlp[-1].bias)


    def forward(self, x: torch.Tensor, t: torch.Tensor, graphEmb: torch.Tensor, timeEmb: torch.Tensor) -> torch.Tensor:
        # 1. adjust input dimension to match the (B, seq_len, d_model) of Transformer
        x = x.permute(0, 2, 1)
        B, L, _ = x.shape

        # 2. fuse all condition information
        # a. process time step t
        step_emb = self.step_mlp(t)
        # b. process knowledge graph and time embedding
        kg_emb = self.kgEmb_linear(graphEmb.to(torch.float32))
        time_emb = self.timeEmb_linear(timeEmb.to(torch.float32))
        # c. merge into a unified condition embedding

        emb_concat = torch.cat([step_emb,kg_emb,time_emb],dim=1)
        cond_emb = self.cond_fusion_mlp(emb_concat)

        # 3. input Embedding
        encoding = self._embedding(x)

        # 4. add position encoding
        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(L, self._d_model)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding=encoding+positional_encoding

        # 5. through Transformer Encoder stack
        for layer in self.layers_encoding:
            encoding = layer(encoding, cond_emb)

        # 6. final linear projection
        output = self._linear(encoding)
        
        # 7. restore original dimension
        return output.permute(0, 2, 1)

    @staticmethod
    def generate_regular_PE(length, d_model):
        """generate standard sinusoidal/cosine position encoding"""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

