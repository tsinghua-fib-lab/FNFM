
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from functools import partial
import numpy as np
from collections import namedtuple
from einops import rearrange, reduce
from torchcfm import ExactOptimalTransportConditionalFlowMatcher
import copy
from torchcfm.optimal_transport import OTPlanSampler
import torchdiffeq
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchdyn.core import NeuralODE

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0

def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon

class TorchWrapper(nn.Module):
    """
    Wraps the CFM model to be compatible with torchdyn.NeuralODE's (t, x) signature.
    The CFM model expects (x, t) as separate inputs.
    """
    def __init__(self, model,graphEmb,timeEmb):
        super().__init__()
        self.model = model
        self.graphEmb=graphEmb
        self.timeEmb=timeEmb

    def forward(self, t, x, *args, **kwargs):
        # t from odeint is (1,) or scalar, x is (batch, dim)
        # We need to expand t to (batch, 1) to concatenate with x
        t_expanded = t.expand(x.shape[0])
        # The underlying CFM model expects x and t separately
        return self.model(x, t_expanded,self.graphEmb,self.timeEmb)



class TorchDiffCfm(nn.Module):
    #batch_size, channels, seq_length
    """
    Diffusion model implemented with torchdiffeq
    """
    def __init__(
        self,
        model,
        channels,
        seq_length,
        train_steps=1000,
        gen_steps=1000,
        loss_type='l2',
        objective='pred_noise',
        solver='dopri5',
        atol=1e-5,
        rtol=1e-5,
   #     sigma=0.01
    ):
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        self.channels = channels
        self.gen_steps = gen_steps
        self.train_steps = train_steps
        self.objective = objective
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        self.sigma=0.1
        self.FM= ConditionalFlowMatcher(sigma=self.sigma)
        self.ot_sampler=OTPlanSampler(method='exact')
        # loss function
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')
    @torch.no_grad()
    def give_traj(self, graphEmb=None, timeEmb=None, batch_size=1, return_intermediate=False):
        """
        Sample using an ODE solver
        """
        device = next(self.parameters()).device
        
        self.model.eval()
        node=NeuralODE(TorchWrapper(self.model,graphEmb,timeEmb),solver="dopri5",sensitivity='adjoint',atol=self.atol,rtol=self.rtol)
        with torch.no_grad():
            # print(batch_size,self.channels,self.seq_length)
            # print("such an order")
            traj = node.trajectory(
                        torch.randn(batch_size, self.channels,self.seq_length , device=device),
                        torch.linspace(0.0, 1.0, self.gen_steps, device=device),
                    )
            # extract final result
            return traj
    def sample(self, graphEmb=None, timeEmb=None, batch_size=1, return_intermediate=False):
        """
        Sample using an ODE solver
        """
        device = next(self.parameters()).device
        
        self.model.eval()
        node=NeuralODE(TorchWrapper(self.model,graphEmb,timeEmb),solver="dopri5",sensitivity='adjoint',atol=self.atol,rtol=self.rtol)
        with torch.no_grad():
            # print(batch_size,self.channels,self.seq_length)
            # print("such an order")
            traj = node.trajectory(
                        torch.randn(batch_size, self.channels,self.seq_length , device=device),
                        torch.linspace(0.0, 1.0, self.gen_steps, device=device),
                    )
            # extract final result
            x_final = traj[-1]  # last time step
            x_final = x_final.view(batch_size, self.channels, self.seq_length)
            self.model.train()
            return x_final

    
    def forward(self, x1, graphEmb=None, timeEmb=None):
        """Forward pass (training) - returns loss and predictions for compatibility"""
        batch_size = x1.shape[0]
        device = x1.device
        # calculate loss and predictions
        x0 = torch.randn_like(x1)
        y0 = torch.arange(len(x0))
        y1 = torch.arange(len(x1))
        x0,x1,y0,y1=self.ot_sampler.sample_plan_with_labels(x0,x1,y0,y1)
        graphEmb=graphEmb[y1]
        timeEmb=timeEmb[y1]
        t=torch.rand(batch_size,device=device).type_as(x0)
        #print(f"x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        #print(f"t shape: {t.shape}, xt shape: {xt.shape}, ut shape: {ut.shape}")
        # normalize t to [B] shape to avoid downstream time embedding errors
        xt = sample_conditional_pt(x0, x1, t, sigma=0.01)
        ut = compute_conditional_vector_field(x0, x1)
        vt=self.model(xt,t, graphEmb, timeEmb)
        loss=torch.mean((vt-ut)**2)  
        return loss
    
    
