'''
Extract parameters from the trained model
'''
import sys
import os
sys.path.append('../../Pretrain')
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from datasets import *
from Models import *
from NetSet import *
from torch.utils.data.sampler import *
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils import *
##########

dataname = "hill"
# model_name='v_STGCN5'
model_name = "v_STGCN5"
num_environments = 4
ifdisturbe=True
if ifdisturbe== False:
    num_environments=1
##########
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract parameters from trained models")
    parser.add_argument("--dataname", default=dataname)
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--num-environments", type=int, default=num_environments)
    args = parser.parse_args()
    dataname = args.dataname
    model_name = args.model_name
    num_environments = args.num_environments

    if ifdisturbe == False:
        num_environments = 1

    with open('../config.yaml') as f:
        config = yaml.full_load(f)
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    if model_name == 'v_STGCN5':
        model = StgnnSet(data_args, task_args, model_args, model='v_STGCN5')  # v_STGCN5
    elif model_name == 'v_GWN':
        model = StgnnSet(data_args, task_args, model_args, model='v_GWN')  # v_GWN
    else:
        print("model_name is not supported")
        exit()
    total_params_length = 0
    for param in model.parameters():
        total_params_length += param.numel()
    nan_tensor_placeholder = torch.full((1, total_params_length), float('nan'), dtype=torch.float32)

    ifFirstRegion = True
    task_params_list = [] 
    allRegionTensor = None 
    total_count = 0
    left_count = 0
    print(f"will handle {num_environments} environments...")

    for env_number in range(0, num_environments):
        print(f"\n--- handling environment: {env_number}/{num_environments} ---")
        
        graph_set = np.load(f"../graph_generator/output/{dataname}/parameter_list.npy")
        env_model_count = 0
        if dataname=='hill':
            graph_set=graph_set[0:400]
        for (ids, graph_index) in tqdm(enumerate(graph_set), desc=f"env {env_number}", total=len(graph_set)):
            finaltensor = None
            if model_name == 'v_STGCN5':
                flag='stgcn'
            elif model_name == 'v_GWN':
                flag='gwn'
            else:
                print("model_name is not supported")
            if ifdisturbe:
                model_path = f'../Param/{flag}/{dataname}/disturbe/model_graph_{graph_index}_env_{env_number}.pth'
            else:
                model_path = f'../Param/{flag}/{dataname}/disturbe_no_pretrain/model_graph_{graph_index}_env_{env_number}.pth'
            if os.path.exists(model_path):
                total_count += 1
                env_model_count += 1
                file = torch.load(model_path, map_location=torch.device('cpu'))
                model.load_state_dict(file['model_state_dict'])
                allparams = list(model.named_parameters())
                iffirst = True
                for singleparams in allparams:
                    astensor = singleparams[1].clone().detach()
                    tensor1D = astensor.flatten().unsqueeze(0)
                    if iffirst:
                        finaltensor = tensor1D
                        iffirst = False
                    else:
                        finaltensor = torch.cat((finaltensor, tensor1D), dim=1)
            else:
                print(f"model_path {model_path} does not exist")
                left_count += 1
                finaltensor = nan_tensor_placeholder

            task_params_list.append(graph_index.tolist())
            
            if ifFirstRegion:
                allRegionTensor = finaltensor
                ifFirstRegion = False
            else:
                allRegionTensor = torch.cat((allRegionTensor, finaltensor), dim=0)

        print(f"environment {env_number} processed, successfully loaded {env_model_count} models.")
    if ifdisturbe:
        output_filename_prefix = f'ModelParams_{flag}_{dataname}_final'
    else:
        output_filename_prefix = f'ModelParams_{flag}_{dataname}_undisturbe'
    np.save(f'{output_filename_prefix}.npy', allRegionTensor.cpu().numpy())
    print(f"\n all weights parameters have been merged and saved to {output_filename_prefix}.npy")
    print(f"   - shape: {allRegionTensor.shape}")
    print(f"   - successfully loaded {total_count} model files.")
    print(f"   - failed to load {left_count} model files.")
    task_params_array = np.array(task_params_list)
    if ifdisturbe:
        np.save(f'TaskParams_{flag}_{dataname}_final.npy', task_params_array)
    else:
        np.save(f'TaskParams_{flag}_{dataname}_undisturbe.npy', task_params_array)
    print(f"corresponding task parameters have been saved, shape: {task_params_array.shape}")
