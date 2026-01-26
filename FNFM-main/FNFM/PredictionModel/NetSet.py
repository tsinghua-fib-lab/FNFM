import sys
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from PredictionModel.Models.meta_stgcn import *
from PredictionModel.Models.meta_gwn import *
from PredictionModel.utils import *
from PredictionModel.datasets import *
from PredictionModel.utils import *
from copy import deepcopy
from tqdm import tqdm
import scipy.sparse as sp
from PredictionModel.datasets import DynamicsDataset
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
try:
    import torch.func
except ImportError:
    print("Warning: torch.func not found. Differentiable evaluation might not work.")
use_x2=False
def asym_adj(adj):
    adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

class StgnnSet(nn.Module): 
    """
    MAML-based Few-shot learning architecture for STGNN
    """
    def node_eval(self, node_index, test_dataloader, logger, test_dataset):
        with torch.no_grad():
            self.model = torch.load('Param/Task3_1/{}/task3_{}.pt' .format(test_dataset, node_index))
            test_start = time.time()  
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):

                A_wave = A_wave[0][node_index,:].unsqueeze(0)
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave.float(), A_wave.float().t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave.float())
                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()
            result_print(result, logger,info_name='Evaluate')
            logger.info("[Test] testing time is {}".format(test_end-test_start))
            return outputs, y_label
    
    def taskEval(self, test_dataloader, logger):
        self.model = torch.load('Param/task1.pt')
        with torch.no_grad():
            test_start = time.time()  
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave.float(), A_wave.float().t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave.float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Target Test] testing time is {}".format(test_end-test_start))

    
    def evalparams(self, param, args,graph_num, basemodel,if_predict=True,save_output=False,target_days=50,if_adjust=False,if_freeze=False,name='', path_config=None,graph=None, evaluate_all_batches=True):
        device = param.device
        param=param.reshape(-1)
        if graph_num is None:
            raise ValueError("graph_num must be specified")
        '''eval the sample from diffusion '''
        if basemodel == 'v_STGCN5':
            indexstart = [0, 256, 512, 768, 1024, 2048, 3072, 4096, 8192, 12288, 16384, 16960]
            shapes = [(32, 8), (32, 2, 1, 4), (32, 2, 1, 4), (32, 2, 1, 4), (32, 8, 1, 4), 
                    (32, 8, 1, 4), (32, 8, 1, 4), (32, 32, 1, 4), (32, 32, 1, 4), 
                    (32, 32, 1, 4), (6, 96)]
        elif basemodel == 'v_GWN':
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
        for key in self.model.state_dict().keys():
            if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key):
                pa = param_prepare[indexstart[index]:indexstart[index+1]]
                pa = torch.reshape(pa, shapes[index])                
                self.model.state_dict()[key].copy_(pa)               
                index = index+1
        self.model.eval()
        # freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # --- path management ---
        if path_config:
            #print("Using the path in `path_config` for evaluation.")
            graph_set_path = path_config['graph_set_path']
            adj_matrix_path = path_config['adjacency_matrix_path']
            trajectories_dir = path_config['trajectories_dir']
        else:
            print("Warning: `path_config` not provided")
            exit(0)
        graph_set = np.load(graph_set_path)
        graph_index=graph_set[graph_num]
        if path_config['pathname'] == 'hill' or path_config['pathname'] == 'HillDisturbe':
            A_wave = np.load(adj_matrix_path)
            trajectory_file = os.path.join(trajectories_dir, f"undisturbed_trajectories_a{graph_index[0]}_h{graph_index[1]}_B{graph_index[2]}.csv")
        elif path_config["pathname"] in ['euroad','twitter','trust','collab','fhn','fhn2']:
            R_val, B_val = graph_index[0], graph_index[1]
            A_wave = np.load(adj_matrix_path)
            trajectory_file = os.path.join(trajectories_dir, f"undisturbed_trajectories_R_{R_val}_B_{B_val}.csv")
        else:
            raise ValueError(f"Unsupported path name: {path_config['pathname']}")
        X_dataset = pd.read_csv(trajectory_file)
        X_sorted = X_dataset.sort_values(by=[X_dataset.columns[0], X_dataset.columns[1]])
        state_col = X_sorted.columns[2]
        X_pivot = X_sorted.pivot(index=X_sorted.columns[0], columns=X_sorted.columns[1], values=state_col)
        X=X_pivot.values[:, :, np.newaxis]
        if path_config["pathname"] =='fhn2' and use_x2:
            trajectory_file_x2 = os.path.join(trajectories_dir, f"undisturbed_trajectories_R_{R_val}_B_{B_val}_x2.csv")
            X_dataset_x2 = pd.read_csv(trajectory_file_x2)
            X_sorted_x2 = X_dataset_x2.sort_values(by=[X_dataset_x2.columns[0], X_dataset_x2.columns[1]])
            state_col_x2 = X_sorted_x2.columns[2]
            X_pivot_x2 = X_sorted_x2.pivot(index=X_sorted_x2.columns[0], columns=X_sorted_x2.columns[1], values=state_col_x2)
            X_x2 = X_pivot_x2.values[:, :, np.newaxis]
            X=np.concatenate((X, X_x2), axis=2)
        metadatapoint=DynamicsDataset(X,A_wave,self.task_args,target_days)
        train_dataset,test_dataset,val_dataset,A_wave=metadatapoint.get_dataset()
        # create data loader
        batch_size=self.task_args['batch_size']
        train_loader = DataLoader(
            train_dataset, 
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
      
        train_meanstd = train_dataset.get_normalize_params()
        test_meanstd = test_dataset.get_normalize_params()
        if(len(train_loader)==0):
            print('train_loader is empty')

        all_outputs = []
        all_labels = []
        for step, (data) in enumerate(test_loader):
            data, A_wave = data.to(device), A_wave.to(device)
            if basemodel == 'v_GWN':
                adj_mx = [A_wave.float(), A_wave.float().t()]
                out = self.model(data, adj_mx)
            else:
                out, meta_graph = self.model(data, A_wave.float())
            mean_tensor = torch.tensor(train_meanstd[0]).to(device)
            std_tensor = torch.tensor(train_meanstd[1]).to(device)
            # adjust dimension to match data shape
            if len(mean_tensor.shape) == 1:
                mean_tensor = mean_tensor.reshape(1, -1, 1)  # [1, feature_dim, 1]
            if len(std_tensor.shape) == 1:
                std_tensor = std_tensor.reshape(1, -1, 1)  # [1, feature_dim, 1]
            outputs_before_fineturn = out*std_tensor+mean_tensor
            y_label_before_fineturn = data.y*std_tensor+mean_tensor
            outputs_before_fineturn = outputs_before_fineturn[:, :1, :]
            y_label_before_fineturn = y_label_before_fineturn[:, :1, :]
            
            all_outputs.append(outputs_before_fineturn)
            all_labels.append(y_label_before_fineturn)
        
        outputs_before_fineturn = torch.cat(all_outputs, dim=0)
        y_label_before_fineturn = torch.cat(all_labels, dim=0)
        return outputs_before_fineturn, y_label_before_fineturn
    
    def __init__(self, data_args, task_args, model_args, model_name='v_STGCN5', node_num=207):
        super(StgnnSet, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.adapt_epochs= model_args['adapt_epochs']
        self.update_lr = model_args['update_lr']
        self.meta_lr = model_args['meta_lr']
        self.update_step = model_args['update_step']
        self.update_step_test = model_args['update_step_test']
        self.task_num = task_args['task_num']
        self.model_name = model_name


        self.loss_lambda = model_args['loss_lambda']
        # print("loss_lambda = ", self.loss_lambda)
        if model_name == 'v_STGCN5':  # STGCN 5.0 which we choose
            self.model = STGCN_NonBias(model_args, task_args)
        elif model_name == 'v_GWN':
            self.model = v_GWN()  
        self.meta_optim = optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.loss_criterion = nn.MSELoss()
        
    def graph_reconstruction_loss(self, meta_graph, adj_graph):   
        adj_graph = adj_graph.unsqueeze(0).float()
        for i in range(meta_graph.shape[0]):
            if i == 0:
                matrix = adj_graph
            else:
                matrix = torch.cat((matrix, adj_graph), 0)
        criteria = nn.MSELoss()
        loss = criteria(meta_graph, matrix.float())
        return loss
      
    def calculate_loss(self, out, y, meta_graph, matrix, stage='target', graph_loss=True, loss_lambda=1):
        if loss_lambda == 0:
            loss = self.loss_criterion(out, y)
        if graph_loss:
            if stage == 'source' or stage == 'target_maml':
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.graph_reconstruction_loss(meta_graph, matrix)
            else:
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.loss_criterion(meta_graph, matrix.float())
            loss = loss_predict + loss_lambda * loss_reconsturct
        else:
            loss = self.loss_criterion(out, y)

        return loss

    def forward(self, data, matrix):
        if self.model_name == 'v_GWN':
            adj_mx = [matrix.float(), matrix.float().t()]
            out = self.model(data, adj_mx)
            meta_graph = matrix
        else:
            out, meta_graph = self.model(data, matrix.float())
        return out, meta_graph

    def finetuning(self, target_dataloader, test_dataloader, target_epochs,logger):
        """
        finetunning stage in MAML
        """
        # Create a model copy: reinstantiate and load weights to avoid deepcopy errors
        device = next(self.model.parameters()).device
        maml_model = self.model.__class__(self.model_args, self.task_args).to(device)
        maml_model.load_state_dict(self.model.state_dict())
        optimizer = optim.Adam(maml_model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        min_MAE = 10000000
        best_result = ''
        best_meta_graph = -1

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            maml_model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave.float(), A_wave.float().t()]
                    out, meta_graph = maml_model(data, adj_mx)
                else:
                    out, meta_graph = maml_model(data, A_wave.float())
                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                    loss = self.loss_criterion(out, data.y)
                else:
                    # loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', graph_loss=False)
                    loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)
            end_time = time.time()
            if epoch % 5 == 0:
                logger.info("[Target Fine-tune] epoch #{}/{}: loss is {}, fine-tuning time is {}".format(epoch+1, target_epochs, avg_train_loss, end_time-start_time))

            # Use TensorBoard to record the training loss during the finetuning phase

        with torch.no_grad():
            test_start = time.time()  
            maml_model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave.float(), A_wave.float().t()]
                    out, meta_graph = maml_model(data, adj_mx)
                else:
                    out, meta_graph = maml_model(data, A_wave.float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Target Test] testing time is {}".format(test_end-test_start))

    def taskTrain(self, taskmode, target_dataloader, test_dataloader, target_epochs,logger):

        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr)
        min_MAE = 10000000
        best_result = ''
        best_meta_graph = -1

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            self.model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave.float(), A_wave.float().t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave.float())

                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN', 'Node_STGCN']:
                    loss = self.loss_criterion(out, data.y)
                else:
                    # loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', graph_loss=False)
                    loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)
            end_time = time.time()
            if epoch % 5 == 0:
                logger.info("[Target Fine-tune] epoch #{}/{}: loss is {}, fine-tuning time is {}".format(epoch+1, target_epochs, avg_train_loss, end_time-start_time))
                if taskmode=='task3':
                    torch.save(self.model,'Param/{}_inside.pt'.format(taskmode))
            
            # Use TensorBoard to record training loss, distinguishing between different taskmodes
        
        torch.save(self.model,'Param/{}_inside.pt'.format(taskmode))

        with torch.no_grad():
            test_start = time.time()  
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave.float(), A_wave.float().t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave.float())

    def finetune_last_layer(self, adjust_loader, test_loader, A_wave, finetune_epochs, finetune_lr,target_days):
        """
        Fine-tune the last layer of the model using adjust_loader.
        """
        last_layer_name = 'fully' # <--- Replace with your layer name here
        # !!! Important: You need to replace 'final_fc' with the actual name of the last layer in your model !!!
        try:
            last_layer = getattr(self.model, last_layer_name)
        except AttributeError:
            print(f"Error: Could not find the layer named '{last_layer_name}' in the model.")
            print("Available model components:", [name for name, _ in self.model.named_children()])
            return
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the parameters of the last layer
        for param in last_layer.parameters():
            param.requires_grad = True
            print(f"Unfreezing parameter in {last_layer_name}: {param.shape}")
        # Confirm which parameters are trainable
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # --- 2. set optimizer ---
        optimizer = optim.Adam(trainable_params, lr=finetune_lr, weight_decay=1e-4) # Use a smaller weight_decay
        print(f"Optimizer set up for fine-tuning with lr={finetune_lr}")
        import time # Ensure time is imported
        from torch.utils.tensorboard import SummaryWriter # Ensure SummaryWriter is imported
        log_dir = f"runs/FineTune_TargetDays_{target_days}_{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging initialized. Log directory: {log_dir}")

        # --- 3. Fine-tuning loop ---
        self.model.train() # Set to training mode
        for epoch in range(finetune_epochs):
            epoch_losses = []
            start_time = time.time()
            for step, (data) in enumerate(adjust_loader):
                data = data.cuda()
                data.node_num = data.node_num[0]
                out, meta_graph = self.model(data, A_wave.cuda()) 
                loss = self.loss_criterion(out, data.y)
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient clipping to prevent exploding gradients
                # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                epoch_losses.append(loss.item())
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            end_time = time.time()
            if epoch % 1 == 0:
                epoch_rmse = []
                for step, (data) in enumerate(test_loader):
                    data = data.cuda()
                    data.node_num = data.node_num[0]
                    out, meta_graph = self.model(data, A_wave.cuda()) 
                    rmse = torch.sqrt(self.loss_criterion(out, data.y))
                    epoch_rmse.append(rmse.item()) # Get scalar value
                avg_rmse = sum(epoch_rmse) / len(epoch_rmse) if epoch_rmse else 0
                if writer:
                    writer.add_scalar(f'FineTune/Loss', avg_epoch_loss, epoch)
                    writer.add_scalar(f'FineTune/RMSE', avg_rmse, epoch)
                print(f"[Fine-tune] Epoch {epoch+1}/{finetune_epochs}: Avg Loss = {avg_epoch_loss:.6f}, Time = {end_time - start_time:.2f}s")
        self.model.eval() # Set back to evaluation mode after fine-tuning is finished
        if writer:
            writer.close()
            print("TensorBoard writer closed.")
        print("Last layer fine-tuning finished.")

        # Optional: After fine-tuning, you can freeze the last layer again or unfreeze all layers, depending on subsequent operations
        # for param in last_layer.parameters():
        #     param.requires_grad = False
        # Or
        # for param in self.model.parameters():
        #     param.requires_grad = True