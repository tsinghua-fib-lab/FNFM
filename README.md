# FNFM: Zero-Shot Forecasting of Network Dynamics through Weight Flow Matching

## Environment Setup

Please ensure you have Python 3.9 installed. Use the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

Please follow the steps below to run the entire project workflow.

### 1. Data Simulation

First, you need to generate simulated graph data. Navigate to the `FNFM-main/Pretrain/graph_generator` directory and run the `graph_simulate.py` script.

```bash
bash simulate.sh
```

or

```python
 cd FNFM-main/Pretrain/graph_generator
CUDA_VISIBLE_DEVICES=0 python graph_simulate.py --mode collab
```

This script will automatically change the directory and execute the graph simulation program.

### 2. Model Pre-training

The pre-training process consists of two main stages. First, we train a general model using the data from all observed training environments. Then, we use the weights of this general model as a starting point to train a specialized model for each individual observed training environments.

#### Stage 1: Train a General Model on All Environments

Run the following script to train a single expert model on all the combined data from observed training environments. This creates a foundational model with broad knowledge.

```bash
bash train_stgcn_for_all.sh
```

or

```python
cd FNFM-main/Pretrain
CUDA_VISIBLE_DEVICES=0 python train_stgcn_all.py --dataname collab --model v_STGCN5 --epochs 80
```

#### Stage 2: Train a Specific Model for Each Environment

Next, use the pre-trained general model from Stage 1 to initialize the weights for training a separate, specialized model for each environment.

```bash
bash train_stgcn_for_each.sh
```

or

```python
cd FNFM-main/Pretrain
CUDA_VISIBLE_DEVICES=0 python main_disturbe.py --graph_range ALL --model v_STGCN5 --dataname collab --epochs 20
```

### 3. Model Parameter Conversion

After pre-training is complete, the model parameters need to be converted into a tensor format for use by the main model.

```bash
bash model2tensor.sh
```

or

```python
cd FNFM-main/Pretrain/PrepareParams
python model2tensor.py --dataname collab --model-name v_STGCN5 --num-environments 5
```

This script will run `FNFM-main/Pretrain/PrepareParams/model2tensor.py` to perform the conversion.

### 4. Main Model Training

Finally, train the FNFM model.

```bash
sh train_fnfm.sh
```

or

```python
cd FNFM-main/FNFM
CUDA_VISIBLE_DEVICES=0 python 1Dmain_torchcfm.py --expIndex 1 --basemodel v_STGCN5 --targetDataset collab --mode AE_CFM --denoise='cfmTransformer' --train_objective='cfm' --ae_arch layer_transformer --cfm_epochs 3000 --ae_epochs 1000
```

This script will start `1Dmain_torchcfm.py`, which loads the converted model parameters and trains the final forecasting model, generating the results for unseen testing environments.

## Script Descriptions

- `simulate.sh`: Runs the graph data simulation.
- `train_stgcn_for_all.sh`: Pre-trains a single expert model using data from all environments.
- `train_stgcn_for_each.sh`: Fine-tunes a specific model for each environment, using the general model for initialization.
- `model2tensor.sh`: Converts the pre-trained model parameters into a tensor file.
- `train_cfm.sh`: Trains the main FNFM model.

You can modify the parameters in these scripts as needed, such as `CUDA_VISIBLE_DEVICES` to specify the GPU to use, `dataname` to choose the dataset, or other model training hyperparameters.
