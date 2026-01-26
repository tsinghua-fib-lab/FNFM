import logging
import os
import sys
import zipfile
import random
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def metric_func(pred, y, times): 
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)
    
    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        return np.mean(diff / y)

    for i in range(times):
        y_i = y[:,i,:]
        pred_i = pred[:,i,:]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        # MAPE = cal_MAPE(pred_i, y_i)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        # result['MAPE'][i] += MAPE
    return result

def output_metrics_to_csv(total_RMSE, total_MAE, total_MAPE, filename='train_k_metrics_output.csv'):
    """
    Save RMSE, MAE, MAPE data to a CSV file.

    Args:
        total_RMSE (list): List of RMSE values.
        total_MAE (list): List of MAE values.
        total_MAPE (list): List of MAPE values.
        filename (str): The name of the CSV file to save, default is 'metrics_output.csv'.
    """
    
    # Ensure all lists have the same length, otherwise an error will be reported or the behavior will be abnormal
    if not (len(total_RMSE) == len(total_MAE) == len(total_MAPE)):
        raise ValueError("The lengths of the input lists (total_RMSE, total_MAE, total_MAPE) must be the same.")

    # Multiply MAPE by 100 to express it as a percentage
    scaled_total_MAPE = [val * 100 for val in total_MAPE]

    # Build the data, each item is a row of the DataFrame
    # Original Excel structure:
    # Row 0: MAE
    # Row 1: MAPE * 100
    # Row 2: RMSE
    data = [
        total_MAE,
        scaled_total_MAPE,
        total_RMSE
    ]

    # Define row index (labels)
    index_labels = ['MAE', 'MAPE (%)', 'RMSE']

    # Define column names (optional, if the original data has a specific meaning, it can be named 'Model 1', 'Fold 1', etc.)
    # Assuming the data has 6 columns (according to the range(6) in the original code), 'Result 1', 'Result 2', etc. are generated here
    column_labels = [f'Result {i+1}' for i in range(len(total_MAE))]

    # Create DataFrame
    df = pd.DataFrame(data, index=index_labels, columns=column_labels)

    # Write the DataFrame to a CSV file
    # index=True will also write the row index of the DataFrame (i.e., 'MAE', 'MAPE (%)', 'RMSE') to the first column of the CSV
    df.to_csv(filename, index=True) 
    
    print(f"Data has been successfully saved to {filename}")

def result_print(result, logger, info_name='Evaluate'):
    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']

    output_metrics_to_csv(total_RMSE, total_MAE, total_MAPE)

    logger.info("========== {} results ==========".format(info_name))
    logger.info(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
    # logger.info("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
    logger.info("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
    logger.info("---------------------------------------")

    if info_name == 'Best':
        logger.info("========== Best results ==========")
        logger.info(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        # logger.info("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        logger.info("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        logger.info("---------------------------------------")


def load_data(dataset_name, stage):
    print("INFO: load {} data @ {} stage".format(dataset_name, stage))

    A = np.load("data/" + dataset_name + "/matrix.npy")
    A = get_normalized_adj(A)
    A = torch.from_numpy(A)
    X = np.load("data/" + dataset_name + "/dataset.npy")
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    # train: 70%, validation: 10%, test: 20%
    # source: 100%, target_1day: 288, target_3day: 288*3, target_1week: 288*7
    if stage == 'train':
        X = X[:, :, :int(X.shape[2]*0.7)]
    elif stage == 'validation':
        X = X[:, :, int(X.shape[2]*0.7):int(X.shape[2]*0.8)]
    elif stage == 'test':
        X = X[:, :, int(X.shape[2]*0.8):]
    elif stage == 'source':
        X = X
    elif stage == 'target_1day':
        X = X[:, :, :288]
    elif stage == 'target_3day':
        X = X[:, :, :288*3]
    elif stage == 'target_1week':
        X = X[:, :, :288*7]
    else:
        print("Error: unsupported data stage")

    print("INFO: A shape is {}, X shape is {}, means = {}, stds = {}".format(A.shape, X.shape, means, stds))

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def generate_train_val_dataset(X, 
                               num_timesteps_input, 
                               num_timesteps_output, 
                               means,
                               stds,
                               val_ratio=0.1, 
                               random_seed=42):
    """
    Generate training and validation sets from time series data.

    This function first generates all possible time window samples, and then randomly
    divides them into training and validation sets according to the ratio.

    Parameters:
    ----------
    X: np.ndarray
        Input time series data with shape (num_nodes, num_features, num_timesteps).
    num_timesteps_input: int
        Length of the input sequence (historical steps for the sliding window).
    num_timesteps_output: int
        Length of the output sequence (future steps to be predicted).
    val_ratio: float, optional
        Proportion of the validation set, default is 0.1.
    random_seed: int, optional
        Random seed to ensure consistent splitting results each time, default is 42.

    Returns:
    -------
    tuple: A tuple containing four PyTorch tensors
        - train_features (torch.Tensor): Training set input features
        - train_targets (torch.Tensor): Training set targets
        - val_features (torch.Tensor): Validation set input features
        - val_targets (torch.Tensor): Validation set targets
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 1. Generate all samples
    # Calculate all possible starting points
    total_window_size = num_timesteps_input + num_timesteps_output
    indices = [(i, i + total_window_size) for i in 
               range(X.shape[2] - total_window_size + 1)]

    features, target = [], []
    for i, j in indices:
        # Input features: (num_nodes, num_features, num_timesteps_input) -> (num_nodes, num_timesteps_input, num_features)
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1))
        )
        # Target: Only predict the first feature (e.g., traffic flow)
        # Shape is (num_nodes, num_timesteps_output)
        target.append(X[:, :, i + num_timesteps_input: j])

    # Convert lists to numpy arrays, then to torch tensors
    all_features = torch.from_numpy(np.array(features)).float()
    all_targets = torch.from_numpy(np.array(target)).float()

    # 2. Create and shuffle indices
    num_samples = all_features.shape[0]
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)

    # 3. Split indices
    val_size = int(num_samples * val_ratio)
    train_size = num_samples - val_size
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    # 4. Split data
    train_features = all_features[train_indices]
    train_targets = all_targets[train_indices]

    val_features = all_features[val_indices]
    val_targets = all_targets[val_indices]
    
    print(f"Total samples: {num_samples}")
    print(f"Train samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")

    return train_features, train_targets, val_features, val_targets

def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds):

    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)] 
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))



def setup_logger(mode, DATA, index, model, aftername):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    log_dir = "Logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # Create directory
    f_handler = logging.FileHandler("Logs/{}_logs_{}_{}_{}.log".format(mode, DATA, model, aftername))

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger,"Logs/{}_logs_{}_{}_{}.log".format(mode, DATA, model, aftername)


def rescale(data, mean ,std):
    return data*std + mean


