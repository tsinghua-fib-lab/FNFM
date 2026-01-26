import os
import sys
import argparse
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from scipy.integrate import solve_ivp, odeint as scipy_odeint
import pandas as pd
import torch
from torchdiffeq import odeint_event, odeint
from dynamics import (
    ConstructDynamicsModel,
    EcologicalModel,
    EpidemicModel,
    SIRPropagation,
    fhn_limit_cycle_dynamics,
)
import itertools


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from limit_cycle_scan import (
    SimulationConfig,
    build_network as fhn_build_network,
    initial_conditions as fhn_initial_conditions,
    compute_limit_cycle_metrics,
    ensure_output_dirs as scan_ensure_output_dirs,
    generate_phase_plot,
    plot_heatmap,
)
class DisplayModule:
    def __init__(self, output_dir, A):

        self.output_dir = output_dir
        self.A = A
        self.G = nx.from_numpy_array(self.A)
        self.node_degrees = [d for _, d in self.G.degree()]

    def plot_undisturbed_trajectories(self, R, B):
        # Load trajectory data
        trajectory_file = os.path.join(self.output_dir, f"undisturbed_trajectories_R{R}_B{B}.csv")
        if not os.path.exists(trajectory_file):
            print(f"Trajectory file not found: {trajectory_file}")
            return
        df = pd.read_csv(trajectory_file)

        # Select representative nodes: based on minimum, maximum, and quantiles of degrees
        degrees = np.array(self.node_degrees)
        selected_nodes = [
            np.argmin(degrees),  # Node with minimum degree
            np.argmax(degrees),  # Node with maximum degree
            int(np.quantile(np.arange(len(degrees)), 0.25)),  # 25th percentile
            int(np.quantile(np.arange(len(degrees)), 0.5)),   # Median
            int(np.quantile(np.arange(len(degrees)), 0.75))   # 75th percentile
        ]
        selected_nodes = list(set(selected_nodes))  # Remove duplicate nodes

        # Plot chart
        plt.figure(figsize=(12, 8))
        for node in selected_nodes:
            node_data = df[df['node_id'] == node]
            plt.plot(node_data['time'], node_data['state'], label=f'Node {node} (Degree={degrees[node]})')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title(f'Undisturbed Trajectories (R={R}, B={B})')
        plt.legend()
        plt.grid(True)
        plot_file = os.path.join(self.output_dir, f'undisturbed_trajectories_plot_R{R}_B{B}.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot saved to {plot_file}")
class NetworkPerturbationSimulator:
    def __init__(self, config=None):
        """
        Initializes the network perturbation simulator.
        Args:
            config: Configuration parameters. If None, default configuration is used.
        """
        # Default configuration
        self.config = {
            'node_num': 200,  # Number of nodes
            'edge_m': 4,      # Number of edges to attach from a new node to existing nodes
            'seed': 42,       # Random seed
            'output_dir': 'output/perturbation',  # Output directory
            'model_type': 'ConstructDynamicsModel',  # Dynamics model type
            'dynamics': {
                'total_t': 60.0,     # Total simulation time
                'sim_dt': 0.01,      # Simulation time step
                'dt': 0.1,           # Sampling time step
                'B': 1.0,            # Decay coefficient
                'a': 0.5,            # Decay exponent
                'h': 1/3,            # Saturation exponent / Hill coefficient
                'w': 1.0,            # Node weight
                # EcologicalModel parameters
                'K': 1.0,            # Carrying capacity
                'D': 0.1,            # Diffusion coefficient
                'E': 0.1,            # Self-inhibition coefficient
                'H': 0.1,            # Interaction coefficient
                'p1': 1.0,           # Density-dependence exponent
            },
            'perturbation': {
                'time_unperturbed': 40.0,  # Simulation time without perturbation
                'time_perturbed':50.0,    # Simulation time after perturbation
                'perturb_percent': 0.1,    # Perturbation percentage
                'recovery_threshold': 0.7,  # Recovery threshold
                'num_nodes_to_perturb': 20  # Number of nodes to perturb
            }
        }
        
        # Update default configuration if a configuration is provided
        if config:
            self.config.update(config)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        # Set random seed
        np.random.seed(self.config['seed'])
        self.A = None  # Adjacency matrix
        self.G = None  # NetworkX graph object
        self.model = None  # Dynamics model
        self.dist_path = None  # Distances between nodes
        self.node_degrees = None  # Node degrees
        # Add logging matrices
        self.steady_state_x = None  # Steady-state xi values
        self.G_matrix = None  # Response matrix Gij
        self.tau_values = None  # Recovery times τi
        # Add trajectory storage
        self.undisturbed_trajectories = []  # Store trajectories without perturbation
        self.disturbed_trajectories = {}    # Store trajectories after perturbation, organized by perturbed node index
        self.steady_state_reached_time = None # To record the time when steady state is reached

    def load_or_generate_network(self,dataname, adjacency_matrix=None, network_file=None):
        """
        Loads or generates a network.
        Args:
            adjacency_matrix: Adjacency matrix. If None, loads from a file or generates a new network.
            network_file: Path to the network file. If None, generates a new network.
        """
        self.A = adjacency_matrix
        #print(f"Using provided adjacency matrix with shape: {self.A.shape}")
        self.G = nx.from_numpy_array(self.A)
        # Calculate shortest path lengths between nodes
        # print("Calculating shortest path lengths between nodes...")
        # self.dist_path = dict(nx.all_pairs_shortest_path_length(self.G))
        # Calculate node degrees
        self.node_degrees = np.sum(self.A, axis=1)
        self.save_network(dataname)
        return self.G, self.A
    
    def initialize_model(self):
        """
        Initializes the dynamics model.
        """
        if self.A is None:
            self.load_or_generate_network()       
        dynamics_config = self.config['dynamics']
        model_type = self.config.get('model_type')
        # Create an arguments object
        class Args(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__ = self
                self.dynamics = model_type    
        args = Args({
            model_type: type(model_type, (), dynamics_config)()
        })
        
        # Initialize different dynamics models based on the model type
        if model_type == 'EcologicalModel':
            self.model = EcologicalModel(args, self.A)
            print("EcologicalModel initialized.")
        elif model_type == 'EpidemicModel':
            self.model = EpidemicModel(args, self.A)
            print("EpidemicModel initialized.")
        elif model_type == 'ConstructDynamicsModel':
            self.model = ConstructDynamicsModel(self.A, self.config)
            print("ConstructDynamicsModel initialized.")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.model
    
    def simulate_unperturbed(self, force_recalculate=True):
        if self.model is None:
            self.initialize_model()
        
        node_num = self.A.shape[0]
        time_unperturbed = self.config['perturbation']['time_unperturbed']
        sim_dt = self.config['dynamics']['sim_dt']
        gradient_threshold = self.config['dynamics'].get('gradient_threshold', 1e-4) # Get threshold from configuration

        #print(f"Simulating unperturbed system to steady state, max duration: {time_unperturbed}, steady state threshold: {gradient_threshold:.1e}...")
        # Set initial state
        x0 = np.random.uniform(0, 0.2, size=node_num)
        # Safety check: constrain state to the [0, 1] interval, which is physically meaningful for epidemic models
        x_safe = np.clip(x0, 0, 1)
        # Set time points
        tspan = np.arange(0, time_unperturbed, sim_dt)
        
        # Use GPU acceleration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(f"Using device: {device}")
        x0_tensor = torch.tensor(x_safe, dtype=torch.float64).to(device)
        tspan_tensor = torch.tensor(tspan, dtype=torch.float64).to(device)
        
        # Clear/reset state
        self.undisturbed_trajectories = []
        self.steady_state_reached_time = None

        def ode_func(t, y):
            y_np = y.cpu().detach().numpy() if device.type == 'cuda' else y.detach().numpy()
            dydt_np = self.model.f(y_np, t.item())
            dydt = torch.tensor(dydt_np, dtype=torch.float64).to(device)

            # Steady-state detection: if time has not been recorded, check if steady state is reached
            if self.steady_state_reached_time is None:
                if torch.max(torch.abs(dydt)) < gradient_threshold:
                    self.steady_state_reached_time = t.item()

            # Store trajectory
            for node_idx in range(node_num):
                self.undisturbed_trajectories.append((t.item(), node_idx, y_np[node_idx]))

            # Display progress
            current_progress = t.item() / time_unperturbed * 100
            last_progress = (t.item() - sim_dt) / time_unperturbed * 100
            # if int(current_progress) % 10 == 0 and int(current_progress) > int(last_progress):
            #     print(f"\rProgress: {current_progress:.1f}% complete", end="", flush=True)
            
            return dydt

        # Solve ODE using torchdiffeq's odeint function
        sol_tensor = odeint(ode_func, x0_tensor, tspan_tensor, method='euler')
        #print(f"\rProgress: 100.0% complete", flush=True)
        
        # Report steady-state detection results
        # if self.steady_state_reached_time is not None:
        #     print(f"Steady state reached at t={self.steady_state_reached_time:.2f}.")
        # else:
        #     print(f"Steady state not reached within {time_unperturbed:.2f}s.")

        # Convert result back to a numpy array
        sol = sol_tensor.cpu().detach().numpy()
        # Get the final state
        y_final = sol[-1]
        print("Unperturbed system simulation complete.")

        # Unconditionally save trajectory data
        self.save_trajectories("undisturbed")
        # Record pseudo-steady-state xi values (i.e., values at the end of the time window)
        self.steady_state_x = y_final.copy()
        
        return y_final
    
    def save_network(self,network_type):
        """
        Saves network information.
        """
        output_dir = self.config['output_dir']
        # Save adjacency matrix
        np.save(os.path.join(output_dir, network_type+'_matrix.npy'), self.A)
        # Save network statistics
        stats = {
            'node_count': self.G.number_of_nodes(),
            'edge_count': self.G.number_of_edges(),
            'average_degree': np.mean([d for _, d in self.G.degree()]),
            'clustering_coefficient': nx.average_clustering(self.G),
            'is_connected': nx.is_connected(self.G)
        }
        with open(os.path.join(output_dir, 'adjacency_matrix_stats.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")


    

    def save_trajectories(self, trajectory_type):
        """
        Saves trajectory data to a file.
        Args:
            trajectory_type: The type of trajectory, used for file naming.
        """
        output_dir = self.config['output_dir']
        if trajectory_type == "undisturbed":
            # Dynamically generate filename based on model type
            model_type = self.config.get('model_type', 'ConstructDynamicsModel')
            
            if model_type == 'EpidemicModel':
                R = self.config['dynamics']['R']
                B = self.config['dynamics']['B']
                trajectory_file = os.path.join(output_dir, f"undisturbed_trajectories_R{R}_B{B}.csv")
            elif model_type == 'EcologicalModel':
                K = self.config['dynamics']['K']
                B = self.config['dynamics']['B']
                trajectory_file = os.path.join(output_dir, f"undisturbed_trajectories_K_{K:.2f}_B_{B:.2f}.csv")
            else: # Default to ConstructDynamicsModel format
                trajectory_file = os.path.join(output_dir, f"undisturbed_trajectories_a{self.config['dynamics']['a']}_h{self.config['dynamics']['h']}_B{self.config['dynamics']['B']}.csv")
            
            with open(trajectory_file, 'w') as f:
                f.write("time,node_id,state\n")  # Write CSV header
                for time, node_id, state in self.undisturbed_trajectories:
                    f.write(f"{time},{node_id},{state:.16f}\n")  # Save floats with high precision
            #df=pd.read_csv(trajectory_file)
            #png_file=os.path.join(output_dir, f"undisturbed_trajectories_a{self.config['dynamics']['a']}_h{self.config['dynamics']['h']}_B{self.config['dynamics']['B']}.png")
            #self.visualize(df,png_file)
            print(f"Undisturbed trajectories saved to: {trajectory_file}")
            # Clear trajectory data to save memory
            self.undisturbed_trajectories = []
        
        elif trajectory_type.startswith("disturbed_node_"):
            # Extract node number
            node_num = int(trajectory_type.split("_")[-1])
            
            # Save perturbed trajectories
            trajectory_file = os.path.join(output_dir, f"disturbed_node_{node_num}_trajectories.csv")
            with open(trajectory_file, 'w') as f:
                f.write("time,node_id,state\n")  # Write CSV header
                for time, node_id, state in self.disturbed_trajectories[node_num]:
                    f.write(f"{time},{node_id},{state:.16f}\n")  # Save floats with high precision
            print(f"Trajectories after perturbing node {node_num} saved to: {trajectory_file}")

            # Clear trajectory data for this node to save memory
            self.disturbed_trajectories[node_num] = []
    def get_trajectory_data(self, trajectory_type):
        print("Attempting to load data:")
        """
        Gets trajectory data. If not in memory, attempts to load from file.
        Args:
            trajectory_type: The type of trajectory.
        Returns:
            trajectory_data: Trajectory data in the format (t, y).
        """
        if trajectory_type == "undisturbed":
            self.load_trajectories("undisturbed")
            # Convert trajectory data to (t, y) format
            df = pd.DataFrame(self.undisturbed_trajectories, columns=['time', 'node_id', 'state'])
            times = df['time'].unique()
            nodes = df['node_id'].unique()
            nodes.sort()
            
            t = times
            y = np.zeros((len(times), len(nodes)))
            
            for i, time in enumerate(times):
                time_data = df[df['time'] == time]
                for _, row in time_data.iterrows():
                    node_idx = np.where(nodes == row['node_id'])[0][0]
                    y[i, node_idx] = row['state']
            
            return t, y
        
        elif trajectory_type.startswith("disturbed_node_"):
            node_num = int(trajectory_type.split("_")[-1])
            
            if node_num not in self.disturbed_trajectories or not self.disturbed_trajectories[node_num]:
                self.load_trajectories(trajectory_type)
            
            if node_num not in self.disturbed_trajectories or not self.disturbed_trajectories[node_num]:
                print(f"Perturbed trajectory data for node {node_num} not found. Rerunning simulation may be required.")
                return None
            
            # Convert trajectory data to (t, y) format
            df = pd.DataFrame(self.disturbed_trajectories[node_num], columns=['time', 'node_id', 'state'])
            times = df['time'].unique()
            nodes = df['node_id'].unique()
            nodes.sort()
            
            t = times
            y = np.zeros((len(times), len(nodes)))
            
            for i, time in enumerate(times):
                time_data = df[df['time'] == time]
                for _, row in time_data.iterrows():
                    node_idx = np.where(nodes == row['node_id'])[0][0]
                    y[i, node_idx] = row['state']
            
            return t, y
        
        return None

def hill_propagation():
    """
    Main function for testing the network perturbation simulator.
    """
    # Create configuration
    config = {
        'node_num': 300,  # Reduce the number of nodes to speed up testing
        'edge_m': 5,
        'seed': 42,
        'output_dir': 'output/hill',
        'model_type': 'ConstructDynamicsModel',  # Use the ecological model
        'dynamics': {
            'total_t': 600.0,     # Total simulation time
            'sim_dt': 0.04,      # Simulation time step
            'dt': 0.04,           # Sampling time step
            # ConstructDynamicsModel parameters
            'B': 1.0,            # Decay coefficient
            'a': 0.5,            # Decay exponent
            'h': 1/3,            # Saturation exponent / Hill coefficient
            'w': 1.0,            # Node weight
        },
        'perturbation': {
            'time_unperturbed': 20.0,  # Reduce simulation time to speed up testing
            'time_perturbed': 400.0,
            'perturb_percent': 0.1,    # Add missing perturbation magnitude parameter
            'recovery_threshold': 0.8,  # Recovery threshold
            'num_nodes_to_perturb': 100  # Reduce the number of perturbed nodes
        }
    }
    try:
        A=np.load(os.path.join(config['output_dir'], 'adjacency_matrix.npy'))# single graph
    except:
        print(f"Generating BA scale-free network with {config['node_num']} nodes...")
        G = nx.barabasi_albert_graph(
            n=config['node_num'], 
            m=config['edge_m'],
            seed=42
        )
        A = nx.to_numpy_array(G)
        print(f"Network generation complete. Nodes: {A.shape[0]}, Edges: {G.number_of_edges()}") # Note: number of edges should be calculated from G
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
        np.save(os.path.join(config['output_dir'], f'adjacency_matrix.npy'), A)
    # Define parameter ranges: a from 0.3 to 0.7, h from 0.2 to 0.5, B from 0.8 to 1.2, with 3 decimal places
    a_list = np.round(np.linspace(0.5,0.6,10),3)
    h_list = np.round(np.linspace(0.33,2.0,10),3)
    B_list = np.round(np.linspace(0.2,0.9,4),3)
    # Generate a list of all possible parameter triplets
    parameter_list = list(itertools.product(a_list, h_list, B_list))
    np.save(os.path.join(config['output_dir'], 'parameter_list.npy'), parameter_list)
    # Simulate for each parameter combination
    with tqdm(
        total=len(parameter_list),
        desc="Hill (a,h,B) grid",
        unit="combo",
    ) as pbar:
        for (a, h, B) in parameter_list:
        # Modify parameters in the configuration
            config['dynamics']['a'] = a
            config['dynamics']['h'] = h
            config['dynamics']['B'] = B
            print(f"Running simulation with a={a}, h={h}, B={B}")
            try:
                # Initialize the simulator
                simulator = NetworkPerturbationSimulator(config)
                simulator.load_or_generate_network('hill',A)
                y_start = simulator.simulate_unperturbed(False)
                print("Simulation complete. Results saved to:", config['output_dir'])
                
            except Exception as e:
                print(f"Runtime error: {e}")
                import traceback
                traceback.print_exc()
            pbar.update(1)

class EpidemicModelTorch:
    def __init__(self, R, B, A_np, alpha=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.A = torch.from_numpy(A_np).float().to(self.device)
        self.A_no_diag = self.A.clone()
        self.A_no_diag.fill_diagonal_(0)
        self.R = R
        self.B = B
        self.alpha = alpha
        self.w = 1.0

    def f(self, t, x):
        x_safe = torch.clamp(x, 0, 1)
        recovery_term = -self.B * (x_safe ** self.alpha)
        infection_pressure = torch.matmul(self.A_no_diag, x_safe)
        spreading_term = self.R * (1 - x_safe) * infection_pressure
        dxdt = self.w * (recovery_term + spreading_term)

        # -- Add detailed debug prints --
        if torch.isnan(dxdt).any():
            print("\n--- [DEBUG] NaN Detected in Simulation ---")
            print(f"Time (t): {t.item()}")
            print(f"Parameters: R={self.R}, B={self.B}, alpha={self.alpha}")
            
            # Check recovery_term
            if torch.isnan(recovery_term).any():
                print("NaN found in 'recovery_term'")
                print(f"x_safe min: {torch.min(x_safe)}, max: {torch.max(x_safe)}")
            
            # Check spreading_term
            if torch.isnan(spreading_term).any():
                print("NaN found in 'spreading_term'")
                print(f"infection_pressure min: {torch.min(infection_pressure)}, max: {torch.max(infection_pressure)}")

            # Print detailed information for nodes causing NaNs
            nan_indices = torch.where(torch.isnan(dxdt))[0]
            print(f"Nodes with NaN output: {nan_indices.cpu().numpy()}")
            for idx in nan_indices[:5]: # Print information for at most 5 nodes
                print(f"  Node {idx}:")
                print(f"    x_safe: {x_safe[idx].item()}")
                print(f"    recovery_term: {recovery_term[idx].item()}")
                print(f"    infection_pressure: {infection_pressure[idx].item()}")
                print(f"    spreading_term: {spreading_term[idx].item()}")
            # -- End of debug prints --

        if torch.isnan(dxdt).any():
            print('nan during simulation!')
            dxdt[torch.isnan(dxdt)] = 0.0
        return dxdt

def save_detailed_trajectory_pandas(t_np, solution_np, R_param, B_param, output_dir):
    """
    Saves detailed node state trajectories to a CSV file using pandas (recommended method).

    Args:
        t_np (np.ndarray): Array of time points.
        solution_np (np.ndarray): State matrix (number of time steps x number of nodes).
        R_param (float): R parameter.
        B_param (float): B parameter.
        output_dir (str): Output directory.
    """
    num_nodes = solution_np.shape[1]
    
    # 1. Create an index representing combinations of (time, node_id).
    #    np.repeat duplicates the time array: [t0, t1, ...] -> [t0, t0, ..., t1, t1, ...]
    #    np.tile repeats the node ID array: [0, 1, ..., 99] -> [0, 1, ..., 99, 0, 1, ..., 99, ...]
    time_col = np.repeat(t_np, num_nodes)
    node_id_col = np.tile(np.arange(num_nodes), len(t_np))
    
    # 2. Flatten the 2D state matrix into a 1D column.
    state_col = solution_np.flatten()
    
    # 3. Create a DataFrame.
    df = pd.DataFrame({
        'time': time_col,
        'node_id': node_id_col,
        'state': state_col
    })
    
    # 4. Define the filename and save.
    trajectory_file = os.path.join(output_dir, f"undisturbed_trajectories_R_{R_param}_B_{B_param}.csv")
    df.to_csv(trajectory_file, index=False, float_format='%.6f')
    
    print(f"Detailed trajectory data has been efficiently saved to: {trajectory_file}")


def fhn_limit_cycle_propagation(quick=False):
    """Simulate FitzHugh–Nagumo dynamics using the limit_cycle_scan configuration."""

    if quick:
        config = SimulationConfig(
            grid_size=2,
            e_range=(0.1, 1.0),
            f_range=(0.1, 1.5),
            total_time=100.0,
            num_time_samples=2000,
            transient_ratio=0.5,
            network_size=100,
            barabasi_m=2,
            random_seed=42,
            amplitude_threshold=0.05,
        )
    else:
        config = SimulationConfig()

    output_root = Path('output') / 'fhn'
    output_root.mkdir(parents=True, exist_ok=True)

    adjacency, degrees = fhn_build_network(config)
    np.save(output_root / 'fhn_matrix.npy', adjacency)
    np.save(output_root / 'degrees.npy', degrees)

    x0_flat = fhn_initial_conditions(config).astype(np.float64)
    times = np.linspace(0.0, config.total_time, config.num_time_samples)
    steady_start = int(config.num_time_samples * config.transient_ratio)

    e_values = np.round(np.linspace(config.e_range[0], config.e_range[1], config.grid_size), 3)
    f_values = np.round(np.linspace(config.f_range[0], config.f_range[1], config.grid_size), 3)
    graph_set=list(itertools.product(e_values, f_values))
    np.save(output_root / 'parameter_list.npy', graph_set)
    max_matrix = np.zeros((config.grid_size, config.grid_size))
    min_matrix = np.zeros((config.grid_size, config.grid_size))
    period_matrix = np.zeros((config.grid_size, config.grid_size))
    amplitude_matrix = np.zeros((config.grid_size, config.grid_size))
    no_cycle_mask = np.zeros((config.grid_size, config.grid_size), dtype=bool)

    result_dir_base = output_root / 'quick_results' if quick else output_root
    result_dir, phase_dir = scan_ensure_output_dirs(result_dir_base)

    records = []

    with tqdm(
        total=config.grid_size * config.grid_size,
        desc='FHN (e,f) grid' if not quick else 'FHN quick grid',
        unit='pt',
    ) as pbar:
        for i, f_val in enumerate(f_values):
            for j, e_val in enumerate(e_values):
                params = {'e': float(e_val), 'f': float(f_val)}
                solution = scipy_odeint(
                    fhn_limit_cycle_dynamics,
                    x0_flat,
                    times,
                    args=(adjacency, degrees, config.network_size, params, 1.0, 0.04),
                )
                trajectory = solution.reshape(config.num_time_samples, config.network_size, 2)
                # Save the network-wide X1 trajectory for each (e,f), consistent with save_detailed_trajectory_pandas column format
                x1_matrix = trajectory[:, :, 0]  # (T, N)
                num_nodes = x1_matrix.shape[1]
                time_col = np.repeat(times, num_nodes)
                node_id_col = np.tile(np.arange(num_nodes), len(times))
                state_col_x1 = x1_matrix.flatten()
                df_fhn_x1 = pd.DataFrame({
                    'time': time_col,
                    'node_id': node_id_col,
                    'state': state_col_x1,
                })
                # Keep the original X1 naming unchanged for backward compatibility
                e_round = np.round(params['e'], 3)
                f_round = np.round(params['f'], 3)
                traj_path_x1 = result_dir / f"undisturbed_trajectories_R_{e_round}_B_{f_round}.csv"
                df_fhn_x1.to_csv(traj_path_x1, index=False, float_format='%.6f')

                # Save X2 separately, using _x2 suffix to distinguish
                x2_matrix = trajectory[:, :, 1]
                state_col_x2 = x2_matrix.flatten()
                df_fhn_x2 = pd.DataFrame({
                    'time': time_col,
                    'node_id': node_id_col,
                    'state': state_col_x2,
                })
                traj_path_x2 = result_dir / f"undisturbed_trajectories_R_{e_round}_B_{f_round}_x2.csv"
                df_fhn_x2.to_csv(traj_path_x2, index=False, float_format='%.6f')

                steady = trajectory[steady_start:, 0, :]

                metrics = compute_limit_cycle_metrics(
                    times[steady_start:],
                    steady,
                    config.amplitude_threshold,
                )

                max_matrix[i, j] = metrics['x1_max']
                min_matrix[i, j] = metrics['x1_min']
                period_matrix[i, j] = metrics['period']
                amplitude_matrix[i, j] = metrics['amplitude']
                no_cycle_mask[i, j] = not metrics['has_limit_cycle']

                records.append(
                    {
                        'e': params['e'],
                        'f': params['f'],
                        'x1_max': metrics['x1_max'],
                        'x1_min': metrics['x1_min'],
                        'amplitude': metrics['amplitude'],
                        'period': metrics['period'],
                        'has_limit_cycle': metrics['has_limit_cycle'],
                    }
                )

                safe_stem = f"phase_e_{params['e']:.3f}_f_{params['f']:.3f}".replace('.', 'p')
                generate_phase_plot(steady, params, metrics['has_limit_cycle'], phase_dir / f"{safe_stem}.png")
                pbar.update(1)

    records_df = pd.DataFrame.from_records(records)
    records_df.to_csv(result_dir / 'limit_cycle_metrics.csv', index=False)

    np.savez(
        result_dir / 'limit_cycle_metrics.npz',
        e_values=e_values,
        f_values=f_values,
        x1_max=max_matrix,
        x1_min=min_matrix,
        amplitude=amplitude_matrix,
        period=period_matrix,
        no_limit_cycle=no_cycle_mask,
    )

    plot_heatmap(
        max_matrix,
        e_values,
        f_values,
        'Max X1 on Limit Cycle',
        'Max X1',
        no_cycle_mask,
        result_dir / 'heatmap_max_x1.png',
    )
    plot_heatmap(
        min_matrix,
        e_values,
        f_values,
        'Min X1 on Limit Cycle',
        'Min X1',
        no_cycle_mask,
        result_dir / 'heatmap_min_x1.png',
    )
    plot_heatmap(
        period_matrix,
        e_values,
        f_values,
        'Oscillation Period',
        'Period (time units)',
        no_cycle_mask,
        result_dir / 'heatmap_period.png',
        cmap='magma',
    )


def collab_propagation():
    output_dir = 'output/collab'
    R_0=np.round(np.linspace(0.01, 0.1, 40), 4)
    os.makedirs(output_dir, exist_ok=True)
    R=0.02
    R_list=[R]
    B_list = np.round(R/R_0, 4)

    # Find the B value you are interested in
    target_B = 0.4264
    if target_B in B_list:
        b_index = np.where(B_list == target_B)[0][0]
        corresponding_R0 = R_0[b_index]
        print(f"Found B = {target_B}")
        print(f"Its index in B_list is: {b_index}")
        print(f"The corresponding R_0 value is: {corresponding_R0}")
    else:
        print(f"B = {target_B} not found in the generated B_list")
        # Find the closest value
        closest_B_index = np.argmin(np.abs(B_list - target_B))
        closest_B = B_list[closest_B_index]
        corresponding_R0 = R_0[closest_B_index]
        print(f"The closest B value is {closest_B} (at index {closest_B_index})")
        print(f"Generated from R_0 = {corresponding_R0}")


    parameter_list = list(itertools.product(R_list, B_list))
    print()
    np.save(os.path.join(output_dir, 'parameter_list.npy'), parameter_list)
    # --- "Guaranteed Success" single-peak parameters ---
    R_param = 0.015# Key: use a high transmission rate to create a rapid outbreak
    B_param = 0.12      # Baseline recovery rate
    EDGE_M = 2         # Keep the network sparse to ensure the epidemic terminates
    TOTAL_T = 10      # Sufficient time to observe the full peak
    NUM_NODES = 200    # Number of nodes in the network
    INITIAL_COUNT = 1  # Start with one "patient zero" to make the curve more classic
    # ---------------------------

    # 1. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Create graph structure
    np.random.seed(42) # Ensure consistency of the graph and initial nodes
    A = np.load(os.path.join(output_dir,'collab_matrix.npy'))
    NUM_NODES=A.shape[0]
    
    #np.save(os.path.join(output_dir, 'adjacency_matrix.npy'), A)
    for R,B in parameter_list:
        R_param = R
        B_param = B
        epidemic_model = EpidemicModelTorch(R=R_param, B=B_param, A_np=A)
        print(f"Epidemic dynamics model initialized (R={R_param}, B={B_param}, m={EDGE_M}, alpha={epidemic_model.alpha})")

        # 4. Set initial conditions
        x0 = np.zeros(NUM_NODES)
        x0 = np.full(NUM_NODES, 0.01)

        # Ensure the number of initially infected individuals does not exceed the total number of nodes
        num_initial_infected = 60
        # Randomly select indices of num_initial_infected nodes
        # replace=False ensures that the same node is not selected multiple times
        initial_infected_indices = np.random.choice(NUM_NODES, num_initial_infected, replace=False)
        x0[initial_infected_indices] = 1.0 
        device = epidemic_model.device
        x0_torch = torch.tensor(x0, dtype=torch.float, device=device)

        # 5. Run simulation
        t_span = torch.linspace(0, TOTAL_T, 500).to(device)
        solution = odeint(epidemic_model.f, x0_torch, t_span)

        # 6. Analyze and visualize results
        solution_np = solution.cpu().numpy()
        t_np = t_span.cpu().numpy()
        total_infected = solution_np.sum(axis=1)
        save_detailed_trajectory_pandas(t_np, solution_np, R_param, B_param, output_dir)
        # Plot single-peak curve
        plt.figure(figsize=(10, 6))
        plt.plot(t_np, total_infected, label=f'Total Infected (R={R_param}, B={B_param}, m={EDGE_M})', lw=2.5, color='crimson')
        plt.title('Epidemic Dynamics: A Clear & Sharp Single-Peak Curve')
        plt.xlabel('Time')
        plt.ylabel('Total Number of Infected Individuals')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Find and mark the peak
        peak_index = np.argmax(total_infected)
        peak_time = t_np[peak_index]
        peak_value = total_infected[peak_index]
        plt.axvline(peak_time, color='gray', linestyle='--', label=f'Peak at t={peak_time:.2f}')
        plt.scatter(peak_time, peak_value, color='black', zorder=5)
        plt.text(peak_time + 2, peak_value * 0.9, f'Peak: {peak_value:.2f}', fontsize=12)
        
        plt.legend()
        
        output_path = os.path.join(output_dir, f'Guaranteed_SinglePeak_R_{R_param}_B_{B_param}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"Image saved to: {output_path}")

def twitter_propagation():
    output_dir = 'output/twitter'
    os.makedirs(output_dir, exist_ok=True)
    R_list = np.round(np.linspace(0.02, 0.08, 20), 3)
    B_list = np.round(np.linspace(0.01, 0.12, 20), 3)
    parameter_list = list(itertools.product(R_list, B_list))
    np.random.seed(42) # Ensure consistency of the graph and initial nodes
    np.save(os.path.join(output_dir, 'parameter_list.npy'), parameter_list)
    TOTAL_T = 50
    A = np.load(os.path.join(output_dir,'twitter_matrix.npy'))
    NUM_NODES=A.shape[0]    
    for R,B in parameter_list:
        R_param = R
        B_param = B
        epidemic_model = SIRPropagation(beta=R_param, gamma=B_param, A=A)
        print(f"Epidemic dynamics model initialized (R={R_param}, B={B_param})")
        # 4. Set initial conditions
        x0 = np.zeros(NUM_NODES)
        x0 = np.full(NUM_NODES, 0.01)
        # Ensure the number of initially infected individuals does not exceed the total number of nodes
        num_initial_infected = 50
        # Randomly select indices of num_initial_infected nodes
        # replace=False ensures that the same node is not selected multiple times
        np.random.seed(42) # Ensure consistency in each selection
        initial_infected_indices = np.random.choice(NUM_NODES, num_initial_infected, replace=False)
        x0[initial_infected_indices] = 1.0 
        x0[0]=1.0       
        device = epidemic_model.device
        x0_torch = torch.tensor(x0, dtype=torch.float, device=device)

        # 5. Run simulation
        t_span = torch.linspace(0, TOTAL_T, 500).to(device)
        solution = odeint(epidemic_model.f, x0_torch, t_span)

        # 6. Analyze and visualize results
        solution_np = solution.cpu().numpy()
        t_np = t_span.cpu().numpy()
        total_infected = solution_np.sum(axis=1)
        save_detailed_trajectory_pandas(t_np, solution_np, R_param, B_param, output_dir)
        # Plot single-peak curve
        plt.figure(figsize=(10, 6))
        plt.plot(t_np, total_infected, label=f'Total Infected (R={R_param}, B={B_param})', lw=2.5, color='crimson')
        plt.title('Epidemic Dynamics: A Clear & Sharp Single-Peak Curve')
        plt.xlabel('Time')
        plt.ylabel('Total Number of Infected Individuals')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Find and mark the peak
        peak_index = np.argmax(total_infected)
        peak_time = t_np[peak_index]
        peak_value = total_infected[peak_index]
        plt.axvline(peak_time, color='gray', linestyle='--', label=f'Peak at t={peak_time:.2f}')
        plt.scatter(peak_time, peak_value, color='black', zorder=5)
        plt.text(peak_time + 2, peak_value * 0.9, f'Peak: {peak_value:.2f}', fontsize=12)
        
        plt.legend()
        
        output_path = os.path.join(output_dir, f'Guaranteed_SinglePeak_R_{R_param}_B_{B_param}.png')
        plt.savefig(output_path)
        print(f"Image saved to: {output_path}")

def euroad_propagation():
    output_dir = 'output/euroad'
    os.makedirs(output_dir, exist_ok=True)
    R_list = np.round(np.linspace(0.02, 0.08, 20), 3)
    B_list = np.round(np.linspace(0.01, 0.12, 20), 3)
    parameter_list = list(itertools.product(R_list, B_list))
    np.save(os.path.join(output_dir, 'parameter_list.npy'), parameter_list)
    # --- "Guaranteed Success" single-peak parameters ---
    R_param = 0.015# Key: use a high transmission rate to create a rapid outbreak
    B_param = 0.12      # Baseline recovery rate
    EDGE_M = 2         # Keep the network sparse to ensure the epidemic terminates
    TOTAL_T = 50      # Sufficient time to observe the full peak
    NUM_NODES = 200    # Number of nodes in the network
    # 1. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # 2. Create graph structure
    np.random.seed(42) # Ensure consistency of the graph and initial nodes
    A = np.load(os.path.join(output_dir,'euroad_matrix.npy'))
    NUM_NODES=A.shape[0]
    for R,B in parameter_list:
        R_param = R
        B_param = B
        epidemic_model = EpidemicModelTorch(R=R_param, B=B_param, A_np=A)
        print(f"Epidemic dynamics model initialized (R={R_param}, B={B_param}, m={EDGE_M}, alpha={epidemic_model.alpha})")
        x0 = np.zeros(NUM_NODES)
        x0 = np.full(NUM_NODES, 0.01)
        num_initial_infected = 50
        initial_infected_indices = np.random.choice(NUM_NODES, num_initial_infected, replace=False)
        x0[initial_infected_indices] = 1.0 
        x0[0]=1.0       
        device = epidemic_model.device
        x0_torch = torch.tensor(x0, dtype=torch.float, device=device)
        t_span = torch.linspace(0, TOTAL_T, 500).to(device)
        solution = odeint(epidemic_model.f, x0_torch, t_span)
        solution_np = solution.cpu().numpy()
        t_np = t_span.cpu().numpy()
        total_infected = solution_np.sum(axis=1)
        save_detailed_trajectory_pandas(t_np, solution_np, R_param, B_param, output_dir)
        plt.figure(figsize=(10, 6))
        plt.plot(t_np, total_infected, label=f'Total Infected (R={R_param}, B={B_param}, m={EDGE_M})', lw=2.5, color='crimson')
        plt.title('Epidemic Dynamics: A Clear & Sharp Single-Peak Curve')
        plt.xlabel('Time')
        plt.ylabel('Total Number of Infected Individuals')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        peak_index = np.argmax(total_infected)
        peak_time = t_np[peak_index]
        peak_value = total_infected[peak_index]
        plt.axvline(peak_time, color='gray', linestyle='--', label=f'Peak at t={peak_time:.2f}')
        plt.scatter(peak_time, peak_value, color='black', zorder=5)
        plt.text(peak_time + 2, peak_value * 0.9, f'Peak: {peak_value:.2f}', fontsize=12)
        plt.legend()        
        output_path = os.path.join(output_dir, f'R_{R_param}_B_{B_param}.png')
        plt.savefig(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select which graph generation/simulation function to run."
    )
    parser.add_argument(
        "--mode",
        choices=["fhn", "collab", "twitter", "euroad","hill"],
        default="collab",
        help="Which simulation to run.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataname = args.mode
    if dataname == 'fhn':
        fhn_limit_cycle_propagation()
    elif dataname == 'collab':
        collab_propagation()
    elif dataname == 'twitter':
        twitter_propagation()
    elif dataname == 'euroad':
        euroad_propagation()
    elif dataname == 'hill':
        hill_propagation()


