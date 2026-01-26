import numpy as np
import torch
class SIRPropagation:
    def __init__(self, beta, gamma, A, learning_rate=1.0):
        """
        Initialize the network SIR information propagation dynamics model.

        Args:
            beta (float): Information propagation/infection rate (corresponding to Promotion Rate).
            gamma (float): Information forgetting/recovery rate (corresponding to Decay Rate).
            A (np.ndarray): The original adjacency matrix (containing positive and negative weights, usually non-negative for follow relationships in Twitter).
            learning_rate (float): A multiplier for the overall dynamics adjustment (usually set to 1.0 in standard SIR).
        """
        self.A = A
        # Set the two user-specified tunable parameters as core rates
        self.beta = beta
        self.gamma = gamma 
        self.learning_rate = learning_rate # Keep the learning rate as an overall speed multiplier, but it is usually 1.0 in standard SIR

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # SIR propagation mainly depends on positive connections, so we use A_plus as the influence matrix
        # A_plus contains all positive weights, A_minus contains the absolute values of all negative weights (although not used in SIR propagation)
        self.A_plus = np.where(self.A > 0, self.A, 0)
        
        # For standard SIR, normalization (d_u_inv_diag) is usually not performed
        # If A_plus needs to be normalized to a row-stochastic matrix, the normalization term needs to be recalculated.
        # Here we assume that A_plus is already a representation of weight/influence and use it directly.
        
        print(f"SIRPropagation initialized with: beta (propagation rate)={self.beta}, gamma (forgetting rate)={self.gamma}")
        print(f"  - A_plus shape (influence matrix): {self.A_plus.shape}")
        
        # Convert NumPy matrix to PyTorch tensor and move to device
        self.A_plus_t = torch.from_numpy(self.A_plus).float().to(self.device)
        # Remove A_minus_t and d_u_inv_diag_t which are not needed in the original framework

    def f(self, t, x_current_t):
        """
        Define the dynamic equation dx/dt to implement the network-based SIR information propagation model.
        
        dx/dt = beta * (1 - x) * (A_plus * x) - gamma * x
        
        Args:
            t (float): Current time.
            x_current_t (torch.Tensor): The current state of all nodes x (informed/infected probability).
            
        Returns:
            torch.Tensor: State change rate dx/dt.
        """
        # Check and handle batch_size
        if x_current_t.dim() == 1:
            x_current_reshaped = x_current_t
        elif x_current_t.dim() == 2 and x_current_t.shape[0] == 1:
            x_current_reshaped = x_current_t.squeeze(0)
        else:
            raise ValueError(f"Unexpected x_current_t shape: {x_current_t.shape}. Expected (node_num,) or (1, node_num).")
            
        x_current_reshaped = x_current_reshaped.to(self.device)
        
        # Ensure x is in the range [0, 1]
        x_safe_t = torch.clamp(x_current_reshaped, 0.0, 1.0)
        
        # 1. Infection term (S -> I): beta * (1 - x) * Influence
        # Neighbor influence term (Influence): A_plus * x_safe_t
        # This is to calculate the total influence/information flow that each node i receives from its infected neighbors
        neighbor_influence_t = torch.matmul(self.A_plus_t, x_safe_t)
        
        # Susceptible term (S): (1 - x_safe_t)
        susceptible_term_t = (1.0 - x_safe_t)
        
        # Information acquisition/infection term: beta * S * Influence
        term_infection_t = self.beta * susceptible_term_t * neighbor_influence_t
        
        # 2. Recovery/forgetting term (I -> R): gamma * x
        term_recovery_t = self.gamma * x_safe_t
        
        # 3. Final dx/dt
        dxdt_t = self.learning_rate * (term_infection_t - term_recovery_t)
        
        # NaN value check and handling (consistent with the original framework)
        if torch.isnan(dxdt_t).any():
            nan_indices = torch.where(torch.isnan(dxdt_t))[0]
            print(f'SIRPropagation: Found nan values at nodes: {nan_indices.cpu().numpy()}')
            dxdt_t = torch.nan_to_num(dxdt_t, nan=0.0)
            print('Replaced nan values with 0.0 and continued simulation')
            
        # Limit the range of the rate of change
        dxdt_t = torch.clamp(dxdt_t, -1e3, 1e3)
        
        # Restore the batch dimension if the original input had one
        if x_current_t.dim() == 2 and x_current_t.shape[0] == 1:
            return dxdt_t.unsqueeze(0)
        else:
            return dxdt_t

    def g(self, x, t):
        """
        Define the random term g(x, t) (no noise added for now).
        """
        # Assume state x is a vector (num_nodes,)
        num_nodes = x.shape[0]
        # Return a zero matrix, indicating no random noise
        return np.zeros((num_nodes, num_nodes))

class TrustDynamics:
    def __init__(self, args, A):
        """
        Initialize the belief network dynamics model (based on the discrete formula you provided)
        Args:
            args: Configuration parameters (args object passed from NetworkPerturbationSimulator)
            A: Adjacency matrix (from Bitcoin Alpha, containing positive and negative weights)
        """
        self.A = A
        param = args[args.dynamics] # args.dynamics should point to 'TrustDynamics'
        
        # New tunable parameters
        self.beta = getattr(param, 'beta', 0.5) # Global update speed/learning rate, range (0, 1]
        self.gamma = getattr(param, 'gamma', 1.0) # Non-linear index of neighbor influence (similar to Hill coefficient)
        self.threshold = getattr(param, 'threshold', 0.5) # Half-saturation threshold for neighbor influence
        
        # Ensure sim_dt is available to convert discrete steps to continuous rate of change
        self.sim_dt = getattr(param, 'sim_dt', 0.01)

        # Separate A_plus and A_minus according to the A matrix (containing positive and negative weights)
        # A_plus[i,j] is the positive influence strength of j on i
        # A_minus[i,j] is the negative influence strength of j on i (stored as a positive value)
        self.A_plus = np.where(self.A > 0, self.A, 0)
        self.A_minus = np.where(self.A < 0, np.abs(self.A), 0)

        # Calculate d_u (denominator term): the total influence strength of each node u (in-degree, weighted)
        # d_u = sum_j (A_uj+ + A_uj-)
        # Here we assume that d_u is the sum of the absolute values of all edges pointing to u (positive and negative weights)
        self.d_u = np.sum(self.A_plus + self.A_minus, axis=1) # axis=1 means summing each row, i.e., the sum of in-degree weights for each node
        
        # Avoid division by zero: replace nodes where d_u is zero with a small value
        self.d_u_safe = np.where(self.d_u == 0, 1e-6, self.d_u)
        
        # Pre-calculate the inverse of d_u for matrix multiplication
        self.d_u_inv_diag = np.diag(1.0 / self.d_u_safe)

        print(f"TrustDynamics initialized with: beta={self.beta}, gamma={self.gamma}, threshold={self.threshold}")
        print(f"  - A_plus shape: {self.A_plus.shape}")
        print(f"  - A_minus shape: {self.A_minus.shape}")
        print(f"  - d_u_safe min: {self.d_u_safe.min()}, max: {self.d_u_safe.max()}")

    def f(self, x_current, t):
        """
        Define the dynamic equation dx/dt, corresponding to the belief network formula you provided
        Args:
            x: current state (x_u, belief probability of each node, range [0, 1])
            t: current time
        Returns:
            dxdt: state change rate
        """
        node_num = x_current.shape[0]
        dxdt = np.zeros(node_num)
        
        # Ensure x is in the range [0, 1] to avoid numerical problems
        x_safe = np.clip(x_current, 1e-6, 1.0 - 1e-6)

        # Introduce non-linear saturation effect (optional, if gamma=1, threshold=0.5, it is linear)
        # This form is similar to the Hill function, which makes the influence of neighbors significant only after x_j reaches a certain value
        if self.gamma != 1.0 or self.threshold != 0.5:
            # Assume the influence on x_j is processed by a saturation function
            x_transformed = (x_safe ** self.gamma) / (x_safe ** self.gamma + self.threshold ** self.gamma)
        else:
            x_transformed = x_safe # If the parameters are default values, maintain linear influence

        # Calculate the first term in the formula: sum_j (A_uj+/du) * xj(t-1)
        # np.dot(self.A_plus, x_transformed) calculates sum_j A_uj+ * x_j_transformed
        # then multiply by d_u_inv_diag for normalization
        term1_influence = np.dot(self.d_u_inv_diag, np.dot(self.A_plus, x_transformed))
        
        # Calculate the second term in the formula: sum_j (A_uj-/du) * (1 - xj(t-1))
        term2_influence = np.dot(self.d_u_inv_diag, np.dot(self.A_minus, (1 - x_transformed)))
        
        # Get the next target state x_target calculated by the discrete formula
        x_target = term1_influence + term2_influence
        
        # Convert discrete update to continuous time rate of change and introduce global update speed beta
        # dxdt = beta * (x_target - x_current) / sim_dt
        dxdt = self.beta * (x_target - x_current) / self.sim_dt
        
        # Ensure the state does not exceed the [0, 1] range (numerical stability)
        # If x is close to 0 and dxdt is negative, set dxdt to 0
        dxdt = np.where((x_current <= 1e-6) & (dxdt < 0), 0, dxdt)
        # If x is close to 1 and dxdt is positive, set dxdt to 0
        dxdt = np.where((x_current >= 1.0 - 1e-6) & (dxdt > 0), 0, dxdt)

        # NaN value check and handling
        if np.isnan(dxdt).any():
            nan_indices = np.where(np.isnan(dxdt))[0]
            print(f'TrustDynamics: Found nan values at nodes: {nan_indices}')
            print(f'State values at these nodes: {x_current[nan_indices]}')
            dxdt[np.isnan(dxdt)] = 0.0
            print('Replaced nan values with 0.0 and continued simulation')
        
        # Limit the rate of change to avoid numerical instability
        dxdt = np.clip(dxdt, -1e3, 1e3)

        return dxdt

    def g(self, x, t):
        """inherent noise (no noise added for now)"""
        return np.diag([0.0] * x.shape[0])
def _calculate_dx_dt_torch(W_matrix, delta, eta, x_state, clip_x_input=True):
    """
    Continuous time rate of change: dx/dt = (W*eta)^T @ x - delta * x
    Params
    ------
    W_matrix : (n,n) torch.Tensor
        Directed weighted adjacency matrix.
    delta : float
        Decay rate δ > 0
    eta : float
        Additional multiplier η
    x_state : (n,) torch.Tensor
        Current state vector (generally within [0,1])
    clip_x_input : bool
        Whether to clip x to [0,1] before calculating dxdt
    Returns
    -------
    dxdt : (n,) torch.Tensor
    """
    # Replace numpy.clip with torch.clamp
    if clip_x_input:
        x_safe = torch.clamp(x_state, 0.0, 1.0)
    else:
        x_safe = x_state

    # Note: Your original code is W.T @ x_safe, but you pass W*eta in activation_euler_step.
    # So, the W_matrix here should be the original W, and then we apply eta here.
    # And, torch.matmul(A, B) is matrix multiplication for 2D matrices.
    # If W_matrix is (n,n) and x_safe is (n,), then W_matrix @ x_safe is (n,).
    # Your original formula is W.T @ x, so we should use W_matrix.T here.
    
    # Core dynamic equation
    dxdt = torch.matmul((W_matrix * eta).T, x_safe) - delta * x_safe
    
    # The original activation_dxdt does not have rate_clip, that is in activation_euler_step.
    # For odeint, we usually do not clip dxdt here because it expects a continuous derivative.
    # If you do need to limit the derivative, you can add torch.clamp(dxdt, -rate_clip, rate_clip) here
    
    return dxdt

class InformationDynamics:
    """
    A simple simulator: given W (first parameter) and delta (second parameter),
    the state x(t) can be evolved on [0,T] using Euler's method.
    """
    # 1. Correct the __init__ method name
    
    def __init__(self, W, delta, eta, dt=0.01, clip01=True,device='cuda:0'):
        # Ensure W is converted to torch.Tensor and is of type float32
        self.W = torch.tensor(W, dtype=torch.float32).to(device)
        self.delta = float(delta)
        self.eta = float(eta)
        self.dt = float(dt) # This dt is not used directly in odeint
        self.clip01 = clip01 # This parameter now controls the clipping of x in _calculate_dx_dt_torch

    # 2. Correct the signature of the f method to conform to the requirements of odeint f(t, x)
    def f(self, t, x):
        # odeint will pass in the current time t and the current state x
        # We need to return dx/dt
        
        # 3. Call our dx/dt calculation function prepared for odeint
        # The W matrix is already stored as a torch.Tensor
        dxdt = _calculate_dx_dt_torch(self.W, self.delta, self.eta, x, clip_x_input=self.clip01)
        
        return dxdt
class FitzHughNagumo:
    def __init__(self, args, A):
        self.L = A - np.diag(np.sum(A, axis=1)) # Difussion matrix: x_j - x_i
        
        param = args[args.dynamics]
        self.a = param.a
        self.b = param.b
        self.c = param.c
        self.epsilon = param.epsilon
        self.k_in = np.sum(A, axis=1) # in-degree
    
    def f(self, x, t):
        
        node_num = x.shape[0] // 2
        x1, x2 = x[:node_num], x[node_num:]
        
        # x1
        f_x1 = x1 - (x1 ** 3)/3 - x2
        outer_x1 = self.epsilon * np.dot(self.L, 1/self.k_in) # epsilon * sum_j Aij * ((x1_j - x1_i) / k_in_i)
        dx1dt = f_x1 + outer_x1
        
        # x2
        f_x2 = self.a + self.b * x1 + self.c * x2
        outer_x2 = 0.0
        dx2dt = f_x2 + outer_x2
        
        if np.isnan(dx1dt).any():
            print('nan during simulation!')
            exit()
        
        dxdt = np.concatenate([dx1dt, dx2dt], axis=0)
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    

def fhn_limit_cycle_dynamics(x_flat,
                             t,
                             adjacency,
                             degrees,
                             network_size,
                             params,
                             coupling_weight=1.0,
                             x2_decay=0.04):
    """FitzHugh–Nagumo network dynamics matching limit_cycle_scan."""

    x = x_flat.reshape(network_size, 2)
    x1 = x[:, 0]
    x2 = x[:, 1]

    degrees_safe = np.where(degrees <= 0.0, 1.0, degrees)
    coupling_sum = adjacency @ x1 - degrees * x1
    coupling_term = coupling_weight * coupling_sum / degrees_safe

    e_param = float(params.get("e", 0.0))
    f_param = float(params.get("f", 0.0))

    dx1_dt = 1.0 * x1 - 1.0 * x2 - 1.0 * (x1 ** 3) - coupling_term
    dx2_dt = e_param + f_param * x1 - x2_decay * x2

    derivatives = np.column_stack((dx1_dt, dx2_dt))
    return derivatives.reshape(-1)


class FitzHughNagumoNetwork:
    def __init__(self, args, A):
        self.A = A.astype(np.float64)
        self.node_num = self.A.shape[0]
        param = args[args.dynamics]
        self.e = float(getattr(param, "e", 0.0))
        self.f = float(getattr(param, "f", 1.0))
        self.coupling_weight = float(getattr(param, "coupling_weight", 1.0))
        self.x2_decay = float(getattr(param, "x2_decay", 0.04))
        self.state_dim_per_node = 2
        degrees = self.A.sum(axis=1, dtype=np.float64)
        self.degrees = np.where(degrees == 0.0, 1.0, degrees)

    def default_initial_state(self, seed=None):
        rng = np.random.default_rng(seed)
        init = rng.uniform(-1.0, 1.0, size=(self.node_num, self.state_dim_per_node))
        return init.reshape(-1).astype(np.float64)

    def f(self, x_flat, t):
        x = x_flat.reshape(self.node_num, self.state_dim_per_node)
        x1 = x[:, 0]
        x2 = x[:, 1]
        coupling_sum = self.A @ x1 - self.degrees * x1
        coupling_term = (self.coupling_weight * coupling_sum) / self.degrees
        dx1_dt = x1 - x2 - (x1 ** 3) - coupling_term
        dx2_dt = self.e + self.f * x1 - self.x2_decay * x2
        dxdt = np.column_stack((dx1_dt, dx2_dt))
        if np.isnan(dxdt).any() or np.isinf(dxdt).any():
            nan_indices = np.argwhere(~np.isfinite(dxdt))
            print(f"FitzHughNagumoNetwork: non-finite derivative entries at indices {nan_indices.tolist()}")
            dxdt = np.nan_to_num(dxdt, nan=0.0, posinf=1e3, neginf=-1e3)
        dxdt = np.clip(dxdt, -1e3, 1e3)
        return dxdt.reshape(-1).astype(np.float64)

    def g(self, x, t):
        dim = self.node_num * self.state_dim_per_node
        return np.zeros((dim, dim))


class HindmarshRose:
    def __init__(self, args, A):
        self.A = A   # Adjacency matrix
        
        param = args[args.dynamics]
        self.a = param.a
        self.b = param.b
        self.c = param.c
        self.u = param.u
        self.s = param.s
        self.r = param.r
        self.epsilon = param.epsilon
        self.v = param.v
        self.lam = param.lam
        self.I = param.I
        self.omega = param.omega
        self.x0 = param.x0
    
    def f(self, x, t):
        
        node_num = x.shape[0] // 3
        x1, x2, x3 = x[:node_num], x[node_num:2*node_num], x[2*node_num:]
        mu_xj = 1 / (1 + np.exp(-self.lam * (x1 - self.omega)))
        
        # x1
        f_x1 = x2 - self.a * x1 ** 3 + self.b * x1 ** 2 - x3 + self.I
        outer_x1 = self.epsilon * (self.v - x1) * np.dot(self.A, mu_xj)
        dx1dt = f_x1 + outer_x1
        
        # x2
        f_x2 = self.c - self.u * x1 ** 2 - x2
        outer_x2 = 0.0
        dx2dt = f_x2 + outer_x2
        
        # x3
        f_x3 = self.r * (self.s * (x1 - self.x0) - x3)
        outer_x3 = 0.0
        dx3dt = f_x3 + outer_x3
        
        if np.isnan(dx1dt).any():
            print('nan during simulation!')
            exit()
        
        dxdt = np.concatenate([dx1dt, dx2dt, dx3dt], axis=0)
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    
    
class CoupledRossler:
    def __init__(self, args, A):
        self.L = A - np.diag(np.sum(A, axis=1)) # Difussion matrix: x_j - x_i
        
        param = args[args.dynamics]
        self.a = param.a
        self.b = param.b
        self.c = param.c
        self.epsilon = param.epsilon
        self.delta = param.delta
    
    def f(self, x, t):
        
        node_num = x.shape[0] // 3
        x1, x2, x3 = x[:node_num], x[node_num:2*node_num], x[2*node_num:]
        omega = np.random.normal(1, self.delta, size=node_num)
        
        # x1
        f_x1 = - omega * x2 - x3
        outer_x1 = self.epsilon * np.dot(self.L, x1)
        dx1dt = f_x1 + outer_x1
        
        # x2
        f_x2 = omega * x1 + self.a * x2
        outer_x2 = 0.0
        dx2dt = f_x2 + outer_x2
        
        # x3
        f_x3 = self.b + x3 * (x1 + self.c)
        outer_x3 = 0.0
        dx3dt = f_x3 + outer_x3
        
        if np.isnan(dx1dt).any():
            print('nan during simulation!')
            exit()
        
        dxdt = np.concatenate([dx1dt, dx2dt, dx3dt], axis=0)
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    

class EpidemicModel:
    def __init__(self, args, A):
        """
        Initialize the epidemic spread model
        Args:
            args: configuration parameters
            A: adjacency matrix
        """
        self.A = A  # Adjacency matrix
        # Create a copy of the adjacency matrix without self-loops for propagation calculation, ensuring i != j
        self.A_no_diag = self.A.copy()
        np.fill_diagonal(self.A_no_diag, 0)

        param = args[args.dynamics]
        self.R = param.R  # Transmission rate
        self.B = param.B  # Recovery rate
        self.alpha = 0.1  # Exponential parameter fixed to 1
        self.w = 1.0      # Node weight fixed to 1

    def f(self, x, t):
        """
        Define the dynamic equation dx/dt (vectorized version)
        Args:
            x: current state
            t: current time
        Returns:
            dxdt: state change rate
        """
        # Safety check: limit the state to the [0, 1] interval, which is consistent with the physical meaning of the epidemic model
        x_safe = np.clip(x, 0, 1)

        # Recovery term (vectorized)
        recovery_term = -self.B * (x_safe ** self.alpha)

        # Spreading term (vectorized)
        # Corresponding to (1 - x[i]) * R * sum_{j, j!=i}(A[i,j] * x[j])
        infection_pressure = np.dot(self.A_no_diag, x_safe)
        spreading_term = self.R * (1 - x_safe) * infection_pressure

        # Combine all terms and multiply by node weight
        dxdt = self.w * (recovery_term + spreading_term)

        if np.isnan(dxdt).any():
            nan_indices = np.where(np.isnan(dxdt))[0]
            print(f'nan during simulation! Found at nodes: {nan_indices}')
            print(f'State values at these nodes: {x[nan_indices]}')
            dxdt[np.isnan(dxdt)] = 0.0
            print('NaN values have been replaced with 0.0 to continue simulation.')

        return dxdt

    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    

class EcologicalModel:
    def __init__(self, args, A):
        """
        Initialize the ecosystem dynamics model
        Args:
            args: Configuration parameters
            A: Adjacency matrix
        """
        self.A = A
        
        param = args[args.dynamics]
        self.B = param.B  # Intrinsic growth rate
        self.K = param.K  # Environmental carrying capacity (recommended value range: integer from 1-400)
        self.D = param.D  # Diffusion coefficient
        self.E = param.E  # Self-inhibition coefficient
        self.H = param.H  # Interaction coefficient
        self.w = np.array(param.w) * np.ones(A.shape[0])  # Node weights
        self.p1 = param.p1  # Density-dependent exponent
        
    def set_carrying_capacity(self, K_new):
        """
        Set a new environmental carrying capacity parameter
        Args:
            K_new: New environmental carrying capacity value (recommended integer between 1-400)
        """
        if K_new <= 0:
            raise ValueError("Environmental carrying capacity must be a positive number")
        self.K = K_new
        print(f"Environmental carrying capacity has been updated to: {self.K}")
        
    def get_carrying_capacity(self):
        """
        Get the current environmental carrying capacity parameter
        Returns:
            The current environmental carrying capacity value
        """
        return self.K
        
    def f(self, x, t):
        """
        Define the dynamic equation dx/dt
        Args:
            x: current state
            t: current time
        Returns:
            dxdt: state change rate
        """
        node_num = x.shape[0]
        dxdt = np.zeros(node_num)
        
        # Ensure state values are within a reasonable range
        x_safe = np.clip(x, 1e-10, 1e6)  # Avoid negative and excessively large values
        
        for i in range(node_num):
            try:
                # Intrinsic growth term - safe calculation
                if x_safe[i] < 0:
                    # If the state is negative, use a linear approximation
                    growth_term = self.B * x_safe[i]
                else:
                    # Normal calculation
                    growth_term = self.B * (x_safe[i] * (1 - (x_safe[i]**self.p1)/self.K))
                
                # Diffusion term - add safety checks
                diffusion_term = 0
                for j in range(node_num):
                    if self.A[i,j] != 0 and i != j:
                        # Ensure the denominator is not zero
                        denominator = max(self.D + self.E * x_safe[i] + self.H * x_safe[j], 1e-10)
                        diffusion_term += self.A[i,j] * ((x_safe[j] * x_safe[i]) / denominator)
                
                # Combine all terms and multiply by node weight
                dxdt[i] = self.w[i] * (growth_term + diffusion_term)
                
                # Check if the calculation result is reasonable
                if not np.isfinite(dxdt[i]):
                    print(f"Warning: The calculation result for node {i} is not a finite value")
                    dxdt[i] = 0.0  # Use a safe value
            
            except Exception as e:
                print(f"Error calculating node {i}: {str(e)}")
                print(f"Current state: x[{i}] = {x[i]}")
                dxdt[i] = 0.0  # Use a safe value
        
        # Final safety check
        if np.isnan(dxdt).any():
            nan_indices = np.where(np.isnan(dxdt))[0]
            print(f'Found nan values at nodes: {nan_indices}')
            print(f'State values at these nodes: {x[nan_indices]}')
            
            # Replace nan values instead of exiting
            dxdt[np.isnan(dxdt)] = 0.0
            print('Replaced nan values with 0.0 and continued simulation')
        
        # Limit the rate of change to avoid numerical instability
        dxdt = np.clip(dxdt, -1e3, 1e3)
            
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    

class NeuralNetworkModel:
    def __init__(self, args, A):
        """
        Initialize the neural network dynamics model
        Args:
            args: Configuration parameters
            A: Adjacency matrix
        """
        self.A = A
        
        param = args[args.dynamics]
        self.alpha = param.alpha  # Non-linear exponent
        self.B = param.B  # Decay coefficient
        self.s = param.s  # Self-activation strength
        self.g = param.g  # Coupling strength
        self.w = np.array(param.w) * np.ones(A.shape[0])  # Node weights
        
    def f(self, x, t):
        """
        Define the dynamic equation dx/dt
        Args:
            x: current state
            t: current time
        Returns:
            dxdt: state change rate
        """
        node_num = x.shape[0]
        dxdt = np.zeros(node_num)
        
        for i in range(node_num):
            # Self-activation term
            self_activation = -self.B * (x[i] ** self.alpha) + self.s * np.tanh(x[i])
            
            # Coupling term
            coupling_term = 0
            for j in range(node_num):
                if self.A[i,j] != 0 and i != j:
                    coupling_term += self.g * self.A[i,j] * np.tanh(x[j])
            
            # Combine all terms and multiply by node weight
            dxdt[i] = self.w[i] * (self_activation + coupling_term)
        
        if np.isnan(dxdt).any():
            print('nan during simulation!')
            exit()
            
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    

class PopulationDynamicsModel:
    def __init__(self, args, A):
        """
        Initialize the population dynamics model
        Args:
            args: Configuration parameters
            A: Adjacency matrix
        """
        self.A = A
        
        param = args[args.dynamics]
        self.R = param.R  # Growth rate
        self.B = param.B  # Death rate
        self.F = param.F  # Basic growth factor
        self.a1 = param.a1  # Growth exponent
        self.b1 = param.b1  # Death exponent
        # Ensure w is an array of the same length as the number of nodes
        self.w = np.array(param.w) * np.ones(A.shape[0])  # Node weights
    
    def f(self, x, t):
        """
        Define the dynamic equation dx/dt
        Args:
            x: current state
            t: current time
        Returns:
            dxdt: state change rate
        """
        node_num = x.shape[0]
        dxdt = np.zeros(node_num)
        
        for i in range(node_num):
            # Base growth and death terms
            base_term = self.F - self.B * (x[i] ** self.b1)
            
            # Interaction term
            interaction_term = 0
            for j in range(node_num):
                if self.A[i,j] != 0 and i != j:
                    interaction_term += self.A[i,j] * (self.R * (x[j] ** self.a1))
            
            # Combine all terms and multiply by node weight
            dxdt[i] = self.w[i] * (base_term + interaction_term)
        
        if np.isnan(dxdt).any():
            print('nan during simulation!')
            exit()
            
        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])
    
class HumanDynamicsModel:
    def __init__(self, args, A):
        """
        Initialize the human dynamics model
        Args:
            args: Configuration parameters
            A: Adjacency matrix (influence matrix)
        """
        self.A = A

        param = args[args.dynamics]  # Assume the dynamics parameter specifies human dynamics
        self.eta = param.eta # Self-inhibition coefficient
        self.mu = param.mu  # Linear influence coefficient
        self.rho = param.rho  # Saturation exponent
        self.w = np.array(param.w) * np.ones(A.shape[0])  # Node weights

    def f(self, x, t):
        """
        Define the human dynamics equation dx/dt
        Args:
            x: Current state (e.g., each person's opinion, behavior, etc.)
            t: Current time
        Returns:
            dxdt: State change rate
        """
        node_num = x.shape[0]
        dxdt = np.zeros(node_num)

        # Ensure state values are within a reasonable range to avoid numerical problems
        x_safe = np.clip(x, 1e-10, 1e6)

        for i in range(node_num):
            try:
                # Self-inhibition term
                suppression_term = - (x_safe[i] ** self.eta)

                # Linear influence and interaction terms
                interaction_term = 0
                for j in range(node_num):
                    if self.A[i, j] != 0:  # Include self-influence
                        # Avoid division by zero
                        denominator = 1 + x_safe[j] ** self.rho
                        interaction_term += self.A[i, j] * (x_safe[j] ** self.rho) / denominator

                # Sum of all terms
                dxdt[i] = self.w[i] * (suppression_term + self.mu + interaction_term)

                # Check the calculation result
                if not np.isfinite(dxdt[i]):
                    print(f"Warning: The calculation result for node {i} is not a finite value")
                    dxdt[i] = 0.0

            except Exception as e:
                print(f"Error calculating node {i}: {str(e)}")
                print(f"Current state: x[{i}] = {x[i]}")
                dxdt[i] = 0.0

        # NaN value check and handling
        if np.isnan(dxdt).any():
            nan_indices = np.where(np.isnan(dxdt))[0]
            print(f'Found nan values at nodes: {nan_indices}')
            print(f'State values at these nodes: {x[nan_indices]}')
            dxdt[np.isnan(dxdt)] = 0.0
            print('Replaced nan values with 0.0 and continued simulation')

        # Limit the rate of change
        dxdt = np.clip(dxdt, -1e3, 1e3)

        return dxdt
    
    def g(self, x, t):
        """inherent noise"""
        return np.diag([0.0] * x.shape[0])

import numpy as np

class ConstructDynamicsModel:
    def __init__(self,A,config):
        """
        Initialize the human dynamics model
        Args:
            args: Configuration parameters
            A: Adjacency matrix (influence matrix)
        """
        self.A = A
        # Use parameters from equation (99)
        self.a=config['dynamics']['a']
        self.h=config['dynamics']['h']
        self.B=config['dynamics']['B']
   


    def f(self, x, t):
        """
        Define the human dynamics equation dx/dt, corresponding to equation (99)
        Args:
            x: Current state (e.g., each person's opinion, behavior, etc.)
            t: Current time
        Returns:
            dxdt: State change rate
        """
        node_num = x.shape[0]
        dxdt = np.zeros(node_num)

        # Ensure state values are within a reasonable range to avoid numerical problems
        x_safe = np.clip(x, 1e-10, 1e6)

        for i in range(node_num):
            try:
                # Self-decay term (corresponding to the first term of equation 99)
                suppression_term = -self.B * (x_safe[i] ** self.a)

                # Interaction term (corresponding to the second term of equation 99, the summation part)
                interaction_term = 0
                for j in range(node_num):
                    if self.A[i, j] != 0: # This judgment is not needed, because when A[i,j] is 0, the contribution is also 0
                        # Avoid division by zero
                        denominator = 1 + x_safe[j] ** self.h
                        interaction_term += (x_safe[j] ** self.h) / denominator

                # Sum of all terms
                dxdt[i] = suppression_term + interaction_term

                # Check the calculation result
                if not np.isfinite(dxdt[i]):
                    print(f"Warning: The calculation result for node {i} is not a finite value")
                    dxdt[i] = 0.0

            except Exception as e:
                print(f"Error calculating node {i}: {str(e)}")
                print(f"Current state: x[{i}] = {x[i]}")
                dxdt[i] = 0.0

        # NaN value check and handling
        if np.isnan(dxdt).any():
            nan_indices = np.where(np.isnan(dxdt))[0]
            print(f'Found nan values at nodes: {nan_indices}')
            print(f'State values at these nodes: {x[nan_indices]}')
            dxdt[np.isnan(dxdt)] = 0.0
            print('Replaced nan values with 0.0 and continued simulation')
        # Limit the rate of change
        dxdt = np.clip(dxdt, -1e3, 1e3)
        return dxdt
    def M0(self,x):
        return -self.B * (x ** self.a)
    def M1(self,x):
        return 1
    def M2(self,x):
        denominator = 1 + x ** self.h
        return (x ** self.h) / denominator


class KuramotoModel:
    def __init__(self, args, A):
        """
        Initialize the Kuramoto oscillator model
        Args:
            args: Configuration parameters
            A: Adjacency matrix
        """
        self.A = A
        self.node_num = A.shape[0]
        
        param = args[args.dynamics]
        self.K = param.K  # Coupling strength
        self.sigma = getattr(param, 'sigma', 0.1)  # Standard deviation of natural frequencies
        self.w_mean = getattr(param, 'w_mean', 1.0)  # Mean of natural frequencies
        
        # Calculate node degrees for normalization
        self.degrees = np.sum(A, axis=1)
        # Avoid division by zero error
        self.degrees = np.where(self.degrees == 0, 1, self.degrees)
        
        # Generate natural frequencies for each oscillator (sampled from a normal distribution)
        np.random.seed(getattr(param, 'seed', 42))
        self.omega = np.random.normal(self.w_mean, self.sigma, self.node_num)
        
        print(f"Kuramoto model initialization complete:")
        print(f"  - Number of nodes: {self.node_num}")
        print(f"  - Coupling strength K: {self.K}")
        print(f"  - Mean of natural frequencies: {self.w_mean}")
        print(f"  - Standard deviation of natural frequencies: {self.sigma}")
        print(f"  - Average degree: {np.mean(self.degrees):.2f}")
        
    def f(self, x, t):
        """
        Define the dynamic equation of the Kuramoto model
        dθ_i/dt = ω_i + (K/k_i) * Σ_j A_ij * sin(θ_j - θ_i)
        
        Args:
            x: Current phase state θ (each element represents the phase of an oscillator)
            t: Current time
        Returns:
            dxdt: Phase change rate
        """
        dxdt = np.zeros(self.node_num)
        
        for i in range(self.node_num):
            # Natural frequency term
            dxdt[i] = self.omega[i]
            
            # Coupling term: K/k_i * Σ_j A_ij * sin(θ_j - θ_i)
            coupling_term = 0.0
            for j in range(self.node_num):
                if self.A[i, j] != 0:
                    # sin(θ_j - θ_i)
                    phase_diff = x[j] - x[i]
                    coupling_term += self.A[i, j] * np.sin(phase_diff)
            
            # Normalization: divide by node degree
            coupling_term = (self.K / self.degrees[i]) * coupling_term
            dxdt[i] += coupling_term
        
        # Numerical stability check
        if np.isnan(dxdt).any():
            nan_indices = np.where(np.isnan(dxdt))[0]
            print(f'Found nan values in Kuramoto model calculation at nodes: {nan_indices}')
            print(f'Phase values at these nodes: {x[nan_indices]}')
            dxdt[np.isnan(dxdt)] = 0.0
            print('Replaced nan values with 0.0 and continued simulation')
        
        return dxdt
    
    def g(self, x, t):
        """inherent noise (no noise added for now)"""
        return np.diag([0.0] * x.shape[0])
    
    def calculate_order_parameter(self, x):
        """
        Calculate the Kuramoto order parameter r = |1/N * Σ_j e^(iθ_j)|
        Used to measure the degree of synchronization of oscillators
        
        Args:
            x: Phase state array
        Returns:
            r: Order parameter (0 means completely asynchronous, 1 means completely synchronous)
            psi: Average phase
        """
        complex_sum = np.mean(np.exp(1j * x))
        r = np.abs(complex_sum)
        psi = np.angle(complex_sum)
        return r, psi
    
    def get_frequency_statistics(self):
        """Return statistics of natural frequencies"""
        return {
            'omega_mean': np.mean(self.omega),
            'omega_std': np.std(self.omega),
            'omega_min': np.min(self.omega),
            'omega_max': np.max(self.omega)
        }