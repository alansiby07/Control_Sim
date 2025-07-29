import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
from matplotlib.widgets import Slider
from scipy.optimize import minimize, differential_evolution

# ----------------------- CONFIGURATION -----------------------
class SimConfig:
    """Centralized configuration for simulation parameters"""
    # Physical parameters
    MASS = 10.0
    SIDE_LENGTH = 1.0
    RW_MASS = 1.0
    RADIUS = 0.2
    I_SC = 1000.0
    I_RW = 5.0
    
    # Control limits
    MAX_WHEEL_SPEED = 150000.0 * 2 * np.pi / 60  # RPM to rad/s
    TORQ_MAX = 15.0
    MAX_ACC = 10.0
    
    # Default controller gains
    K1_DEFAULT = 4.0
    K2_DEFAULT = 10.0
    
    # Initial conditions
    INITIAL_THETA = np.deg2rad(10)
    INITIAL_VEL = 0.0
    
    # Time parameters
    T_END = 10.0
    N_POINTS = 1000
    
    # Optimization parameters
    SETTLING_TOLERANCE = 0.02
    MAX_SETTLING_TIME = 5.0
    MAX_OVERSHOOT_DEG = 2.0

# ----------------------- CORE SIMULATION -----------------------
class SpacecraftController:
    """Encapsulates spacecraft dynamics and control logic"""
    
    def __init__(self, config=None):
        self.config = config or SimConfig()
        self._cache = {}  # For memoization
    
    def solve_ode_vectorized(self, k1, k2, t, initial_theta=None, initial_vel=None):
        """Optimized ODE solver using vectorized operations"""
        if initial_theta is None:
            initial_theta = self.config.INITIAL_THETA
        if initial_vel is None:
            initial_vel = self.config.INITIAL_VEL
        
        # Cache key for memoization
        cache_key = (k1, k2, len(t), initial_theta, initial_vel)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Characteristic equation coefficients
        a, b, c = 1, k2 / self.config.I_SC, k1 / self.config.I_SC
        
        # Compute roots
        discriminant = b*b - 4*a*c
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            lam1 = (-b + sqrt_disc) / (2*a)
            lam2 = (-b - sqrt_disc) / (2*a)
        else:
            real_part = -b / (2*a)
            imag_part = np.sqrt(-discriminant) / (2*a)
            lam1 = complex(real_part, imag_part)
            lam2 = complex(real_part, -imag_part)
        
        # Compute coefficients A and B
        if lam1 != lam2:
            A = (initial_vel - lam2 * initial_theta) / (lam1 - lam2)
        else:  # Repeated roots case
            A = initial_theta
        B = initial_theta - A
        
        # Vectorized solution
        exp1 = np.exp(lam1 * t)
        exp2 = np.exp(lam2 * t)
        
        theta = A * exp1 + B * exp2
        ang_vel = A * lam1 * exp1 + B * lam2 * exp2
        
        result = (theta.real, ang_vel.real)
        self._cache[cache_key] = result
        return result
    
    def compute_control_torque(self, theta, ang_vel, k1, k2):
        """Compute control torque with saturation"""
        torque = -k1 * theta - k2 * ang_vel
        return np.clip(torque, -self.config.TORQ_MAX, self.config.TORQ_MAX)
    
    def evaluate_performance(self, k1, k2, t):
        """Comprehensive performance evaluation"""
        try:
            theta, ang_vel = self.solve_ode_vectorized(k1, k2, t)
        except:
            return None
        
        # Check stability
        if not self._is_stable(k1, k2):
            return None
        
        # Performance metrics
        final_theta = theta[-1]
        tolerance = max(self.config.SETTLING_TOLERANCE * abs(final_theta), np.deg2rad(0.5))
        
        # Settling time calculation
        within_band = np.abs(theta - final_theta) < tolerance
        settling_time = self._compute_settling_time(t, within_band)
        
        # Overshoot calculation
        peak_theta = np.max(theta)
        overshoot_deg = np.rad2deg(max(0, peak_theta - final_theta))
        
        # Energy metric
        torque = self.compute_control_torque(theta, ang_vel, k1, k2)
        energy_metric = np.sqrt(np.mean(torque**2))
        
        return {
            'k1': k1, 'k2': k2,
            'settling_time': settling_time,
            'overshoot_deg': overshoot_deg,
            'energy_metric': energy_metric,
            'theta': theta,
            'ang_vel': ang_vel,
            'torque': torque
        }
    
    def _is_stable(self, k1, k2):
        """Check system stability"""
        a, b, c = 1, k2 / self.config.I_SC, k1 / self.config.I_SC
        discriminant = b*b - 4*a*c
        
        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            roots = [(-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)]
        else:
            real_part = -b / (2*a)
            roots = [real_part, real_part]  # Both roots have same real part
        
        return all(np.real(root) < 0 for root in roots)
    
    def _compute_settling_time(self, t, within_band):
        """Optimized settling time computation"""
        if np.all(within_band):
            return t[0]
        
        outside_indices = np.where(~within_band)[0]
        if len(outside_indices) == 0:
            return np.nan
        
        settle_index = min(outside_indices[-1] + 1, len(t) - 1)
        return t[settle_index]

# ----------------------- OPTIMIZATION ENGINE -----------------------
class GainOptimizer:
    """Optimized gain finding with multiple strategies"""
    
    def __init__(self, controller, config=None):
        self.controller = controller
        self.config = config or SimConfig()
    
    def find_optimal_gains(self, k1_range=None, k2_range=None, t=None):
        """Main optimization routine"""
        if k1_range is None:
            k1_range = np.linspace(0.1, 300, 100)  # Reduced resolution
        if k2_range is None:
            k2_range = np.linspace(0.1, 300, 100)
        if t is None:
            t = np.linspace(0, self.config.T_END, self.config.N_POINTS)
        
        print("Starting gain optimization...")
        
        # Strategy 1: Coarse grid search
        best_grid = self._grid_search_coarse(k1_range, k2_range, t)
        
        # Strategy 2: Fine optimization around best region
        if best_grid:
            best_optimized = self._local_optimization(best_grid, t)
            return best_optimized or best_grid
        
        return None
    
    def _grid_search_coarse(self, k1_range, k2_range, t):
        """Efficient coarse grid search"""
        best_result = None
        best_cost = float('inf')
        valid_count = 0
        
        # Use meshgrid for vectorized evaluation where possible
        for k1 in k1_range[::2]:  # Skip every other point for speed
            for k2 in k2_range[::2]:
                result = self.controller.evaluate_performance(k1, k2, t)
                
                if result is None:
                    continue
                
                # Multi-objective cost function
                cost = self._compute_cost(result)
                
                if (result['settling_time'] < self.config.MAX_SETTLING_TIME and 
                    result['overshoot_deg'] < self.config.MAX_OVERSHOOT_DEG and
                    cost < best_cost):
                    
                    best_cost = cost
                    best_result = result
                    valid_count += 1
        
        print(f"Grid search found {valid_count} valid candidates")
        return best_result
    
    def _local_optimization(self, initial_result, t):
        """Local optimization around promising region"""
        k1_init, k2_init = initial_result['k1'], initial_result['k2']
        
        def objective(gains):
            k1, k2 = gains
            if k1 <= 0 or k2 <= 0:
                return 1e6
            
            result = self.controller.evaluate_performance(k1, k2, t)
            return self._compute_cost(result) if result else 1e6
        
        bounds = [(max(0.1, k1_init - 50), k1_init + 50),
                  (max(0.1, k2_init - 50), k2_init + 50)]
        
        try:
            opt_result = minimize(objective, [k1_init, k2_init], 
                                bounds=bounds, method='L-BFGS-B')
            
            if opt_result.success:
                k1_opt, k2_opt = opt_result.x
                return self.controller.evaluate_performance(k1_opt, k2_opt, t)
        except:
            pass
        
        return None
    
    def _compute_cost(self, result):
        """Multi-objective cost function"""
        if result is None:
            return float('inf')
        
        w_settle, w_overshoot, w_energy = 1.0, 2.0, 0.1
        
        settling_penalty = result['settling_time'] if result['settling_time'] < 10.0 else result['settling_time'] * 2
        overshoot_penalty = result['overshoot_deg'] if result['overshoot_deg'] < 5.0 else result['overshoot_deg'] * 3
        energy_penalty = result['energy_metric'] / self.config.TORQ_MAX
        
        return (w_settle * settling_penalty + 
                w_overshoot * overshoot_penalty + 
                w_energy * energy_penalty)

# ----------------------- VISUALIZATION -----------------------
class Plotter:
    """Streamlined plotting functionality"""
    
    def __init__(self, config=None):
        self.config = config or SimConfig()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_time_domain(self, result, save_path=None):
        """Optimized time domain plotting"""
        t = np.linspace(0, self.config.T_END, len(result['theta']))
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Plot data
        plot_data = [
            (np.rad2deg(result['theta']), 'Angle θ (deg)', 'blue'),
            (np.rad2deg(result['ang_vel']), 'Angular Velocity ω (deg/s)', 'green'),
            (result['torque'], 'Control Torque τ (Nm)', 'red')
        ]
        
        for i, (data, label, color) in enumerate(plot_data):
            axes[i].plot(t, data, label=label, color=color)
            axes[i].set_ylabel(label.split(' ')[0] + ' ' + label.split(' ')[-1])
            axes[i].grid(True)
            axes[i].legend()
        
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.suptitle(f"Spacecraft Control Response (k1={result['k1']:.2f}, k2={result['k2']:.2f})", 
                     fontsize=14, y=1.02)
        
        if save_path:
            self._save_plot(save_path, "time_domain")
        
        plt.show()
    
    def plot_root_locus_simple(self, i_sc, k_fixed=5.0, mode='k1', save_path=None):
        """Simplified root locus plot"""
        k_vals = np.linspace(0.1, 100.0, 200)
        roots = []
        
        for k in k_vals:
            if mode == 'k1':
                a, b, c = 1, k_fixed / i_sc, k / i_sc
            else:
                a, b, c = 1, k / i_sc, k_fixed / i_sc
            
            discriminant = b*b - 4*a*c
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                root_pair = [(-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)]
            else:
                real_part = -b / (2*a)
                imag_part = np.sqrt(-discriminant) / (2*a)
                root_pair = [complex(real_part, imag_part), complex(real_part, -imag_part)]
            
            roots.append(root_pair)
        
        roots = np.array(roots)
        
        plt.figure(figsize=(10, 6))
        colors = ['tab:blue', 'tab:orange']
        
        for i in range(2):
            plt.plot(roots[:, i].real, roots[:, i].imag, 
                    color=colors[i], label=f'Root {i+1}')
        
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
        plt.xlabel("Real Axis")
        plt.ylabel("Imaginary Axis")
        plt.title(f"Root Locus: Sweeping {mode.upper()} (fixed {['k2','k1'][mode=='k1']}={k_fixed})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            self._save_plot(f"{save_path}_root_locus_{mode}", "root_locus")
        
        plt.show()
    
    def _save_plot(self, filename, subfolder):
        """Centralized plot saving"""
        folder_path = os.path.join("plots", subfolder)
        os.makedirs(folder_path, exist_ok=True)
        filepath = os.path.join(folder_path, f"{filename}_{self.timestamp}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')

# ----------------------- MAIN SIMULATION -----------------------
def main():
    """Streamlined main simulation function"""
    config = SimConfig()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"simulation_output_{timestamp}.txt")
    os.makedirs("logs", exist_ok=True)
    
    # Redirect stdout
    original_stdout = sys.stdout
    
    try:
        with open(log_path, "w") as log_file:
            sys.stdout = log_file
            
            print("=== Spacecraft Control Simulation ===")
            print(f"Timestamp: {timestamp}")
            print(f"System parameters: I_sc={config.I_SC}, T_max={config.TORQ_MAX}")
            
            # Initialize components
            controller = SpacecraftController(config)
            optimizer = GainOptimizer(controller, config)
            plotter = Plotter(config)
            
            # Time vector
            t = np.linspace(0, config.T_END, config.N_POINTS)
            
            # Find optimal gains
            best_result = optimizer.find_optimal_gains(t=t)
            
            if best_result:
                print(f"\n=== OPTIMAL GAINS FOUND ===")
                print(f"k1 = {best_result['k1']:.3f}")
                print(f"k2 = {best_result['k2']:.3f}")
                print(f"Settling time: {best_result['settling_time']:.2f}s")
                print(f"Overshoot: {best_result['overshoot_deg']:.2f}°")
                print(f"Energy metric: {best_result['energy_metric']:.3f}")
                
            else:
                print("No suitable gains found with current criteria")
                
                # Use default gains for demonstration
                best_result = controller.evaluate_performance(
                    config.K1_DEFAULT, config.K2_DEFAULT, t)
                
                if best_result:
                    print(f"\nUsing default gains: k1={config.K1_DEFAULT}, k2={config.K2_DEFAULT}")
    
    finally:
        sys.stdout = original_stdout
        print(f"Simulation completed. Results saved to {log_path}")
        
        # Generate plots (restore stdout for interactive plotting)
        if 'best_result' in locals() and best_result:
            controller = SpacecraftController(config)
            plotter = Plotter(config)
            
            plotter.plot_time_domain(best_result)
            plotter.plot_root_locus_simple(config.I_SC, best_result['k2'], 'k1')
            plotter.plot_root_locus_simple(config.I_SC, best_result['k1'], 'k2')

if __name__ == "__main__":
    main()
