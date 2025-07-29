import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.widgets import Slider
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

#------------------------MAIN SIMULATION FUNCTION-------------------

def main():
    # Creates timestamp for when the simulation was run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"simulation_output_{timestamp}.txt"

    # Makes a file path and folder for the logs to go into when the simulation is run
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file_name)

    # Store original stdout
    original_stdout = sys.stdout

    try:
        with open(log_path, "w") as log_file:
            sys.stdout = log_file

            # Physical parameters
            mass = 10.0  # kg
            side_length = 1.0  # m
            rw_mass = 1.0  # kg
            radius = 0.2  # m

            # Moment of inertia
            i_sc = 10.0  # kg⋅m^2
            i_rw = 0.5  # kg⋅m^2

            # Actuator limits
            max_wheel_speed = 500.0 * 2 * np.pi / 60  # Convert RPM to rad/s
            torque_max = 1.0  # Nm
            max_acc = 10.0  # rad/s^2

            # Controller Gains
            k1 = 4.0
            k2 = 10.0

            # Initial States
            initial_theta = np.deg2rad(10)  # rad
            initial_vel = 0.0  # rad/s

            # Time Setup (in seconds)
            dt = 0.001  # 1ms timestep
            t_end = 10.0  # 10 seconds
            t = np.arange(0, t_end, dt)

            print(f"Running simulation with dt={dt}s, duration={t_end}s")
            print(f"Controller gains: k1={k1}, k2={k2}")
            print(f"Initial conditions: theta={np.rad2deg(initial_theta):.1f}°, omega={initial_vel}°/s")

            # Run main simulation
            theta, ang_vel, torque_history, wheel_speed_history = simulate_response(
                k1, k2, t, initial_theta, initial_vel, i_sc,
                torque_max, max_wheel_speed, i_rw, max_acc
            )

            # Generate gain sweep ranges
            # Estimate gains that are needed
            print("=== GAIN ESTIMATION ===")
            k1_est, k2_est = estimate_required_gains(i_sc, desired_settling_time=15.0, desired_damping_ratio=0.7)
           
            # Generate gain sweep ranges based on estimates
            k1_min, k1_max = max(50.0, k1_est * 0.2), k1_est * 2.0    
            k2_min, k2_max = max(20.0, k2_est * 0.2), k2_est * 2.0   

            k1_values = np.linspace(k1_min, k1_max, 15)  # Reduced points for efficiency
            k2_values = np.linspace(k2_min, k2_max, 15)

            print(f"Searching k1 range: {k1_min:.1f} to {k1_max:.1f}")
            print(f"Searching k2 range: {k2_min:.1f} to {k2_max:.1f}")

            # Plot results for current gains
            plot_all(t, theta, ang_vel, torque_history, k1=k1, k2=k2, i_sc=i_sc)

            # Find optimal gains
            print("\nSearching for optimal gains...")
            valid_gains, best_gain, overshoot_list, settling_time_list = valid_gains_finder(
                k1_range=k1_values, k2_range=k2_values, t=t,
                initial_theta=initial_theta, initial_vel=initial_vel,
                i_sc=i_sc, torque_max=torque_max, rw_speed_max=max_wheel_speed,
                i_rw=i_rw, max_acc=max_acc,
                overshoot_limit=2.0,
                settling_time_limit=5.0,
                w_settle=1.0, w_overshoot=1.0
            )

            if best_gain is not None:
                k1_best, k2_best, ts_best, os_best = best_gain
                print(f"\nGenerating plots for best gains: k1={k1_best:.2f}, k2={k2_best:.2f}")
                print(f"Performance: settling_time={ts_best:.2f}s, overshoot={os_best:.2f}°")

                # Simulate with best gains
                theta_best, ang_vel_best, torque_best, wheel_speed_best = simulate_response(
                    k1_best, k2_best, t, initial_theta, initial_vel, i_sc,
                    torque_max, max_wheel_speed, i_rw, max_acc
                )

                # Plot best gain results
                best_gain_plots(t, theta_best, ang_vel_best, torque_best,
                              k1_best, k2_best, i_sc)
            else:
                print("No valid gains found - skipping best gain plots")

    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        print(f"Simulation completed. Results saved to {log_path}")


#---------------Main Simulation Function--------------

def simulate_response(k1, k2, t, initial_theta, initial_vel, i_sc,
                      torque_max, rw_speed_max, i_rw, max_acc):
    """Simulate spacecraft attitude control response using RK4 integration"""

    dt = t[1] - t[0]
    n_steps = len(t)

    # Pre-allocate arrays for efficiency
    theta = np.zeros(n_steps)
    ang_vel = np.zeros(n_steps)
    torque_history = np.zeros(n_steps)
    wheel_speed_history = np.zeros(n_steps)

    # Initial conditions
    theta[0] = initial_theta
    ang_vel[0] = initial_vel
    wheel_speed = 0.0  # Initial wheel speed (rad/s)

    for i in range(1, n_steps):
        # Calculate control torque and wheel acceleration
        torque_actual, rw_acc = control_calc(
            theta[i-1], ang_vel[i-1], k1, k2, i_rw, torque_max, max_acc
        )

        # Update wheel speed with saturation
        wheel_speed_new = wheel_speed + rw_acc * dt
        wheel_speed = np.clip(wheel_speed_new, -rw_speed_max, rw_speed_max)

        # If wheel hits speed limit, recalculate actual torque
        if abs(wheel_speed_new) > rw_speed_max:
            # Wheel can't accelerate further in this direction
            if (wheel_speed >= rw_speed_max and rw_acc > 0) or \
               (wheel_speed <= -rw_speed_max and rw_acc < 0):
                rw_acc = 0
                torque_actual = 0

        
        # RK4 steps
        th0, vel0 = theta[i-1], ang_vel[i-1]

        k1_th, k1_vel = spacecraft_dynamics(th0, vel0, torque_actual, i_sc)
        k2_th, k2_vel = spacecraft_dynamics(th0 + 0.5 * dt * k1_th, vel0 + 0.5 * dt * k1_vel, torque_actual, i_sc)
        k3_th, k3_vel = spacecraft_dynamics(th0 + 0.5 * dt * k2_th, vel0 + 0.5 * dt * k2_vel, torque_actual, i_sc)
        k4_th, k4_vel = spacecraft_dynamics(th0 + dt * k3_th, vel0 + dt * k3_vel, torque_actual, i_sc)

        # Update states
        theta[i] = th0 + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        ang_vel[i] = vel0 + (dt / 6.0) * (k1_vel + 2 * k2_vel + 2 * k3_vel + k4_vel)

        # Store history
        torque_history[i-1] = torque_actual
        wheel_speed_history[i-1] = wheel_speed

    # Store final values
    torque_history[-1] = torque_actual
    wheel_speed_history[-1] = wheel_speed

    return theta, ang_vel, torque_history, wheel_speed_history


#---------------Saving Plots Function--------------

def save_plot(filename, subfolder="misc"):
    """Creates directory structure and saves plots"""
    base_folder = "plots"
    folder_path = os.path.join(base_folder, subfolder)
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filepath}")


#---------------Control Calculation Function--------------

def control_calc(theta, ang_vel, k1, k2, i_rw, torque_max, max_acc):
    """Calculate control torque and reaction wheel acceleration"""
    if i_rw <= 0:
        raise ValueError("Reaction wheel inertia must be positive")

    # PD control law
    torque_cmd = -k1 * theta - k2 * ang_vel

    # Apply torque saturation
    torque = np.clip(torque_cmd, -torque_max, torque_max)

    # Calculate reaction wheel acceleration (Newton's 2nd law for rotation)
    rw_acc = -torque / i_rw  # Negative due to action-reaction principle

    # Apply acceleration limits
    rw_acc = np.clip(rw_acc, -max_acc, max_acc)

    # Actual torque based on limited acceleration
    torque_actual = -rw_acc * i_rw

    return torque_actual, rw_acc

# RK4 integration for spacecraft dynamics
def spacecraft_dynamics(theta_val, ang_vel_val, torque_actual, i_sc):
    """Spacecraft equations of motion"""
    ang_acc = torque_actual / i_sc
    return ang_vel_val, ang_acc



#----------------------Optimal Gain Finder Function-------------------------

def evaluate_performance(theta, t, tolerance_factor=0.02, min_tolerance_deg=0.5):
    """Evaluate settling time and overshoot for a response"""

    # Convert to degrees for analysis
    theta_deg = np.rad2deg(theta)

    # Check for divergence or invalid values
    if np.any(np.isnan(theta_deg)) or np.any(np.isinf(theta_deg)):
        return 0.0, float('inf')  # Invalid response

    final_theta_deg = theta_deg[-1]

    # If final value is too large, system likely diverged
    if abs(final_theta_deg) > 1000:
        return 0.0, float('inf')

    # Calculate overshoot
    peak_theta_deg = np.max(theta_deg)
    overshoot_deg = max(0, peak_theta_deg - final_theta_deg)

    # Calculate settling time (2% criterion or 0.5°, whichever is larger)
    tolerance_deg = max(tolerance_factor * abs(final_theta_deg), min_tolerance_deg)

    # Find last time outside tolerance band
    outside_band = np.abs(theta_deg - final_theta_deg) > tolerance_deg

    if np.any(outside_band):
        last_out_idx = np.where(outside_band)[0][-1]
        settling_time = t[min(last_out_idx + 1, len(t) - 1)]
    else:
        settling_time = 0.0

    return settling_time, overshoot_deg


def simulate_single_case(args):
    """Worker function for parallel gain evaluation"""
    (k1, k2, t, initial_theta, initial_vel, i_sc, torque_max,
     rw_speed_max, i_rw, max_acc, overshoot_limit, settling_time_limit,
     w_settle, w_overshoot) = args

    try:
        # Reject negative damping immediately
        if k2 <= 0:
            return k1, k2, 0, 0, float('inf')

        # Run simulation
        theta, ang_vel, torque_history, _ = simulate_response(
            k1, k2, t, initial_theta, initial_vel, i_sc,
            torque_max, rw_speed_max, i_rw, max_acc
        )

        # checks for any divergence
        theta_deg = np.rad2deg(theta)
        if np.any(np.abs(theta_deg) > 1000) or np.any(np.isnan(theta_deg)) or np.any(np.isinf(theta_deg)):
            return k1, k2, 0, 0, float('inf')

        # Performance evaluation
        settling_time_sim, overshoot_sim_deg = evaluate_performance(theta, t)

        # If settling time is unreasonably large, it will reject early
        if settling_time_sim > settling_time_limit * 3:
            return k1, k2, settling_time_sim, overshoot_sim_deg, float('inf')

        # Use simulation results directly
        settling_time = settling_time_sim
        overshoot_deg = overshoot_sim_deg

        # Cost function
        if settling_time <= 0 or settling_time > settling_time_limit * 2:
            cost = float('inf')
        else:
            # Normalize costs appropriately
            settling_cost = w_settle * (settling_time / settling_time_limit)
            overshoot_cost = w_overshoot * (overshoot_deg / overshoot_limit)

            # Reduce saturation penalty since we need higher gains
            torque_saturation_penalty = np.mean(np.abs(torque_history) >= torque_max * 0.95) * 1.0

            # Adjust gain penalties for higher expected gains
            k1_penalty = 0.001 * (k1 / 500.0)**2 if k1 > 500 else 0
            k2_penalty = 0.001 * (k2 / 200.0)**2 if k2 > 200 else 0

            cost = settling_cost + overshoot_cost + torque_saturation_penalty + k1_penalty + k2_penalty

        return k1, k2, settling_time, overshoot_deg, cost

    except Exception as e:
        print(f"Error simulating k1={k1:.2f}, k2={k2:.2f}: {e}")
        return k1, k2, 0, 0, float('inf')


def valid_gains_finder(k1_range, k2_range, t, initial_theta, initial_vel, i_sc,
                       torque_max, rw_speed_max, i_rw, max_acc,
                       overshoot_limit=2.0, settling_time_limit=5.0,
                       w_settle=1.0, w_overshoot=1.0, verbose=True):
    """Find optimal controller gains using parallel grid search"""

    print(f"Searching {len(k1_range)} x {len(k2_range)} = {len(k1_range) * len(k2_range)} gain combinations")

    # Prepare tasks for parallel execution
    tasks = []
    for k1 in k1_range:
        for k2 in k2_range:
            task = (k1, k2, t, initial_theta, initial_vel, i_sc, torque_max,
                   rw_speed_max, i_rw, max_acc, overshoot_limit, settling_time_limit,
                   w_settle, w_overshoot)
            tasks.append(task)

    # Run parallel evaluation
    valid_gains = []
    best_gain = None
    best_cost = float('inf')
    overshoot_list = []
    settling_time_list = []

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(simulate_single_case, tasks))

    # Process results
    for k1, k2, ts, os, cost in results:
        if ts > 0 and cost < float('inf'):
            # Check if gains meet criteria
            if ts <= settling_time_limit and os <= overshoot_limit:
                valid_gains.append((k1, k2, ts, os))
                overshoot_list.append(os)
                settling_time_list.append(ts)

                if verbose and len(valid_gains) <= 10:  # Limit verbose output
                    print(f"Valid: k1={k1:.2f}, k2={k2:.2f}, ts={ts:.2f}s, os={os:.2f}°")

            # Track best overall
            if cost < best_cost:
                best_cost = cost
                best_gain = (k1, k2, ts, os)

    if verbose:
        print(f"\nFound {len(valid_gains)} valid gain combinations")
        if best_gain:
            k1, k2, ts, os = best_gain
            print(f"Best overall: k1={k1:.2f}, k2={k2:.2f}, ts={ts:.2f}s, os={os:.2f}°")

    return valid_gains, best_gain, overshoot_list, settling_time_list


def estimate_required_gains(i_sc, desired_settling_time=5.0, desired_damping_ratio=0.7):
    """
    Estimate required gains for desired performance
    For second-order system: s² + (k2/I)s + (k1/I) = 0
    """
    # Desired natural frequency from settling time
    omega_n = 4.6 / (desired_damping_ratio * desired_settling_time)

    # Required gains
    k1_est = omega_n**2 * i_sc
    k2_est = 2 * desired_damping_ratio * omega_n * i_sc

    print(f"Estimated gains for ts={desired_settling_time}s, ζ={desired_damping_ratio}:")
    print(f"k1 ≈ {k1_est:.1f}")
    print(f"k2 ≈ {k2_est:.1f}")

    return k1_est, k2_est


#---------------Root Locus Functions----------------------

def compute_roots(i_sc, k1, k2):
    """Calculate roots of characteristic equation: s² + (k2/I)s + (k1/I) = 0"""
    if i_sc <= 0:
        raise ValueError("Spacecraft inertia must be positive")

    # Characteristic polynomial coefficients
    coeffs = [1, k2 / i_sc, k1 / i_sc]
    return np.roots(coeffs)


def generate_root_locus_sweep(i_sc, k_fixed, sweep_vals, mode='k1'):
    """Generate root locus data for plotting"""
    all_roots = []
    for k in sweep_vals:
        if mode == 'k1':
            roots = compute_roots(i_sc, k, k_fixed)
        else:  # mode == 'k2'
            roots = compute_roots(i_sc, k_fixed, k)
        all_roots.append(roots)
    return np.array(all_roots)




#----------------Plotting Functions----------------------

def plot_all(t, theta, ang_vel, torque, k1, k2, i_sc):
    """Generate all plots for current simulation"""
    plot_time_domain(t, theta, ang_vel, torque, k1=k1, k2=k2, subfolder="time_domain")
    plot_root_locus(i_sc, mode='k1', k_fixed=k2, subfolder="root_locus")
    plot_root_locus(i_sc, mode='k2', k_fixed=k1, subfolder="root_locus")
    analysis(t, theta, k1=k1, k2=k2)


def plot_time_domain(t, theta, ang_vel, torque, k1, k2, subfolder="time_domain"):
    """Plot time domain response"""

    # Print summary data
    print(f"\nTime-Domain Results for k1={k1}, k2={k2}")
    print("Time (s) | Angle (deg) | Angular Velocity (deg/s) | Torque (Nm)")
    print("-" * 65)
    for i in range(0, len(t), max(1, len(t)//10)):
        print(f"{t[i]:6.2f} | {np.rad2deg(theta[i]):11.2f} | "
              f"{np.rad2deg(ang_vel[i]):23.2f} | {torque[i]:10.2f}")

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Angle plot
    axes[0].plot(t, np.rad2deg(theta), 'b-', linewidth=2, label='Angle θ (deg)')
    axes[0].set_ylabel("Angle (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title(f"Spacecraft Attitude Control Response (k1={k1}, k2={k2})")

    # Angular velocity plot
    axes[1].plot(t, np.rad2deg(ang_vel), 'g-', linewidth=2, label='Angular Velocity ω (deg/s)')
    axes[1].set_ylabel("Angular Velocity (deg/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Torque plot
    axes[2].plot(t, torque, 'r-', linewidth=2, label='Control Torque τ (Nm)')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"time_domain_k1_{k1}_k2_{k2}_{timestamp}.png", subfolder=subfolder)
    plt.show()


def analysis(t, theta, k1, k2, subfolder="analysis"):
    """Analyze and plot settling time and overshoot"""

    settling_time, overshoot_deg = evaluate_performance(theta, t)

    theta_deg = np.rad2deg(theta)
    final_deg = theta_deg[-1]
    tolerance_deg = max(0.02 * abs(final_deg), 0.5)

    plt.figure(figsize=(12, 8))
    plt.plot(t, theta_deg, 'b-', linewidth=2, label='θ(t) [deg]')
    plt.axhline(final_deg, linestyle='--', color='gray', alpha=0.7, label='Final Value')
    plt.axhline(final_deg + tolerance_deg, linestyle=':', color='green', alpha=0.7,
                label='±2% or 0.5° Band')
    plt.axhline(final_deg - tolerance_deg, linestyle=':', color='green', alpha=0.7)

    # Mark overshoot peak
    peak_idx = np.argmax(theta_deg)
    plt.plot(t[peak_idx], theta_deg[peak_idx], 'ro', markersize=8, label='Overshoot Peak')

    # Mark settling time
    if settling_time > 0:
        plt.axvline(settling_time, linestyle='--', color='red', alpha=0.7, label='Settling Time')
        plt.text(settling_time, theta_deg[int(settling_time/t[1])],
                f'  Settling Time = {settling_time:.2f}s',
                color='red', va='center', ha='left')

    plt.title(f'Attitude Response Analysis (k1={k1}, k2={k2})')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"analysis_k1_{k1}_k2_{k2}_{timestamp}.png", subfolder=subfolder)

    print(f"\nPerformance Analysis for k1={k1}, k2={k2}")
    print(f"Overshoot: {overshoot_deg:.2f}°")
    print(f"Settling Time: {settling_time:.2f}s")

    plt.show()
    return settling_time, overshoot_deg


def plot_root_locus(i_sc, mode='k1', k_fixed=5.0, sweep_range=(0.1, 50.0),
                   num_points=200, subfolder="root_locus"):
    """Plot root locus with interactive slider"""

    k_vals = np.linspace(sweep_range[0], sweep_range[1], num_points)
    roots = generate_root_locus_sweep(i_sc, k_fixed, k_vals, mode=mode)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Plot root locus curves
    colors = ['tab:blue', 'tab:orange']
    lines = []

    for i in range(roots.shape[1]):
        line, = ax.plot(roots[:, i].real, roots[:, i].imag,
                       color=colors[i % 2], linewidth=2, label=f'Root {i+1}')
        lines.append(line)

    # Add reference lines
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)

    # Formatting
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(f"Root Locus: Varying {'k1' if mode == 'k1' else 'k2'} "
                f"(k{'2' if mode == 'k1' else '1'}={k_fixed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Add slider for interactive control
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, f"k{'2' if mode == 'k1' else '1'}", 0.1, 50.0,
                   valinit=k_fixed, valfmt='%.1f')

    def update_plot(val):
        """Update root locus when slider changes"""
        new_fixed = slider.val
        new_roots = generate_root_locus_sweep(i_sc, new_fixed, k_vals, mode=mode)
        for i, line in enumerate(lines):
            line.set_xdata(new_roots[:, i].real)
            line.set_ydata(new_roots[:, i].imag)
        ax.set_title(f"Root Locus: Varying {'k1' if mode == 'k1' else 'k2'} "
                    f"(k{'2' if mode == 'k1' else '1'}={new_fixed:.1f})")
        fig.canvas.draw_idle()

    slider.on_changed(update_plot)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_plot(f"root_locus_{mode}_k_fixed_{k_fixed}_{timestamp}.png", subfolder=subfolder)
    plt.show()


def best_gain_plots(t, theta_best, ang_vel_best, torque_best, k1_best, k2_best, i_sc):
    """Generate plots for optimal gains"""
    base_folder = "best_gains"
    plot_time_domain(t, theta_best, ang_vel_best, torque_best,
                    k1=k1_best, k2=k2_best,
                    subfolder=os.path.join(base_folder, "time_domain"))
    plot_root_locus(i_sc, mode='k1', k_fixed=k2_best,
                   subfolder=os.path.join(base_folder, "root_locus"))
    plot_root_locus(i_sc, mode='k2', k_fixed=k1_best,
                   subfolder=os.path.join(base_folder, "root_locus"))
    analysis(t, theta_best, k1=k1_best, k2=k2_best,
            subfolder=os.path.join(base_folder, "analysis"))


#----------------Run Simulation------------------------
if __name__ == "__main__":
    main()
