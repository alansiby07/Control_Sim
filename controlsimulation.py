import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


#------------------------MAIN SIMULATION FUNCTION------------------- 
def main():
    mass = 10.0  #VARIABLE
    side_length = 1.0   #VARIABLE

    rw_mass = 1.0    #VARIABLE
    radius = 0.2     #VARIABLE

    i_sc = (1/6) * mass * side_length**2
    i_rw = (4/3) * rw_mass * radius**2


    wheel_speed = 0.0 #VARIABLE
    max_wheel_speed = 3000 * 2 * np.pi / 60   #To be adjusted by user  #RPM * (1 rev/60 secs) 

    torq_max = 0.05  # To be adjusted by user
    max_acc = 10.0 #VARIABLE

    #Controller Gains

    k1 = 4.0   #VARIABLE
    k2 = 10.0   #VARIABLE

    #Initial States

    initial_theta = np.deg2rad(10)
    initial_vel = 0.0

    #Time Setup

    t = np.linspace(0, 10, 1000)
    t_change = t[1] - t[0]
    t_end = t[-1]

    #Runs functions to solve for theta and ang_vel
    theta, ang_vel = solve_ode(k1, k2, i_sc, t, initial_theta, initial_vel)

    torque_list = []
    wheel_speed_list = []
    rw_acc_list = []


    #Runs functions to calculate torque and reaction wheel acceleration and adjusts wheel_speed based off of that
    
    for i in range(len(t)):
        torque, rw_acc = control_calc(theta[i], ang_vel[i], k1, k2, i_rw, torq_max,
                                                   max_acc, t_change, max_wheel_speed,)
        # Reaction Wheel Speed Calculation
    
        wheel_speed += -rw_acc * t_change
        wheel_speed = np.clip(wheel_speed, -max_wheel_speed, max_wheel_speed)

        torque_list.append(torque)
        wheel_speed_list.append(wheel_speed)
        rw_acc_list.append(rw_acc)

    #Keeps track of the values of each of these so that they can be accurately displayed

    torque = np.array(torque_list)
    wheel_speed = np.array(wheel_speed_list)
    rw_acc = np.array(rw_acc_list)

    k1_values = np.linspace(1, 100, 50)
    k2_values = np.linspace(1, 100, 50)

    #Uses function that displays all of the plots
    plot_all(t, theta, ang_vel, torque, k1=k1, k2=k2, i_sc=i_sc,
             k1_vals=k1_values, k2_vals=k2_values, annotate_points=[(k1, k2)])

    #Uses the function that finds the best possible gains
    valid_k = valid_gains_finder(k1_values, k2_values, t=t, initial_theta= np.deg2rad(10), 
                                 initial_vel=0.0, i_sc=i_sc,torq_max=0.05)




#---------------2nd ODE Solver Function--------------
def solve_ode(k1, k2, i_sc, t, initial_theta, initial_vel):
    a = 1
    b = k2/i_sc
    c = k1/i_sc

    discriminant = b**2.0 - 4.0 * a * c

    lam1 = (-b + np.sqrt(discriminant)) / 2
    lam2 = (-b - np.sqrt(discriminant)) / 2
    A = (initial_vel - (initial_theta * lam2)) / (lam1 - lam2)
    B = initial_theta - A

    theta = (A * np.exp(lam1 * t)) + (B * np.exp(lam2 * t))
    ang_vel = (A * lam1 * np.exp(lam1 * t)) + (B * lam2 * np.exp(lam2 * t))

    return theta, ang_vel



#----------------Torque & Reaction Wheel Accleration Calculator Function--------------
def control_calc(theta, ang_vel, k1, k2, i_rw, torq_max, max_acc, t_change, max_wheel_speed):
    torque = (-k1 * theta) - (k2 * ang_vel)
    torque = np.clip(torque, -torq_max, torq_max)

    # Angular Acceleration Calculation
    rw_acc = torque / i_rw
    rw_acc = np.clip(rw_acc, -max_acc, max_acc)

    return torque, rw_acc



#----------------------Optimal Gain Finder Function-------------------------

def valid_gains_finder(k1_range, k2_range, t, initial_theta, initial_vel, i_sc, torq_max):
    valid_gains = []
    best_gain = None
    best_cost = float('inf')

    w_settle = 1.0
    w_overshoot = 2.0
    
    for k1 in k1_range:
        for k2 in k2_range:
            a = 1
            b = k2/i_sc
            c = k1/i_sc

            discriminant = b**2.0 - 4.0 * a * c

            if discriminant < 0:
                continue

            lam1 = (-b + np.sqrt(discriminant)) / 2
            lam2 = (-b - np.sqrt(discriminant)) / 2
            A = (initial_vel - (initial_theta * lam2)) / (lam1 - lam2)
            B = initial_theta - A

            theta = (A * np.exp(lam1 * t)) + (B * np.exp(lam2 * t))
            ang_vel = (A * lam1 * np.exp(lam1 * t)) + (B * lam2 * np.exp(lam2 * t))
            torque = (-k1 * theta) - (k2 * ang_vel)
            torque = np.clip(torque, -torq_max, torq_max)

            final_theta = theta[-1]
            peak_theta = np.max(theta)
            overshoot_rad = peak_theta - final_theta
            overshoot_deg = np.rad2deg(overshoot_rad)

            tolerance = 0.02 * np.abs(final_theta)
            within_band = np.abs(theta - final_theta) < tolerance

            last_out_band = np.where(~within_band)[0]

            if len(last_out_band) == 0:
                settling_time = 0.0
                settle_index = 0
            else:
                settle_index = last_out_band[-1] + 1
                settling_time = t[settle_index] if settle_index < len(t) else np.nan

            if settling_time < 5.0 and overshoot_deg < 2.0:
                valid_gains.append((k1, k2, settling_time, overshoot_deg))
            
            if np.isfinite(settling_time):
                cost = w_settle * settling_time + w_overshoot * overshoot_deg
                if cost < best_cost:
                    best_cost = cost
                    best_gain = (k1, k2, settling_time, overshoot_deg)
    if valid_gains:
        print("Valid gains (k1, k2, settling_time, overshoot):")
        for j in valid_gains:
            print(f"k1={j[0]}, k2={j[1]} & ts={j[2]:.2f}s, os={j[3]:.2f}°")

    else:
        print("There are no gain combinations that can meeet the desgin requirement.")
    
    return valid_gains



#----------------Plotting Functions----------------------

#Plots both the time_domain plots and the root locus plot

def plot_all(t, theta, ang_vel, torque, k1, k2, i_sc, k1_vals, k2_vals, annotate_points):
    plot_time_domain(t, theta, ang_vel, torque)
    #plot_root_locus(i_sc, k1_vals, k2_vals, annotate_points)
    #plot_root_locus(i_sc, k1, k2)
    plot_root_locus(i_sc, mode='k1', k_fixed=5.0)
    plot_root_locus(i_sc, mode='k2', k_fixed=5.0)
    analysis(t, theta)



#Plots all of the data that shows the change in angular velocity, position, and torque as time passes

def plot_time_domain(t, theta, ang_vel, torque):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, np.rad2deg(theta), label='Angle θ (deg)', color='blue')
    plt.ylabel("Angle (deg)")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, np.rad2deg(ang_vel), label='Angular Velocity ω (deg/s)', color='green')
    plt.ylabel("Angular Velocity (deg/s)")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, torque, label='Control Torque τ (Nm)', color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.suptitle("1-Axis Spacecraft with Control (State-Space)", fontsize=16, y=1.02)
    plt.show()



#Plots a graph for the Overshoot and Settling Time

def analysis(t, theta):
    final_theta = theta[-1]
    peak_theta = np.max(theta)
    overshoot_rad = peak_theta - final_theta

    theta_deg = np.rad2deg(theta)
    final_deg = np.rad2deg(final_theta)
    overshoot_deg = np.rad2deg(overshoot_rad)

    tolerance = 0.02 * np.abs(final_theta)
    within_band = np.abs(theta - final_theta) < tolerance

    last_out_band = np.where(~within_band)[0]

    if len(last_out_band) == 0:
        settling_time = 0.0
        settle_index = 0
    else:
        settle_index = last_out_band[-1] + 1
        settling_time = t[settle_index] if settle_index < len(t) else np.nan

    plt.figure(figsize=(10, 6))
    plt.plot(t, theta_deg, label='θ(t) [deg]', color = 'blue')
    plt.axhline(final_deg, linestyle='--', color='gray', label='Final Value')
    plt.axhline(final_deg + tolerance, linestyle=':', color='green', label='±2%')
    plt.axhline(final_deg - tolerance, linestyle=':', color='green')

    peak_time = t[np.argmax(theta)]
    plt.plot(peak_time, np.rad2deg(peak_theta), 'ro', label='Overshoot Peak')

    plt.axvline(settling_time, linestyle='--', color='red', label='Settling Time')
    plt.text(settling_time, theta_deg[settle_index], f'Settling Time = {settling_time:.2f}s', color='red', 
             va='bottom', ha='right')
    
    plt.title('Angular Response with Overshoot & Settling Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()



#Plots the root locus based off the gains allows for the one fixed value to change via slider 

def plot_root_locus(i_sc, mode='k1', k_fixed=5.0, sweep_range=(0.1, 100.0), num_points=500):
    k_vals = np.linspace(sweep_range[0], sweep_range[1], num_points)
    roots = generate_root_locus_sweep(i_sc, k_fixed, k_vals, mode=mode)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.25, bottom=0.25)

    colors = ['tab:blue', 'tab:orange']
    lines = []

    for i in range(roots.shape[1]):
        (line,) = ax.plot(roots[:, i].real, roots[:, i].imag, color=colors[i % 2], label=f'Root {i+1}')
        lines.append(line)

    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel("Real Axis")
    ax.set_ylabel("Imaginary Axis")
    ax.set_title(f"Root Locus (s² + (k2/I)s + (k1/I) = 0)\nSweeping {'k1' if mode=='k1' else 'k2'}")

    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    #Creates a slider that will allow for the fixed gain to change
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, f"Fixed {'k2' if mode == 'k1' else 'k1'}", 0.1, 50.0, valinit=k_fixed)

    #Updates the value of the fixed value as the slider is moved and recalculates the root locus value

    def update(val):
        new_fixed = slider.val
        new_roots = generate_root_locus_sweep(i_sc, new_fixed, k_vals, mode=mode)
        for i, line in enumerate(lines):
            line.set_ydata(new_roots[:, i].imag)
            line.set_xdata(new_roots[:, i].real)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()



#Calculates the roots from the characteristic equation

def compute_roots(i_sc, k1, k2):
    a = 1
    b = k2 / i_sc
    c = k1 / i_sc
    return np.roots([a, b, c])



#Generates the root locus cruves for the plots

def generate_root_locus_sweep(i_sc, k_fixed, sweep_vals, mode='k1'):
    all_roots = []
    for k in sweep_vals:
        if mode == 'k1':
            roots = compute_roots(i_sc, k, k_fixed)
        else:
            roots = compute_roots(i_sc, k_fixed, k)
        all_roots.append(roots)
    return np.array(all_roots)




#----------------Run Simulation------------------------
if __name__ == "__main__":
    main()




