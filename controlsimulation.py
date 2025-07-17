import numpy as np
import matplotlib.pyplot as plt
import control as ctrl


#----------- Physical Parameters---------------

mass = 10.0  #VARIABLE
side_length = 1.0   #VARIABLE

rw_mass = 1.0    #VARIABLE
radius = 0.2     #VARIABLE

i_sc = (1/6) * mass * side_length**2
i_rw = (4/3) * rw_mass * radius**2

torq_max = 0.05  # To be adjusted by user
max_acc = 10.0 #VARIABLE
   
#--------------Controller Gains------------------

k1 = 4.0   #VARIABLE
k2 = 5.0   #VARIABLE



#----------------Initial States----------------

initial_theta = np.deg2rad(10)
initial_vel = 0.0


#--------------Time----------------------

t = np.linspace(0, 10, 1000)
t_change = t[1] - t[0]


#--------------Root Locus Transfer Function----------------

numerator = [k2, k1]
denominator = [i_sc, k2, k1]
sys = ctrl.TransferFunction(numerator, denominator)



#-----------2nd ODE Solver Function----------
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


#------------Torque & Reaction Wheel Speed Calculator Function------------
def control_calc(theta, ang_vel, k1, k2, i_rw, torq_max, max_acc, t_change):
    torque = (-k1 * theta) - (k2 * ang_vel)
    torque = np.clip(torque, -torq_max, torq_max)

    
    #Angular Acceleration Calculatiokn

    rw_acc = torque / i_rw


    #Reaction Wheel Speed Calc
    max_wheel_speed = 3000 * 2 * np.pi / 60   #RPM * (1 rev/60 secs)
    wheel_speed = -np.cumsum(rw_acc * t_change) 
    wheel_speed = np.clip(wheel_speed, -max_wheel_speed, max_wheel_speed)

    return torque, wheel_speed


#----------------Plotting Functions----------------------

def plot_time_domain(t, theta, ang_vel, torque):
    plt.figure(figsize=(12, 8))


    #Plots Angular Position Time-Domain Plot
    plt.subplot(3, 1, 1)
    plt.plot(t, np.rad2deg(theta), label='Angle θ (deg)', color='blue')
    plt.ylabel("Angle (deg)")
    plt.grid(True)
    plt.legend()


    #Plots Angular Velocity Time-Domian Plot
    plt.subplot(3, 1, 2)
    plt.plot(t, np.rad2deg(ang_vel), label='Angular Velocity ω (deg/s)', color='green')
    plt.ylabel("Angular Velocity (deg/s)")
    plt.grid(True)
    plt.legend()


    #Plots Torque Time-Domain Plot
    plt.subplot(3, 1, 3)
    plt.plot(t, torque, label='Control Torque τ (Nm)', color='red')
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.grid(True)
    plt.legend()
    
    #Title, Layout
    plt.tight_layout()
    plt.suptitle("1-Axis Spacecraft with Control (State-Space)", fontsize=16, y=1.02)
    plt.show()



#Plotting of Root Locus Plot

def plot_root_locus(sys):
    plt.figure()

    ctrl.root_locus(sys, plot=True, grid=True)
    plt.title("Root Locus of Controlled System")
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

#Plots both the time_domain plots and the root locus plot

def plot_all(t, theta, ang_vel, torque, sys):
    plot_time_domain(t, theta, ang_vel, torque)
    plot_root_locus(sys)
    analysis(t, theta)

def valid_gains_finder(k1_range, k2_range, t, initial_theta, initial_vel, i_sc, torq_max):
    valid_gains = []
    t_end = t[-1]

    for k1 in k1_range:
        for k2 in k2_range:
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

    if valid_gains:
        print("Valid gains (k1, k2, settling_time, overshoot):")
        for j in valid_gains:
            print(f"k1={j[0]}, k2={j[1]} & ts={j[2]:.2f}s, os={j[3]:.2f}°")

    else:
        print(" There are no gain combinations that can meeet the desgin requirement.")
    
    return valid_gains

#----------------Run Simulation------------------------
theta, ang_vel = solve_ode(k1, k2, i_sc, t, initial_theta, initial_vel)
torque, wheel_speed = control_calc(theta, ang_vel, k1, k2, i_rw, torq_max, max_acc, t_change)
plot_all(t, theta, ang_vel, torque, sys)

k1_values = np.arange(1, 21)
k2_values = np.arange(1, 21)

valid_k = valid_gains_finder(k1_values, k2_values, t=t, initial_theta= np.deg2rad(10), 
                            initial_vel=0.0, i_sc=i_sc,torq_max=0.05)




