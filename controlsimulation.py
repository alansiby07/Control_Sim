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



#--------------Controller Gains------------------

k1 = 4.0   #VARIABLE
k2 = 5.0   #VARIABLE



#----------------Initial States----------------

initial_theta = np.deg2rad(10)
initial_vel = 0.0


#--------------Time----------------------

t = np.linspace(0, 10, 1000)



#-----------Solution to 2nd ODE System----------

a = 1
b = k2/i_sc
c = k1/i_sc

discriminant = np.abs(b**2.0 - 4.0 * a * c)

lam1 = (-b + np.sqrt(discriminant)) / 2
lam2 = (-b - np.sqrt(discriminant)) / 2
A = (initial_vel - (initial_theta * lam2)) / (lam1 - lam2)
B = initial_theta - A

theta = (A * np.exp(lam1 * t)) + (B * np.exp(lam2 * t))
ang_vel = (A * lam1 * np.exp(lam1 * t)) + (B * lam2 * np.exp(lam2 * t))



#------------Torque & Reaction Wheel Speed Calculations------------

torque = (-k1 * theta) - (k2 * ang_vel)
torq_max = 0.05  # To be adjusted by user
torque = np.clip(torque, -torq_max, torq_max)

#Reaction Wheel Speed Calc

wheel_speed = -np.cumsum(torque) * (t[1] - t[0]) / i_rw



#--------------Root Locus Transfer Function----------------

num = [k2, k1]
den = [i_sc, k2, k1]
sys = ctrl.TransferFunction(num, den)



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
    plt.suptitle("1-Axis Spacecraft with PD Control (State-Space)", fontsize=16, y=1.02)
    plt.show()



#Plotting of Root Locus Plot

def plot_root_locus(sys):
    plt.figure()

    ctrl.root_locus(sys, plot=True, grid=True)
    plt.title("Root Locus of PD-Controlled System")
    plt.show()


#Plots both the time_domain plots and the root locus plot

def plot_all(t, theta, ang_vel, torque, sys):
    plot_time_domain(t, theta, ang_vel, torque)
    plot_root_locus(sys)



#----------------Run Simulation------------------------

plot_all(t, theta, ang_vel, torque, sys)





