import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

mass = 10.0
side_length = 1.0
i_sc = (1/6) * mass * side_length**2


k1 = 4.0
k2 = 5.0

initial_theta = np.deg2rad(10)
initial_vel = 0.0

a = 1
b = k2/i_sc
c = k1/i_sc

discriminant = np.abs(b**2.0 - 4.0 * a * c)

t= np.linspace(0, 10, 1000)


#Solution for Time-Domain

lam1 = (-b + np.sqrt(discriminant)) / 2
lam2 = (-b - np.sqrt(discriminant)) / 2
A = (initial_vel - (initial_theta * lam2)) / (lam1 - lam2)
B = initial_theta - A

theta = (A * np.exp(lam1 * t)) + (B * np.exp(lam2 * t))
ang_vel = (A * lam1 * np.exp(lam1 * t)) + (B * lam2 * np.exp(lam2 * t))

#Output

torque = (-k1 * theta) - (k2 * ang_vel)


#Plotting
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
plt.suptitle("1-Axis Spacecraft with PD Control (State-Space)", fontsize=16, y=1.02)
plt.show()






"""
HI 
"""










































#ALT CODE (THERE IS A BUG WITH ctrl.initial_response)
'''
#System Paramters

mass = input("Mass of spacecraft ")
side_length = input("Length of each side ")
mi_sc = float(1/6) * float(mass) * float(side_length)**2 #moment of inertia for the spacecraft
mi_rw =0 #I believe it is inputted in by user but will check again


#Controller Gains
k1 =2 #gains for the angular positon
k2 = 3#gains for the angular velocity




#State Space Matrices and Modelling

A_cl = np.array([[0, 1], [-k1/mi_sc, -k2/mi_sc]])

#Not used because closed loop A matrix is here 

A = np.array([0, 1], [0, 0])


B = np.array([[0], [1/mi_sc]])
C = np.array([1, 0])
D = np.array([0])

#Creates the state-space system
sys = ctrl.ss(A_cl, B, C, D)    


#Sim Setup
t = np.linspace(0, 10, 1000)
x0 = np.array([np.deg2rad(10), 0]) 
y_out = ctrl.initial_response(sys, T=t, X0=x0, return_x=True)
t_out =  ctrl.initial_response(sys, T=t, X0=x0, return_x=True)


#Outputs
ang_position = y_out[0]
ang_velocity = y_out[1]
torque = (-k1 * ang_position) - (k2 * ang_velocity)



#Plotting

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t_out, np.rad2deg(ang_position), label='Angle θ (deg)', color='blue')
plt.ylabel("Angle (deg)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_out, np.rad2deg(ang_velocity), label='Angular Velocity ω (deg/s)', color='green')
plt.ylabel("Angular Velocity (deg/s)")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_out, torque, label='Control Torque τ (Nm)', color='red')
plt.xlabel("Time (s)")
plt.ylabel("Torque (Nm)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.suptitle("1-Axis Spacecraft with PD Control (State-Space)", fontsize=16, y=1.02)
plt.show()

'''
