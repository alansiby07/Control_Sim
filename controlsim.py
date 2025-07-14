import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

#System Paramters

mass = input("Mass of spacecraft")
side_length = input("Length of each side")
mi_sc = (1/6) * mass * side_length**2 #moment of inertia for the spacecraft
mi_rw = #I believe it is inputted in by user but will check again


#Controller Gains
k1 = #gains for the angular positon
k2 = #gains for the angular velocity




#State Space Matrices and Modelling

A_cl = np.array([0, 1], [-k1/mi_sc, -k2/mi_sc])

#Not used because closed loop A matrix is here 
'''
A = np.array([0, 1], [0, 0])
'''

B = np.array([0], [1/mi_sc])
C = np.array([1, 0])
D = np.array([0])

#Creates the state-space system
sys = ctrl.ss(A_cl, B, C, D)


#Sim Setup
t = np.linspace(0, 10, 1000)
x0 = 




#Step Response

step_input_deg = 10
step_input_rad = np.deg2rad(step_input_deg) #changes degrees to radians for equations





