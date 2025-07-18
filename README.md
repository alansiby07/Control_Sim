# Spacecraft Attitude Control (1 Axis)

This is a Python simulation that models a spacecraft rotating around one axis using a reaction wheel and a PD controller.

## Main File
- `controlsimulation.py` – Runs the simulation and shows the plots.

## What It Does
- Uses PD control: τ = -k1 * θ - k2 * ω
- Simulates a cube spacecraft and a reaction wheel
- Applies torque with limits (max torque, max acceleration, max wheel speed)
- Plots angle, angular velocity, and torque over time
- Computes settling time and overshoot
- Generates a root locus plot with an interactive slider
- Finds valid (k1, k2) gain pairs based on performance

## How to Run
1. Install the required libraries:
   ```bash
   pip install numpy matplotlib

