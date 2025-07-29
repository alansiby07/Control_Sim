# Spacecraft 1-Axis PD Control Simulation

This repository contains code that simulates a 1-degree-of-freedom spacecraft attitude control system using a reaction wheel actuated by a PD (Proportional-Derivative) controller.

## What the Code Does

The simulation models the rotational dynamics of a spacecraft controlled by a single-axis reaction wheel. The user specifies physical parameters and PD control gains. The script:

- Solves the spacecraft's equations of motion using the 4th-order Runge-Kutta (RK4) method
- Applies a PD controller to command torque through the reaction wheel
- Enforces actuator constraints like maximum torque and wheel speed limits
- Sweeps through a range of `k1` and `k2` gains to identify combinations that minimize settling time and overshoot
- Generates time-domain plots, root locus plots, and performance analysis visualizations
- Automatically saves all results (plots and logs) in timestamped directories for traceability

## Requirements

- Python 3.8 or higher

Required Python libraries:
- numpy
- matplotlib

## Setup Instructions
It is recommended to create a virtual environment.

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/spacecraft-pd-simulation.git
cd spacecraft-pd-simulation
```

### Step 2: Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install required packages

```bash
pip install numpy matplotlib
```

## Running the Simulation

Run the simulation script:

```bash
python simulation.py
```

Replace `simulation.py` with the actual filename if different.


### Log Output

- Simulation metadata and messages are stored in `logs/` with a timestamped filename.

### Plot Output

- Time-domain response plots of angle, angular velocity, and torque
- Root locus plots sweeping `k1` and `k2`
- Settling time and overshoot analysis
- Best-performing gain results (saved separately)

## Customization

In the  `main()` function in order to simulate different scenarios to change:

- Physical parameters (mass, inertia, torque/speed limits)
- Controller gains `k1`, `k2`
- Gain sweep ranges: `k1_values`, `k2_values`
- Performance criteria: `overshoot_limit`, `settling_time_limit`
- Simulation duration or resolution: `t_end`, `dt`

## Troubleshooting

- If plots do not display, try removing `plt.show()` and manually inspect the saved `.png` files in the `plots/` directory.
- If simulations are slow, reduce the number of gain values in `k1_values` and `k2_values`.

---

