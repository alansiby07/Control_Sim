# ControlSimulation: 1-Axis Spacecraft Attitude Control

This project simulates and analyzes the attitude dynamics of a 1-axis spacecraft using a proportional-derivative (PD) controller and a reaction wheel actuator. It computes system responses, evaluates control performance, and visualizes the effects of varying controller gains using root locus analysis.

---

## File

- `controlsimulation.py` – The main simulation script containing physical modeling, control logic, and plotting.

---

## Features

- PD control for single-axis spacecraft attitude regulation
- Reaction wheel torque generation with saturation limits
- Time-domain response plots for angle, angular velocity, and control torque
- Automated overshoot and settling time evaluation
- Root locus plotting with interactive gain slider
- Search for valid gain combinations meeting design constraints

---

## System Overview

- **Dynamics**: Spacecraft modeled as a rigid cube; the reaction wheel is a solid disk aligned with the control axis.
- **Control Law**:
  \[
  \tau = -k_1 \cdot \theta - k_2 \cdot \omega
  \]
- **Control Constraints**: Includes maximum torque, acceleration, and wheel speed saturation
- **Performance Metrics**: Settling time and overshoot computed from the simulated trajectory
- **Root Locus**: Visualizes closed-loop pole movement across gain values

---

## How to Run

1. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

2. Execute the simulation:
   ```bash
   python controlsimulation.py
   ```

---

## Key Parameters (modifiable in `main()`)

| Parameter          | Description                        |
|-------------------|------------------------------------|
| `mass`            | Spacecraft mass                    |
| `side_length`     | Cube side length                   |
| `rw_mass`         | Reaction wheel mass                |
| `radius`          | Reaction wheel radius              |
| `torq_max`        | Maximum torque allowed             |
| `max_acc`         | Maximum wheel acceleration         |
| `max_wheel_speed` | Maximum allowable wheel speed      |
| `k1`, `k2`        | PD controller gains                |
| `initial_theta`   | Initial angular displacement       |
| `initial_vel`     | Initial angular velocity           |

---

## Output and Visualization

- **Time-Domain Plots**: Angle, angular velocity, and control torque vs. time
- **Overshoot and Settling Time**: Computed for performance evaluation
- **Root Locus Plots**: Displayed with interactive gain control using Matplotlib sliders
- **Valid Gain Region**: Prints gain values that meet design performance criteria

---

## Example Console Output

```
Valid gains (k1, k2, settling_time, overshoot):
k1=12.0, k2=25.0 & ts=4.87s, os=1.32°
...
```

---

## Extensibility

- Can be adapted to full 3-axis spacecraft models
- Control law can be modified to include feedforward or nonlinear terms
- Environmental disturbances or noise can be added to test robustness

---

## License

This project is licensed under the GNU General Public License v3.0.  
See the `LICENSE` file or visit [https://www.gnu.org/licenses/gpl-3.0.en.html](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.

---

## Author

Developed by Alan Siby.



