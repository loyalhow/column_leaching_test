import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# Other Model Parameters (fixed)
column_radius = 0.015 / 2  # m
column_length = 0.1  # m
porosity = 0.56
flow_rate = 1e-6 / 60  # m^3/s
tracer_concentration_initial = 1.0
num_cells = 200
courant_number = 0.05

# Experimental data
time_min = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170])
t_experimental_sec = time_min * 60
C_C0_experimental = np.array([0, 0.014760515, 0.175887523, 0.485472549, 0.707604097, 0.835156691, 0.914233079, 0.951444023, 0.968148511, 0.976842201, 0.981451698, 0.984140508, 0.986234813, 0.988211131, 0.987942628, 0.989891922, 0.990487704, 0.988122501, 0.990777338, 0.992974945, 0.991334211, 0.988668633, 0.861194867, 0.375147094, 0.197946708, 0.116836586, 0.077614338, 0.051630525, 0.03407941, 0.023896686, 0.016742431, 0.012989476, 0.011099304, 0.007171042])

# Derived Parameters (calculated once outside the optimization)
column_area = np.pi * column_radius**2
column_volume = column_area * column_length
pore_water_velocity_val = flow_rate / (column_area * porosity)
time_for_1_pv_val = pore_volume / flow_rate
injection_duration_val = 10 * time_for_1_pv_val
delta_x_val = column_length / num_cells

def simulate_breakthrough(disp_coeff):
    delta_t_val = courant_number * delta_x_val / pore_water_velocity_val
    total_simulation_time_val = 20 * time_for_1_pv_val
    num_time_steps_val = int(total_simulation_time_val / delta_t_val)
    if num_time_steps_val == 0:
        num_time_steps_val = 1

    concentration_sim = np.zeros(num_cells)
    effluent_concentration_sim = np.zeros(num_time_steps_val)
    time_points_sim = np.linspace(0, total_simulation_time_val, num_time_steps_val)
    pore_volumes_sim = time_points_sim / time_for_1_pv_val

    for t in range(num_time_steps_val):
        if time_points_sim[t] <= injection_duration_val:
            concentration_sim[0] = tracer_concentration_initial
        else:
            concentration_sim[0] = 0.0

        concentration_next_sim = np.copy(concentration_sim)
        for i in range(1, num_cells - 1):
            dispersion_term = disp_coeff * (concentration_sim[i + 1] - 2 * concentration_sim[i] + concentration_sim[i - 1]) / delta_x_val**2
            advection_term = -pore_water_velocity_val * (concentration_sim[i] - concentration_sim[i - 1]) / delta_x_val
            concentration_next_sim[i] = concentration_sim[i] + delta_t_val * (dispersion_term + advection_term)

        concentration_next_sim[-1] = concentration_next_sim[-2]
        concentration_sim = concentration_next_sim
        effluent_concentration_sim[t] = concentration_sim[-1]

    # Interpolate simulated data at experimental pore volumes
    interp_func = interp1d(pore_volumes_sim, effluent_concentration_sim, kind='linear', bounds_error=False, fill_value="extrapolate")
    simulated_at_experimental_pv = interp_func(pore_volumes_experimental)
    return simulated_at_experimental_pv

def objective_function(disp_coeff_array): # Expecting an array from minimize
    disp_coeff = disp_coeff_array[0] # Extract the scalar value
    simulated_values = simulate_breakthrough(disp_coeff)
    error = np.sum((simulated_values - C_C0_experimental)**2)
    return error

# Initial guess for the dispersion coefficient
initial_guess = np.array([3.6e-7]) # Pass the initial guess as an array

# Perform optimization
result = minimize(objective_function, initial_guess, method='Nelder-Mead')
optimized_dispersion_coefficient = result.x[0]
print(f"Optimized Dispersion Coefficient: {optimized_dispersion_coefficient:.2e}")

# Run simulation with the optimized coefficient
dispersion_coefficient = optimized_dispersion_coefficient

# Initialize Concentration Array
concentration = np.zeros(num_cells)
effluent_concentration = np.zeros(num_time_steps)
time_points = np.linspace(0, total_simulation_time, num_time_steps)
pore_volumes_eluted = time_points / time_for_1_pv

# Simulation Loop
for t in range(num_time_steps):
    # Inlet Boundary Condition
    if time_points[t] <= injection_duration:
        concentration[0] = tracer_concentration_initial
    else:
        concentration[0] = 0.0

    # Numerical Scheme (Explicit Finite Difference)
    concentration_next = np.copy(concentration)
    for i in range(1, num_cells - 1):
        dispersion_term = dispersion_coefficient * (concentration[i + 1] - 2 * concentration[i] + concentration[i - 1]) / delta_x**2
        advection_term = -pore_water_velocity * (concentration[i] - concentration[i - 1]) / delta_x
        concentration_next[i] = concentration[i] + delta_t * (dispersion_term + advection_term)

    # Outlet Boundary Condition (Advective Outflow - simple approximation)
    concentration_next[-1] = concentration_next[-2]

    concentration = concentration_next
    effluent_concentration[t] = concentration[-1]

# Experimental data
time_min = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170])
t_experimental_sec = time_min * 60  # Convert minutes to seconds
C_C0_experimental = np.array([0, 0.014760515, 0.175887523, 0.485472549, 0.707604097, 0.835156691, 0.914233079, 0.951444023, 0.968148511, 0.976842201, 0.981451698, 0.984140508, 0.986234813, 0.988211131, 0.987942628, 0.989891922, 0.990487704, 0.988122501, 0.990777338, 0.992974945, 0.991334211, 0.988668633, 0.861194867, 0.375147094, 0.197946708, 0.116836586, 0.077614338, 0.051630525, 0.03407941, 0.023896686, 0.016742431, 0.012989476, 0.011099304, 0.007171042])

# Convert experimental time to pore volumes
pore_volumes_experimental = t_experimental_sec / time_for_1_pv

# Plotting the Results
plt.figure(figsize=(10, 6))
plt.plot(pore_volumes_eluted, effluent_concentration, label='Simulated Breakthrough Curve')
plt.scatter(pore_volumes_experimental, C_C0_experimental, label='Experimental Data', marker='o', color='red')
plt.xlabel('Pore Volumes (PV)')
plt.ylabel('Normalized Effluent Concentration (C/C0)')
plt.title('Simulated Breakthrough Curve with Injection Stopped at 10 PV')
plt.xlim(0, 20)
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.show()