import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, cpu_count
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Constants for argon
epsilon = 1.65e-21  # Depth of the potential well in Joules
sigma = 3.4e-10  # Distance at which the potential is zero in meters
mass = 6.63e-26  # Mass of argon atom in kg
k_B = 1.38e-23  # Boltzmann constant in J/K

# Set up the FCC lattice
def create_fcc_lattice(a, n):
    positions = []
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    for x in range(n):
        for y in range(n):
            for z in range(n):
                offset = np.array([x, y, z])
                for b in base:
                    positions.append((offset + b) * a)
    return np.array(positions)

# Lennard-Jones potential and force
def lennard_jones(r2):
    r6 = (sigma ** 2 / r2) ** 3
    r12 = r6 ** 2
    potential = 4 * epsilon * (r12 - r6)
    force = 24 * epsilon * (2 * r12 - r6) / r2
    return potential, force

# Apply periodic boundary conditions
def apply_pbc(position, box_length):
    return position - box_length * np.round(position / box_length)

# Compute forces using Lennard-Jones potential with PBC
def compute_forces(positions, box_length):
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    N = len(positions)
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = apply_pbc(positions[i] - positions[j], box_length)
            r2 = np.dot(r_vec, r_vec)
            if r2 < (2.5 * sigma) ** 2:  # Squared cut-off distance
                potential, force = lennard_jones(r2)
                f = force * r_vec
                forces[i] += f
                forces[j] -= f
                potential_energy += potential
    return forces, potential_energy

# Velocity-Verlet integration
def velocity_verlet(positions, velocities, forces, dt, box_length):
    new_positions = positions + velocities * dt + 0.5 * forces * dt**2 / mass
    new_positions = apply_pbc(new_positions, box_length)
    new_forces, _ = compute_forces(new_positions, box_length)
    new_velocities = velocities + 0.5 * (forces + new_forces) * dt / mass
    return new_positions, new_velocities, new_forces

# Standard Verlet integration
def standard_verlet(positions, velocities, forces, dt, box_length):
    new_positions = positions + velocities * dt + 0.5 * forces * dt**2 / mass
    new_positions = apply_pbc(new_positions, box_length)
    new_forces, _ = compute_forces(new_positions, box_length)
    new_velocities = velocities + new_forces * dt / mass
    return new_positions, new_velocities, new_forces

# Leapfrog Verlet integration
def leapfrog_verlet(positions, velocities, forces, dt, box_length):
    half_dt = 0.5 * dt
    velocities_half = velocities + forces * half_dt / mass
    new_positions = positions + velocities_half * dt
    new_positions = apply_pbc(new_positions, box_length)
    new_forces, _ = compute_forces(new_positions, box_length)
    new_velocities = velocities_half + new_forces * half_dt / mass
    return new_positions, new_velocities, new_forces

# Function to run MD for a single temperature using the specified Verlet algorithm
def run_md_for_temperature(T, seed, verlet_algorithm, dt, sampling_steps=20000):
    np.random.seed(seed)
    # Initialize the system
    a = 5.26e-10  # Initial guess for the lattice parameter
    n = 3  # Number of unit cells in each dimension
    positions = create_fcc_lattice(a, n)
    box_length = n * a  # Length of the simulation box
    N = len(positions)
    velocities = np.random.randn(N, 3) * np.sqrt(k_B * 5 / mass)  # Initialize velocities for 5K
    forces, _ = compute_forces(positions, box_length)

    # Rescale velocities to the desired temperature
    current_temperature = np.sum(mass * np.sum(velocities**2, axis=1)) / (3 * N * k_B)
    scale_factor = np.sqrt(T / current_temperature)
    velocities *= scale_factor

    # Equilibrate the system
    equilibration_steps = int(1.5 * sampling_steps)  # 10% of sampling steps
    for _ in range(equilibration_steps):
        positions, velocities, forces = verlet_algorithm(positions, velocities, forces, dt, box_length)

    # Calculate the average potential energy and energy drift
    potential_energy = 0.0
    initial_total_energy = 0.0
    count = 0
    total_energy_drift = 0.0

    for step in range(sampling_steps):
        positions, velocities, forces = verlet_algorithm(positions, velocities, forces, dt, box_length)
        kinetic_energy = 0.5 * mass * np.sum(velocities**2)
        _, potential_energy_step = compute_forces(positions, box_length)
        total_energy = kinetic_energy + potential_energy_step
        potential_energy += potential_energy_step
        if step == 0:
            initial_total_energy = total_energy
        total_energy_drift += abs(total_energy - initial_total_energy)
        count += 1

    potential_energy /= count
    energy_drift = total_energy_drift / count

    return dt, potential_energy, energy_drift

# Function to optimize lattice parameter
def potential_energy_for_lattice(a):
    positions = create_fcc_lattice(a, 3)
    box_length = 3 * a
    _, potential_energy = compute_forces(positions, box_length)
    return potential_energy

def optimize_lattice_parameter():
    initial_guesses = np.linspace(4.0e-10, 6.5e-10, 50)
    results = [minimize(potential_energy_for_lattice, [guess], bounds=[(4.0e-10, 6.5e-10)]) for guess in initial_guesses]
    best_result = min(results, key=lambda x: x.fun)
    return best_result.x[0]

# Function to test different time steps
def test_time_steps(T, seed, verlet_algorithm, time_steps, sampling_steps=20000):
    with Pool(16) as pool:
        args = [(T, seed, verlet_algorithm, dt, sampling_steps) for dt in time_steps]
        results = pool.starmap(run_md_for_temperature, args)
    return results

# Generate PDF report
def generate_pdf_report(filename, melting_point, boiling_point, a_opt, temperatures, potential_energies, runtime, best_algorithm_name, best_time_step, reason):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.drawString(50, height - 50, "Molecular Dynamics Simulation Report")
    c.drawString(50, height - 70, f"Optimized Lattice Parameter: {a_opt:.2e} m")
    c.drawString(50, height - 90, f"Melting Point: {melting_point} K")
    c.drawString(50, height - 110, f"Boiling Point: {boiling_point} K")
    c.drawString(50, height - 130, f"Total Runtime: {runtime:.2f} seconds")
    c.drawString(50, height - 150, f"Best Verlet Algorithm: {best_algorithm_name}")
    c.drawString(50, height - 170, f"Best Time Step: {best_time_step:.2e} s")
    
    # Wrap the reason text to avoid overflowing off the page
    reason_text = f"Reason: {reason}"
    max_chars_per_line = 78
    wrapped_reason = "\n".join([reason_text[i:i+max_chars_per_line] for i in range(0, len(reason_text), max_chars_per_line)])
    text_lines = wrapped_reason.split('\n')
    for i, line in enumerate(text_lines):
        c.drawString(50, height - 190 - i*20, line)

    # Save the plot as an image and insert it into the PDF
    plot_filename = "temp_plot.png"
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, potential_energies, marker='o', linestyle='-', label='Average Potential Energy (MD Simulation)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average Potential Energy (J)')
    plt.title('Average Potential Energy vs Temperature for Argon')
    plt.axvline(melting_point, color='r', linestyle='--', label=f'Melting Point: {melting_point} K')
    plt.axvline(boiling_point, color='g', linestyle='--', label=f'Boiling Point: {boiling_point} K')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.close()

    # Add the plot image to the PDF
    c.drawImage(plot_filename, 50, 200, width=500, height=300)
    
    c.save()

# Main script
if __name__ == "__main__":
    start_time = time.time()  # Start time measurement

    a_opt = optimize_lattice_parameter()
    print(f"Optimized lattice parameter: {a_opt}")

    # Define parameters for time step testing
    temperatures = np.arange(5, 205, 5)
    time_steps = [1e-15, 2e-15, 5e-15, 5e-16]  # Different time steps to test
    seeds = np.random.randint(0, 10000, len(temperatures))  # Different seeds for each temperature

    # Define Verlet algorithms
    verlet_algorithms = [velocity_verlet, standard_verlet, leapfrog_verlet]
    algorithm_names = ["Velocity-Verlet", "Standard Verlet", "Leapfrog Verlet"]

    # Run a short preliminary simulation to select the best Verlet algorithm and time step
    preliminary_steps = 500
    potential_energies_per_algorithm = []
    energy_drifts_per_algorithm = []

    best_time_steps = []

    for algorithm in verlet_algorithms:
        potential_energies = []
        energy_drifts = []
        for T, seed in zip(temperatures[:3], seeds[:3]):
            results = test_time_steps(T, seed, algorithm, time_steps, preliminary_steps)
            for dt, potential_energy, energy_drift in results:
                potential_energies.append(potential_energy)
                energy_drifts.append(energy_drift)
            best_time_step = min(results, key=lambda x: x[2])[0]  # Choose dt with minimal energy drift
            best_time_steps.append(best_time_step)
        potential_energies_per_algorithm.append(np.mean(potential_energies))
        energy_drifts_per_algorithm.append(np.mean(energy_drifts))

    # Multi-criteria decision based on average potential energy and energy drift
    best_algorithm_index = np.argmin([pe + ed for pe, ed in zip(potential_energies_per_algorithm, energy_drifts_per_algorithm)])
    best_verlet_algorithm = verlet_algorithms[best_algorithm_index]
    best_algorithm_name = algorithm_names[best_algorithm_index]
    best_time_step = best_time_steps[best_algorithm_index]
    reason = (f"Lowest average potential energy ({potential_energies_per_algorithm[best_algorithm_index]:.2e} J) "
              f"and minimal energy drift ({energy_drifts_per_algorithm[best_algorithm_index]:.2e} J) "
              f"during preliminary simulation.")
    print(f"Best Verlet algorithm: {best_algorithm_name}")
    print(f"Best time step: {best_time_step}")

    # Set sampling steps
    sampling_steps = 20000

    # Run the MD simulation in parallel using the best Verlet algorithm and best time step
    with Pool(16) as pool:
        results = pool.starmap(run_md_for_temperature, [(T, seed, best_verlet_algorithm, best_time_step, sampling_steps) for T, seed in zip(temperatures, seeds)])
    
    dts, potential_energies, energy_drifts = zip(*results)
    
    end_time = time.time()  # End time measurement
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f} seconds")

    # Smooth the potential energy data
    smoothed_potential_energies = gaussian_filter1d(potential_energies, sigma=2)

    # Calculate first and second derivatives
    first_derivative = np.gradient(smoothed_potential_energies, temperatures)
    second_derivative = np.gradient(first_derivative, temperatures)

    # Plotting the average potential energy as a function of temperature
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, smoothed_potential_energies, marker='o', linestyle='-', label='Smoothed Potential Energy (MD Simulation)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Smoothed Potential Energy (J)')
    plt.title('Smoothed Potential Energy vs Temperature for Argon')
    plt.legend()
    plt.grid(True)

    # Find melting and boiling points
    melting_point = temperatures[np.argmax(first_derivative)]
    boiling_point = temperatures[np.argmax(second_derivative)]

    plt.axvline(melting_point, color='r', linestyle='--', label=f'Melting Point: {melting_point} K')
    plt.axvline(boiling_point, color='g', linestyle='--', label=f'Boiling Point: {boiling_point} K')
    plt.legend()
    plt.show()

    # Generate PDF report
    generate_pdf_report("MD_Simulation_Report.pdf", melting_point, boiling_point, a_opt, temperatures, smoothed_potential_energies, runtime, best_algorithm_name, best_time_step, reason)
