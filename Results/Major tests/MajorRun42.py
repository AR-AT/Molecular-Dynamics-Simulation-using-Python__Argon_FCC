import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tqdm import tqdm
from numba import njit, prange
import logging
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
script_base_name = os.path.splitext(os.path.basename(__file__))[0]

# Set up logging with the script's base name and directory
log_file_path = os.path.join(script_dir, f'{script_base_name}.log')
logging.basicConfig(filename=log_file_path, 
            level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

EPSILON = 1.654017502e-21  
SIGMA = 3.405e-10  
MASS = 6.63e-26  
K_B = 1.38e-23 

def create_fcc_lattice(a, n):
    """Create an FCC lattice."""
    base = np.array([[0, 0, 0],
                     [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    positions = np.zeros((n**3 * 4, 3))
    index = 0
    for x in range(n):
        for y in range(n):
            for z in range(n):
                offset = np.array([x, y, z])
                for b in base:
                    positions[index] = (offset + b) * a
                    index += 1
    return positions

@njit
def lennard_jones(r2):
    """Calculate the Lennard-Jones potential and force."""
    r6 = (SIGMA ** 2 / r2) ** 3
    r12 = r6 ** 2
    potential = 4 * EPSILON * (r12 - r6)
    force = 24 * EPSILON * (2 * r12 - r6) / r2
    return potential, force

@njit
def apply_pbc(position, box_length):
    """Apply periodic boundary conditions."""
    return position - box_length * np.round(position / box_length)

@njit
def custom_norm(arr, axis=1):
    """Compute the norm along a specified axis."""
    if axis == 1:
        return np.sqrt(np.sum(arr**2, axis=1))
    elif axis == 0:
        return np.sqrt(np.sum(arr**2, axis=0))
    else:
        raise ValueError("Invalid axis")

@njit(parallel=True)
def compute_forces(positions, box_length):
    """Compute forces using Lennard-Jones potential with PBC."""
    N = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    for i in prange(N):
        for j in range(i + 1, N):
            r_vec = apply_pbc(positions[i] - positions[j], box_length)
            r2 = np.dot(r_vec, r_vec)
            if r2 < (2.5 * SIGMA) ** 2:
                potential, force = lennard_jones(r2)
                f = force * r_vec
                forces[i] += f
                forces[j] -= f
                potential_energy += potential
                
    return forces, potential_energy

@njit
def velocity_verlet_adaptive(positions,
            velocities, forces, dt, box_length, max_force_threshold=1e-13):
    """Velocity-Verlet integration with adaptive time step."""
    new_positions = positions + velocities * dt + 0.5 * forces * dt**2 / MASS
    new_positions = apply_pbc(new_positions, box_length)
    new_forces, _ = compute_forces(new_positions, box_length)
    max_force = np.max(custom_norm(new_forces, axis=1))
    
    if max_force > max_force_threshold:
        dt = max(0.9 * dt * (max_force_threshold / max_force)**0.5, 1e-16)
    
    new_velocities = velocities + 0.5 * (forces + new_forces) * dt / MASS
    return new_positions, new_velocities, new_forces, dt

@njit
def standard_verlet_adaptive(positions,
    velocities, forces, dt, box_length, max_force_threshold=1e-13):
    """Standard Verlet integration with adaptive time step."""
    new_positions = positions + velocities * dt + 0.5 * forces * dt**2 / MASS
    new_positions = apply_pbc(new_positions, box_length)
    new_forces, _ = compute_forces(new_positions, box_length)
    max_force = np.max(custom_norm(new_forces, axis=1))

    if max_force > max_force_threshold:
        dt = max(0.9 * dt * (max_force_threshold / max_force)**0.5, 1e-16)

    new_velocities = velocities + new_forces * dt / MASS
    return new_positions, new_velocities, new_forces, dt

@njit
def leapfrog_verlet_adaptive(positions,
    velocities, forces, dt, box_length, max_force_threshold=1e-13):
    """Leapfrog Verlet integration with adaptive time step."""
    half_dt = 0.5 * dt
    velocities_half = velocities + forces * half_dt / MASS
    new_positions = positions + velocities_half * dt
    new_positions = apply_pbc(new_positions, box_length)
    new_forces, _ = compute_forces(new_positions, box_length)
    max_force = np.max(custom_norm(new_forces, axis=1))

    if max_force > max_force_threshold:
        dt = max(0.9 * dt * (max_force_threshold / max_force)**0.5, 1e-16)

    new_velocities = velocities_half + new_forces * half_dt / MASS
    return new_positions, new_velocities, new_forces, dt




def adaptive_equilibration(positions, velocities, forces, dt, box_length,
                           verlet_algorithm, initial_equilibration_window=500,
                           initial_threshold=5e-25, min_threshold=5e-27,
                           max_threshold=5e-21, max_steps=100000):
    """Perform adaptive equilibration with dynamic threshold and window adjustment."""
    equilibration_steps = 0
    window_size = initial_equilibration_window
    recent_potential_energies = np.zeros(window_size)
    index = 0
    forces, _ = compute_forces(positions, box_length)
    threshold = initial_threshold
    equilibration_window = initial_equilibration_window
    
    with tqdm(total=equilibration_window, desc="Equilibration") as pbar:
        while equilibration_steps < max_steps:
            positions, velocities, forces, dt = verlet_algorithm(positions,
                velocities, forces, dt, box_length)
            _, potential_energy = compute_forces(positions, box_length)
            
            recent_potential_energies[index % window_size] = potential_energy # to Reduce Ram 
            index += 1
            
            if index >= window_size:
                mean_potential_energy = np.mean(recent_potential_energies)
                std_potential_energy = np.std(recent_potential_energies)
                logging.info(f"((len(recent_p_energies) > equiwindow)) STD: {std_potential_energy:.5e}, MEAN: {mean_potential_energy:.5e}")

                # Dynamically adjust the threshold based on the standard deviation
                if std_potential_energy > threshold:
                    threshold = min(threshold * 1.09, max_threshold)
                    new_window_size = min(int(window_size * 1.00), len(recent_potential_energies) * 1.0)
                    if new_window_size > window_size:
                        recent_potential_energies = np.resize(recent_potential_energies, new_window_size)
                        window_size = new_window_size
                    logging.info(f"((std_potential_energy > threshold)) threshold: {threshold:.5e}, MEAN: {mean_potential_energy:.5e}")
                else:
                    threshold = max(threshold * 0.6, min_threshold)
                    logging.info(f"((std_potential_energy < threshold)) threshold: {threshold:.5e}, MEAN: {mean_potential_energy:.5e}")

                if std_potential_energy < threshold and np.abs(potential_energy - mean_potential_energy) < threshold:
                    logging.info(f"Equilibration completed in {equilibration_steps} steps with dt={dt:.2e}")
                    break
                
                pbar.update(1)
            equilibration_steps += 1
            if equilibration_steps % 1000 == 0:
                logging.info(f"Step {equilibration_steps}: Potential Energy = {potential_energy:.5e}, dt = {dt:.2e}")
    
    return positions, velocities, forces, dt, equilibration_steps




def run_md_for_temperature(T, seed, verlet_algorithm,
                           dt, sampling_steps=60000):
    """Run MD simulation for a single temperature using the specified Verlet algorithm."""
    np.random.seed(seed)
    a = 5.26e-10  # Initial guess for the lattice parameter
    n = 3  # Number of unit cells in each dimension
    positions = create_fcc_lattice(a, n)
    box_length = n * a  # Length of the simulation box
    N = len(positions)
    
   
    velocities = np.random.randn(N, 3) * np.sqrt(K_B * 5 / MASS)
    forces, _ = compute_forces(positions, box_length)

    # Rescale velocities to the desired temperature
    current_temperature = np.sum(MASS * np.sum(velocities**2, axis=1)) / (3 * N * K_B)
    scale_factor = np.sqrt(T / current_temperature)
    velocities *= scale_factor

    # Adaptive equilibration
    positions, velocities, forces, dt, equilibration_steps = adaptive_equilibration(
        positions, velocities, forces,
        dt, box_length, verlet_algorithm)

    logging.info(f"Equilibration steps: {equilibration_steps}")

    print(f"Equilibration steps: {equilibration_steps}")

    potential_energy = 0
    initial_total_energy = 0
    count = 0
    total_energy_drift = 0

    with tqdm(total=sampling_steps, desc="Progress") as pbar:
        potential_energy = 0
        count = 0
        total_energy_drift = 0
        initial_total_energy = None
    
        for step in range(sampling_steps):
            positions, velocities, forces, dt = verlet_algorithm(
                positions, velocities, forces, dt, box_length
            )
            kinetic_energy = 0.5 * MASS * np.sum(velocities**2)
            _, potential_energy_step = compute_forces(positions, box_length)
            total_energy = kinetic_energy + potential_energy_step
            potential_energy += potential_energy_step
    
            if step == 0:
                initial_total_energy = total_energy
    
            total_energy_drift += abs(total_energy - initial_total_energy)
            count += 1
            pbar.update(1)
            if step % 1000 == 0:
                logging.info(f"Step {step}: Kinetic Energy = {kinetic_energy:.5e}, Potential Energy = {potential_energy_step:.5e}, Total Energy = {total_energy:.5e}")
        
        average_potential_energy = potential_energy / count / N  
        energy_drift = total_energy_drift / count
    
    return dt, average_potential_energy, energy_drift

def potential_energy_for_lattice(a):
    """Calculate potential energy for a given lattice parameter."""
    positions = create_fcc_lattice(a, 3)
    box_length = 3 * a
    _, potential_energy = compute_forces(positions, box_length)
    return potential_energy

def optimize_lattice_parameter():
    """Optimize the lattice parameter."""
    initial_guesses = np.linspace(5e-10, 6e-10, 1000)
    results = [minimize(potential_energy_for_lattice,
                        [guess], bounds=[(5e-10, 6e-10)]) for
               guess in initial_guesses]
    best_result = min(results, key=lambda x: x.fun)
    return best_result.x[0]

def test_time_steps(T, seed, verlet_algorithm,
                    time_steps, sampling_steps=60000):
    """Test different time steps."""
    with Pool(16) as pool:
        args = [(T, seed, verlet_algorithm,
                 dt, sampling_steps) for dt in time_steps]
        results = pool.starmap(run_md_for_temperature, args)
    return results

def run_preliminary_steps_parallel(temperatures, seeds,
                   verlet_algorithms, algorithm_names, 
               time_steps, preliminary_steps):
    """Run preliminary steps sequentially."""
    results = []
    for alg in verlet_algorithms:
        alg_results = []
        for T, seed, dt in zip(temperatures, seeds, time_steps):
            result = run_md_for_temperature(T, seed, alg,
                                            dt, preliminary_steps)
            alg_results.append(result)
        results.append(alg_results)
    
    potential_energies_per_algorithm = []
    energy_drifts_per_algorithm = []
    best_time_steps = []

    for i, alg_results in enumerate(results):
        if len(alg_results) == 0:
            print(f"No results for algorithm {algorithm_names[i]}")  
            logging.warning(f"No results for algorithm {algorithm_names[i]}")
            
            continue
        potential_energies = [res[1] for res in alg_results]
        energy_drifts = [res[2] for res in alg_results]
        best_time_step = time_steps[np.argmin([res[2] for res in alg_results])]
        
        potential_energies_per_algorithm.append(np.mean(potential_energies))
        energy_drifts_per_algorithm.append(np.mean(energy_drifts))
        best_time_steps.append(best_time_step)
    
    return potential_energies_per_algorithm, energy_drifts_per_algorithm, best_time_steps



def find_best_sampling_steps(T, seed, verlet_algorithm,
                time_step, sampling_step_range):
    """Find the best sampling steps."""
    results = []
    for sampling_steps in sampling_step_range:
        dt, potential_energy, energy_drift = run_md_for_temperature(T, seed,
                            verlet_algorithm, time_step, sampling_steps)
        results.append((sampling_steps, potential_energy, energy_drift))
    return results






def find_critical_boiling_melting_points(temperatures, potential_energies):
    # Compute the first and second derivatives
    first_derivative = np.gradient(potential_energies, temperatures)
    second_derivative = np.gradient(first_derivative, temperatures)
    
    # Critical point: maximum in the second derivative
    critical_point_index = np.argmax(second_derivative)
    critical_point = temperatures[critical_point_index]

    # Boiling point: second highest in the second derivative before the critical point
    pre_critical_second_derivative = second_derivative[:critical_point_index]
    boiling_point_index = np.argmax(pre_critical_second_derivative)
    boiling_point = temperatures[boiling_point_index]

    # Melting point: maximum in the first derivative before the boiling point
    pre_boiling_first_derivative = first_derivative[:boiling_point_index]
    melting_point_index = np.argmax(pre_boiling_first_derivative)
    melting_point = temperatures[melting_point_index]

    return melting_point, boiling_point, critical_point








def plot_potential_energy_vs_temperature(temperatures, potential_energies, melting_point, boiling_point, critical_point):
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, potential_energies, marker='o', linestyle='-', label='Average Potential Energy (MD Simulation)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average Potential Energy (J)')
    plt.title('Average Potential Energy vs Temperature for Argon')

    if melting_point is not None:
        plt.axvline(melting_point, color='r', linestyle='--', label=f'Melting Point: {melting_point} K')
    if boiling_point is not None:
        plt.axvline(boiling_point, color='g', linestyle='--', label=f'Boiling Point: {boiling_point} K')
    if critical_point is not None:
        plt.axvline(critical_point, color='b', linestyle='--', label=f'Critical Point: {critical_point} K')
    
    plt.legend()
    plt.grid(True)
    plt.show()








def generate_pdf_report(melting_point, boiling_point, a_opt, temperatures, potential_energies, runtime, best_algorithm_name, best_time_step, reason):
    script_base_name = os.path.splitext(os.path.basename(__file__))[0]
    filename = os.path.join(script_dir, f"{script_base_name}.pdf")
    plot_filename = os.path.join(script_dir, f"{script_base_name}.png")

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.drawString(50, height - 50, "Molecular Dynamics Simulation Report")
    c.drawString(50, height - 70, f"Optimized Lattice Parameter: {a_opt:.4e} m")
    if melting_point is not None:
        c.drawString(50, height - 90, f"Melting Point: {melting_point} K")
    if boiling_point is not None:
        c.drawString(50, height - 110, f"Boiling Point: {boiling_point} K")
    c.drawString(50, height - 130, f"Total Runtime: {runtime:.2f} seconds")
    c.drawString(50, height - 150, f"Best Verlet Algorithm: {best_algorithm_name}")
    c.drawString(50, height - 170, f"Best Time Step: {best_time_step:.2e} s")

    reason_text = f"Reason: {reason}"
    max_chars_per_line = 78
    wrapped_reason = "\n".join([reason_text[i:i + max_chars_per_line] for i in range(0, len(reason_text), max_chars_per_line)])
    for i, line in enumerate(wrapped_reason.split('\n')):
        c.drawString(50, height - 190 - i * 20, line)

    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, potential_energies, marker='o', linestyle='-', label='Average Potential Energy per Atom (MD Simulation)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average Potential Energy per Atom (J)')
    plt.title('Average Potential Energy per Atom vs Temperature for Argon')
    if melting_point is not None:
        plt.axvline(melting_point, color='r', linestyle='--', label=f'Melting Point: {melting_point} K')
    if boiling_point is not None:
        plt.axvline(boiling_point, color='g', linestyle='--', label=f'Boiling Point: {boiling_point} K')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.close()

    try:
        c.drawImage(plot_filename, 50, 200, width=500, height=300)
    except FileNotFoundError:
        logging.error(f"File {plot_filename} not found. Skipping image in the PDF report.")

    c.save()



def main():
    """Main function to run the MD simulation."""
    start_time = time.time()

    a_opt = optimize_lattice_parameter()
    print(f"Optimized lattice parameter: {a_opt}")
    logging.info(f"Optimized lattice parameter: {a_opt}")

    temperatures = np.arange(5, 205, 5)
    preliminary_temperatures = np.arange(5, 205, 15)

    time_steps = [
        1e-15, 5e-16, 1e-16, 5e-15
    ]

    seeds = np.random.randint(0, 10000, len(temperatures))
    preliminary_seeds = np.random.randint(0, 10000, len(preliminary_temperatures))

    verlet_algorithms = [
        velocity_verlet_adaptive,
        standard_verlet_adaptive,
        leapfrog_verlet_adaptive
    ]
    algorithm_names = [
        "Velocity-Verlet (Adaptive)",
        "Standard Verlet (Adaptive)",
        "Leapfrog Verlet (Adaptive)"
    ]

    preliminary_steps = 8000
    potential_energies_per_algorithm, energy_drifts_per_algorithm, best_time_steps = run_preliminary_steps_parallel(
        preliminary_temperatures,
        preliminary_seeds,
        verlet_algorithms,
        algorithm_names, time_steps, preliminary_steps)

    if not potential_energies_per_algorithm or not energy_drifts_per_algorithm or not best_time_steps:
        print("Error: No valid preliminary results obtained.")
        logging.error("No valid preliminary results obtained.")
        return

    best_algorithm_index = np.argmin([pe + ed for pe, ed in zip(potential_energies_per_algorithm, energy_drifts_per_algorithm)])
    best_verlet_algorithm = verlet_algorithms[best_algorithm_index]
    best_algorithm_name = algorithm_names[best_algorithm_index]
    best_time_step = best_time_steps[best_algorithm_index]
    reason = (f"Lowest average potential energy ({potential_energies_per_algorithm[best_algorithm_index]:.2e} J) "
              f"and minimal energy drift ({energy_drifts_per_algorithm[best_algorithm_index]:.2e} J) "
              f"during preliminary simulation.")
    print(f"Best Verlet algorithm: {best_algorithm_name}")
    print(f"Best time step: {best_time_step}")

    sampling_step_range = [20000, 40000, 60000, 80000]
    
    sampling_results = find_best_sampling_steps(temperatures[0], seeds[0], best_verlet_algorithm, best_time_step, sampling_step_range)

    for sampling_steps, potential_energy, energy_drift in sampling_results:
        print(f"Sampling steps: {sampling_steps}, Potential energy: {potential_energy:.2e} J, Energy drift: {energy_drift:.2e} J")
        logging.info(f"Sampling steps: {sampling_steps}, Potential energy: {potential_energy:.2e} J, Energy drift: {energy_drift:.2e} J")

    best_sampling_steps = min(sampling_results, key=lambda x: x[2])[0]
    print(f"Best sampling steps: {best_sampling_steps}")
    logging.info(f"Best sampling steps: {best_sampling_steps}")

    with Pool(16) as pool:
        results = pool.starmap(run_md_for_temperature, [(T, seed, best_verlet_algorithm, best_time_step, best_sampling_steps) for T, seed in zip(temperatures, seeds)])

    dts, potential_energies, energy_drifts = zip(*results)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f} seconds")

    melting_point, boiling_point, critical_point = find_critical_boiling_melting_points(temperatures, potential_energies)

    logging.info(f"Detected Melting Point: {melting_point} K")
    logging.info(f"Detected Boiling Point: {boiling_point} K")
    logging.info(f"Detected Critical Point: {critical_point} K")

    plot_potential_energy_vs_temperature(temperatures, potential_energies, melting_point, boiling_point, critical_point)

    generate_pdf_report(melting_point, boiling_point, a_opt, temperatures, potential_energies, runtime, best_algorithm_name, best_time_step, reason)



if __name__ == "__main__":
    main()
