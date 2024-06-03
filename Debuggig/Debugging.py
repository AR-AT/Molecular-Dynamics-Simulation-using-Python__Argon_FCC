import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
#from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tqdm import tqdm
from numba import njit, prange


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
            velocities, forces, dt, box_length, max_force_threshold=1e-11):
    """Velocity-Verlet integration with adaptive time step."""
    new_positions = positions + velocities * dt + 0.5 * forces * dt**2 / MASS
    new_positions = apply_pbc(new_positions, box_length)
    new_forces, _ = compute_forces(new_positions, box_length)
    max_force = np.max(custom_norm(new_forces, axis=1))
    
    if max_force > max_force_threshold:
        dt = max(0.9 * dt * (max_force_threshold / max_force)**0.5, 1e-15)
    
    new_velocities = velocities + 0.5 * (forces + new_forces) * dt / MASS
    return new_positions, new_velocities, new_forces, dt


@njit
def standard_verlet_adaptive(positions,
    velocities, forces, dt, box_length, max_force_threshold=5e-11):
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
    velocities, forces, dt, box_length, max_force_threshold=5e-11):
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


def adaptive_equilibration(positions, velocities,
                           forces, dt, box_length,
                           verlet_algorithm,
                           equilibration_window=150, threshold=1e-23):
    """Perform adaptive equilibration."""
    equilibration_steps = 0
    recent_potential_energies = []
    forces, _ = compute_forces(positions, box_length)
    with tqdm(total=equilibration_window, desc="Equilibration") as pbar:
        while True:
            positions, velocities, forces, dt = verlet_algorithm(positions,
                velocities, forces, dt, box_length)
            _, potential_energy = compute_forces(positions, box_length)
            recent_potential_energies.append(potential_energy)
            if len(recent_potential_energies) > equilibration_window:
                recent_potential_energies.pop(0)
                mean_potential_energy = np.mean(recent_potential_energies)
                if np.abs(potential_energy - mean_potential_energy) < threshold:
                    break
                pbar.update(1)
            equilibration_steps += 1
            
           # if equilibration_steps % 1000 == 0:
           #     print(f"Step {equilibration_steps}, dt: {dt:.2e}, Potential Energy: {potential_energy:.5e}")
    return positions, velocities, forces, dt, equilibration_steps

def run_md_for_temperature(T, seed, verlet_algorithm, dt, sampling_steps=60000):
    """Run MD simulation for a single temperature using the specified Verlet algorithm."""
    
   
    np.random.seed(seed)
    a = 5.26e-10  # Initial guess for the lattice parameter
    n = 3  # Number of unit cells in each dimension
    positions = create_fcc_lattice(a, n)
    box_length = n * a  # Length of the simulation box
    N = len(positions)
    
    # Initialize velocities for 5K
    velocities = np.random.randn(N, 3) * np.sqrt(K_B * 5 / MASS)
    forces, _ = compute_forces(positions, box_length)

    # Rescale velocities to the desired temperature
    current_temperature = np.sum(MASS * np.sum(velocities**2, axis=1)) / (3 * N * K_B)
    scale_factor = np.sqrt(T / current_temperature)
    velocities *= scale_factor

    # Adaptive equilibration
    positions, velocities, forces, dt, equilibration_steps = adaptive_equilibration(
        positions, velocities, forces, dt, box_length, verlet_algorithm
    )
    print(f"Equilibration steps: {equilibration_steps}")

    potential_energy = 0
    initial_total_energy = 0
    count = 0
    total_energy_drift = 0

    with tqdm(total=sampling_steps, desc="MD Simulation") as pbar:
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
    
        average_potential_energy = potential_energy / count
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
    initial_guesses = np.linspace(5e-10, 6e-10, 300)
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

def find_best_sampling_steps(T, seed, verlet_algorithm,
                time_step, sampling_step_range):
    """Find the best sampling steps."""
    results = []
    for sampling_steps in sampling_step_range:
        dt, potential_energy, energy_drift = run_md_for_temperature(T, seed,
                            verlet_algorithm, time_step, sampling_steps)
        results.append((sampling_steps, potential_energy, energy_drift))
    return results

def generate_pdf_report(filename,
                        melting_point, boiling_point, a_opt,
                        temperatures, potential_energies, runtime,
                        best_algorithm_name, best_time_step, reason):
    """Generate a PDF report."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.drawString(50, height - 50, "Molecular Dynamics Simulation Report")
    c.drawString(50, height - 70, f"Optimized Lattice Parameter: {a_opt:.4e} m")
    c.drawString(50, height - 90, f"Melting Point: {melting_point} K")
    c.drawString(50, height - 110, f"Boiling Point: {boiling_point} K")
    c.drawString(50, height - 130, f"Total Runtime: {runtime:.2f} seconds")
    c.drawString(50, height - 150, f"Best Verlet Algorithm: {best_algorithm_name}")
    c.drawString(50, height - 170, f"Best Time Step: {best_time_step:.2e} s")

    reason_text = f"Reason: {reason}"
    max_chars_per_line = 78
    wrapped_reason = "\n".join([reason_text[i:i+max_chars_per_line] for i in range(0,
                            len(reason_text), max_chars_per_line)])
    text_lines = wrapped_reason.split('\n')
    for i, line in enumerate(text_lines):
        c.drawString(50, height - 190 - i*20, line)

    plot_filename = "temp_plot2.png"
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, potential_energies,
    marker='o', linestyle='-', label='Average Potential Energy (MD Simulation)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average Potential Energy (J)')
    plt.title('Average Potential Energy vs Temperature for Argon')
    plt.axvline(melting_point, color='r', linestyle='--',
        label=f'Melting Point: {melting_point} K')
    plt.axvline(boiling_point, color='g', linestyle='--',
        label=f'Boiling Point: {boiling_point} K')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.close()

    c.drawImage(plot_filename, 50, 200, width=500, height=300)
    c.save()

def main():
    
    """Main function to run the MD simulation."""
    start_time = time.time()  # Start time measurement

    a_opt = optimize_lattice_parameter()
    print(f"Optimized lattice parameter: {a_opt}")

    temperatures = np.arange(5, 205, 5)
    time_steps = [1e-15]  # Different time steps to test
    seeds = np.random.randint(0, 10000, len(temperatures))

    verlet_algorithms = [velocity_verlet_adaptive]
                   #      standard_verlet_adaptive, leapfrog_verlet_adaptive]
    algorithm_names = ["Velocity-Verlet (Adaptive)"]
               # "Standard Verlet (Adaptive)", "Leapfrog Verlet (Adaptive)"]

    preliminary_steps = 100
    potential_energies_per_algorithm = []
    energy_drifts_per_algorithm = []
    best_time_steps = []

    for algorithm in verlet_algorithms:
        potential_energies = []
        energy_drifts = []
        for T, seed in zip(temperatures[:3], seeds[:3]):
            results = test_time_steps(T, seed, algorithm, time_steps,
                                      preliminary_steps)
            for dt, potential_energy, energy_drift in results:
                potential_energies.append(potential_energy)
                energy_drifts.append(energy_drift)
            best_time_step = min(results, key=lambda x: x[2])[0]
            best_time_steps.append(best_time_step)
        potential_energies_per_algorithm.append(np.mean(potential_energies))
        energy_drifts_per_algorithm.append(np.mean(energy_drifts))

    best_algorithm_index = np.argmin([pe + ed for pe,
                                ed in zip(potential_energies_per_algorithm,
                                                    energy_drifts_per_algorithm)])
    best_verlet_algorithm = verlet_algorithms[best_algorithm_index]
    best_algorithm_name = algorithm_names[best_algorithm_index]
    best_time_step = best_time_steps[best_algorithm_index]
    reason = (f"Lowest average potential energy ({potential_energies_per_algorithm[best_algorithm_index]:.2e} J) "
              f"and minimal energy drift ({energy_drifts_per_algorithm[best_algorithm_index]:.2e} J) "
              f"during preliminary simulation.")
    print(f"Best Verlet algorithm: {best_algorithm_name}")
    print(f"Best time step: {best_time_step}")

    sampling_step_range = [10000]
    sampling_results = find_best_sampling_steps(temperatures[0],
                seeds[0], best_verlet_algorithm, best_time_step, sampling_step_range)

    for sampling_steps, potential_energy, energy_drift in sampling_results:
        print(f"Sampling steps: {sampling_steps}, Potential energy: {potential_energy:.2e} J, Energy drift: {energy_drift:.2e} J")

    best_sampling_steps = min(sampling_results, key=lambda x: x[2])[0]
    print(f"Best sampling steps: {best_sampling_steps}")

    with Pool(16) as pool:
        results = pool.starmap(run_md_for_temperature,
                               [(T, seed, best_verlet_algorithm,
                                 best_time_step, best_sampling_steps)
                                for T, seed in zip(temperatures, seeds)])
    
    dts, potential_energies, energy_drifts = zip(*results)
    
    end_time = time.time()  # End time measurement
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f} seconds")
    
    first_derivative = np.gradient(potential_energies, temperatures)
    second_derivative = np.gradient(first_derivative, temperatures)
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, potential_energies,
             marker='o', linestyle='-',
             label='Average Potential Energy (MD Simulation)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Average Potential Energy (J)')
    plt.title('Average Potential Energy vs Temperature for Argon')
    plt.legend()
    plt.grid(True)
    
    melting_point = temperatures[np.argmax(first_derivative)]
    
    post_melting_temperatures = temperatures[temperatures > melting_point]
    post_melting_second_derivative = second_derivative[temperatures > melting_point]
    boiling_point = post_melting_temperatures[np.argmax(post_melting_second_derivative)]
    
    plt.axvline(melting_point,
                color='r', linestyle='--',
                label=f'Melting Point: {melting_point} K')
    plt.axvline(boiling_point,
                color='g', linestyle='--',
                label=f'Boiling Point: {boiling_point} K')
    plt.legend()
    plt.show()
    
    generate_pdf_report("MD_Simulation_Report2(removed smoothing).pdf",
    melting_point, boiling_point, a_opt,
    temperatures, potential_energies,  # This should be the list of average potential energies
    runtime, best_algorithm_name, best_time_step, reason)


if __name__ == "__main__":
    main()
