import numpy as np

def simulate_trajectories_mcmc_advanced(start_positions, initial_velocity, grid_size, num_simulations, wind_variance, gravity=9.81):
    landing_positions = np.zeros((grid_size, grid_size))
    out_of_bounds_count = 0
    dt = 0.1

    for start_pos in start_positions:
        for _ in range(num_simulations):
            pos = np.array(start_pos, dtype=float)
            vel = np.array(initial_velocity, dtype=float)

            while True:
                vel[1] += np.random.normal(0, wind_variance) * dt
                vel[0] += gravity * dt
                pos += vel * dt

                if pos[1] < 0 or pos[1] >= grid_size:
                    out_of_bounds_count += 1
                    break
                
                if pos[0] >= grid_size:
                    x_index = int(np.clip(pos[1], 0, grid_size - 1))
                    landing_positions[grid_size - 1, x_index] += 1
                    break

    bottom_row_probability = landing_positions[-1, :] / np.sum(landing_positions[-1, :])
    return bottom_row_probability, out_of_bounds_count
