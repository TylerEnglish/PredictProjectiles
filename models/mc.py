import numpy as np

def simulate_trajectories_advanced(start_positions, velocity, grid_size, num_simulations, gravity=9.81, wind_resistance_factor=0.1):
    # Initialize a grid to record where each object lands
    landing_positions = np.zeros((grid_size, grid_size))
    dt = 0.1  # Define a time step for the simulation

    # Loop over each starting position to simulate multiple trajectories
    for start_pos in start_positions:
        for _ in range(num_simulations):
            # Initialize position and velocity
            pos = np.array(start_pos, dtype=float)
            vel = np.array(velocity, dtype=float)
            
            # Keep updating the object's position until it would land
            while pos[0] < grid_size - 1:  # Prevent going out of the bottom edge
                vel[0] += gravity * dt  # Apply gravity to the y-velocity
                vel[1] *= (1 - wind_resistance_factor)  # Apply wind resistance to the x-velocity

                pos += vel * dt  # Update position with the new velocity
                
                # If object hits a side wall, it should bounce back
                if pos[1] < 0 or pos[1] >= grid_size:
                    vel[1] = -vel[1]
                pos[1] = np.clip(pos[1], 0, grid_size - 1)

            # Record the landing position
            y_index = min(int(pos[0]), grid_size - 1)
            x_index = int(pos[1])
            landing_positions[y_index, x_index] += 1

    ## Normalize the counts to get a probability distribution of landing positions
    probability_field = landing_positions / np.sum(landing_positions)
    return probability_field

