import sys
import os

# Adjust the path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from models.mc import simulate_trajectories_advanced  # Use the updated function name


def test_probability_sum():
    # Test to ensure the sum of probabilities equals 1
    start_positions = [(0, 50)]
    velocity = (1, 0)
    grid_size = 100
    num_simulations = 1000
    probability_field = simulate_trajectories_advanced(start_positions, velocity, grid_size, num_simulations, gravity=9.81, wind_resistance_factor=0.1)
    assert np.isclose(np.sum(probability_field), 1), "Total probability should sum to 1"

def test_edge_conditions():
    # Test to ensure no probabilities exist beyond the grid edges
    start_positions = [(0, 0)]
    velocity = (1, 0)
    grid_size = 100
    num_simulations = 1000
    probability_field = simulate_trajectories_advanced(start_positions, velocity, grid_size, num_simulations, gravity=9.81, wind_resistance_factor=0.1)
    assert np.all(probability_field[:, -1] == 0), "No probabilities should exist beyond the right edge of the grid"
    assert np.any(probability_field[-1, :] > 0), "Objects are expected to reach the bottom row of the grid"

if __name__ == "__main__":
    test_probability_sum()
    test_edge_conditions()
    print("MC Tests passed.")
