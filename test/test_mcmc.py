import sys
import os

# Adjust the path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from models.mcmc import simulate_trajectories_mcmc_advanced



def test_variance_effect_bottom_layer_assertions():
    start_positions = [(0, 50)]
    initial_velocity = (1, 0)
    grid_size = 100
    num_simulations = 1000
    low_variance = 0.1
    high_variance = 20

    # Simulate for low and high variance
    bottom_row_prob_low, out_of_bounds_low = simulate_trajectories_mcmc_advanced(
        start_positions, initial_velocity, grid_size, num_simulations, low_variance, gravity=9.81)
    bottom_row_prob_high, out_of_bounds_high = simulate_trajectories_mcmc_advanced(
        start_positions, initial_velocity, grid_size, num_simulations, high_variance, gravity=9.81)

    assert np.isclose(np.sum(bottom_row_prob_low), 1), "Sum of low variance bottom row probabilities should be 1"
    assert np.isclose(np.sum(bottom_row_prob_high), 1), "Sum of high variance bottom row probabilities should be 1"
    assert out_of_bounds_low >= 0, "Out-of-bounds count should be non-negative for low variance"
    assert out_of_bounds_high >= 0, "Out-of-bounds count should be non-negative for high variance"

    print("All assertions passed for variance effect on bottom layer.")

if __name__ == "__main__":
    test_variance_effect_bottom_layer_assertions()