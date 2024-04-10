# Import necessary libraries for the game and statistical processing
import sys
import os
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from models.mc import simulate_trajectories_advanced
from models.mcmc import simulate_trajectories_mcmc_advanced

# Adjust the path for module imports to ensure our custom modules are found
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the FallingObjectsGame class from the Game package
from Game.falling_object import FallingObjectsGame, GameObject

# Define color constants used by the GameObject for easy reference and change
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0)]

# Define the size of the grid that the game will use
GRID_SIZE = 100

# Utility functions for running simulations and calculating accuracy
def get_initial_conditions(game_objects):
    """ Get the starting point and velocity for all objects in the game. """
    start_positions = [(obj.position[0], obj.position[1]) for obj in game_objects]
    initial_velocity = (1, 0)  # Example: assuming objects fall vertically down
    return start_positions, initial_velocity

# Function to convert actual landing positions into a probability distribution
def get_actual_distribution(actual_positions, grid_size):
    """ Converts a list of actual landing positions into a normalized 2D histogram (probability distribution). """
    # Initialize a 2D array for counts, focusing only on the last row for landings
    counts = np.zeros(grid_size)
    
    # Increment the count for each actual landing position
    for y, x in actual_positions:
        if y == grid_size - 1:  # Only count landings on the bottom row
            counts[int(x)] += 1
    
    # Normalize the counts to get a probability distribution
    probability_distribution = counts / np.sum(counts)
    return probability_distribution

def run_simulations(start_positions, initial_velocity, grid_size, num_simulations, wind_variance):
    """ Perform the simulations using both MC and MCMC methods and get the results. """
    mc_predictions = simulate_trajectories_advanced(start_positions, initial_velocity, grid_size, num_simulations)
    # Only get the probability distribution from MCMC, not the out of bounds count
    mcmc_predictions, _ = simulate_trajectories_mcmc_advanced(start_positions, initial_velocity, grid_size, num_simulations, wind_variance)
    return mc_predictions, mcmc_predictions

def create_probability_df(probability_field, grid_size):
    """ Convert the probability array to a DataFrame for better data manipulation. """
    return pd.DataFrame(probability_field, index=[f'Row {i}' for i in range(grid_size)], columns=[f'Col {i}' for i in range(grid_size)])

# Functions to calculate and compare how close the simulations are to actual results
def calculate_mse(predicted, actual):
    """ Compute the mean squared error to quantify the difference between predicted and actual results. """
    return ((predicted - actual) ** 2).mean()

def calculate_kl_divergence(predicted, actual):
    """ Calculate the Kullback-Leibler divergence between two distributions. """
    # Ensure the predicted array is 1D and has the same length as actual
    predicted = predicted.ravel()[:len(actual)]
    predicted = np.clip(predicted, 1e-10, 1)  # Avoid division by zero
    actual = np.clip(actual, 1e-10, 1)
    return entropy(actual, predicted)

def visualize_results(mc_predictions, mcmc_bottom_row_distribution, out_of_bounds_count, actual_positions, grid_size):
    # Proceed with visualization only if mcmc_bottom_row_distribution is an array
    if isinstance(mcmc_bottom_row_distribution, np.ndarray):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # MC Predictions for bottom row distribution
        axs[0].bar(range(grid_size), mc_predictions[-1], color='blue', alpha=0.7)
        axs[0].set_title("MC Bottom Row Distribution")
        axs[0].set_xlabel('Position along bottom layer')
        axs[0].set_ylabel('Probability')

        # MCMC Predictions for bottom row distribution
        axs[1].bar(range(grid_size), mcmc_bottom_row_distribution, color='orange', alpha=0.7)
        axs[1].set_title("MCMC Bottom Row Distribution")
        axs[1].set_xlabel('Position along bottom layer')

        # Out of Bounds Count
        axs[2].bar(['Out of Bounds'], [out_of_bounds_count], color='red')
        axs[2].set_title("Out of Bounds Count")
        axs[2].set_ylabel('Count')

        plt.tight_layout()
        plt.show()
    else:
        print("Cannot visualize results because MCMC bottom row distribution is not an array.")



# Initialize the game and run the main loop
game = FallingObjectsGame()
actual_positions = []  # List to keep track of where objects actually land

# Run the game until the user closes the window
running = True
while running:
    game.run_step()

    # Here we would collect actual landing positions
    # This is just placeholder logic
    if len(actual_positions) < 5:
        actual_positions.append((99, 20 * len(actual_positions)))

    # Handle quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

game.close()  # Close the game window

# Run simulations for the initial state of your game objects
start_positions, initial_velocity = get_initial_conditions(game.objects)
mc_predictions = simulate_trajectories_advanced(start_positions, initial_velocity, GRID_SIZE, 1000)

# mcmc_predictions should be unpacked from the result
mcmc_bottom_row_distribution, out_of_bounds_count = simulate_trajectories_mcmc_advanced(start_positions, initial_velocity, GRID_SIZE, 1000, 0.1)


# Convert actual positions to a probability distribution (not implemented here)
actual_distribution = get_actual_distribution(actual_positions, GRID_SIZE)

# Calculate how well our simulations did compared to actual results
mc_mse, mc_kl = calculate_mse(mc_predictions, actual_distribution), calculate_kl_divergence(mc_predictions, actual_distribution)
mcmc_mse, mcmc_kl = calculate_mse(mcmc_bottom_row_distribution, actual_distribution), calculate_kl_divergence(mcmc_bottom_row_distribution, actual_distribution)

# Print out the error metrics for analysis
print(f"MC Mean Squared Error: {mc_mse}")
print(f"MC KL Divergence: {mc_kl}")
print(f"MCMC Mean Squared Error: {mcmc_mse}")
print(f"MCMC KL Divergence: {mcmc_kl}")

# Visualize the simulation results and compare with actual game outcomes
# Visualize the results
visualize_results(mc_predictions, mcmc_bottom_row_distribution, out_of_bounds_count, actual_positions, GRID_SIZE)


