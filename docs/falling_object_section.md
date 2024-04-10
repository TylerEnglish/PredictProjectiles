# Falling Object Section

## Overview
This project simulates falling objects in a game environment and applies statistical simulations to predict landing positions using Monte Carlo (MC) and Markov Chain Monte Carlo (MCMC) techniques. It integrates game development with advanced statistical modeling to analyze and predict outcomes based on predefined physical rules.

## FallingObjectsGame
The core of the game is defined in `FallingObjectsGame`, which manages game objects, their behaviors, and interactions. Objects are spawned, updated, and rendered within a defined environment. The game operates on a grid, with each object's movement influenced by gravity and wind.

### Key Components:
- **GameObject**: Represents individual objects with properties such as position, velocity, and size. It's responsible for updating its state based on game physics.
- **Game Loop**: Handles the creation of objects, updates to their state, and rendering. The game loop also detects when objects are out of bounds.

## Simulation Models
Two simulation models are used to predict the landing positions of the objects: MC and MCMC. These models simulate the trajectory of objects under the influence of gravity and wind, aiming to estimate the probability distribution of landing positions.

### MC Simulation
Applies a straightforward approach to simulate object trajectories by directly applying gravity and wind resistance. It provides a basic prediction model for where objects might land.

### MCMC Simulation
Introduces randomness into the simulation to account for variability in wind strength and direction. It's more advanced and provides a probabilistic view of potential landing positions, taking into account the uncertainty introduced by wind variance.

## Main Logic
The main script ties the game and simulations together. It extracts initial conditions from the game objects, runs both MC and MCMC simulations based on these conditions, and analyzes the accuracy of these simulations compared to the actual game outcomes.

### Visualization
To understand the effectiveness of the predictions, visualizations are generated comparing the actual landing positions against the simulated probability distributions. Additionally, the script calculates error metrics (MSE and KL divergence) to quantitatively assess the prediction accuracy.

### Out of Bounds Handling
A notable feature of the MCMC simulation is its ability to track objects that go out of bounds. This aspect is crucial for understanding the impact of wind variance and for refining the simulation parameters to enhance prediction accuracy.

## Conclusion
This project demonstrates the application of statistical simulations within a game environment to predict outcomes. Through MC and MCMC models, it provides insights into the dynamics of falling objects influenced by physical forces, offering a blend of game development and statistical analysis.