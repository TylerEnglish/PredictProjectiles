# Markov Chain Monte Carlo (MCMC) Implementation

## Introduction to MCMC

Markov Chain Monte Carlo (MCMC) methods are a class of algorithms used to sample from probability distributions based on constructing a Markov chain that has the desired distribution as its equilibrium distribution. The state of the chain after a number of steps is then used as a sample of the desired distribution. Typically used in contexts where direct sampling is challenging, MCMC methods are crucial in Bayesian statistics, machine learning, and data science.

## Key Concepts

- **Markov Chain**: A stochastic model describing a sequence of possible events in which the probability of each event depends only on the state attained in the previous event.
- **Monte Carlo Simulation**: A computational algorithm that relies on repeated random sampling to obtain numerical results. The underlying concept is to use randomness to solve problems that might be deterministic in principle.
- **Proposal Distribution**: A function used to propose new states in the Markov chain. This function is crucial for the efficiency of the MCMC method.
- **Target Distribution**: The distribution from which we want to sample. The goal of MCMC is to create a Markov chain that has this target distribution as its equilibrium distribution.

## Model Definition

[Model](../../models/b_qnet_mcmc.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

class MCMC:
    def __init__(self, proposal_distribution, target_distribution):
        """
        Initialize the MCMC instance.

        Parameters
        ----------
        proposal_distribution: function
            The proposal distribution Q(i, j) for transitioning from state i to state j.

        target_distribution: function
            The target distribution P(i) that we want to sample from.
        """
        if not callable(proposal_distribution) or not callable(target_distribution):
            raise ValueError("Both proposal_distribution and target_distribution must be callable.")
        
        self.proposal_distribution = proposal_distribution
        self.target_distribution = target_distribution
        self.current_state = None

    def metropolis_hastings(self, initial_state, num_samples):
        """
        Metropolis-Hastings algorithm for MCMC.

        Parameters
        ----------
        initial_state: int
            The initial state of the Markov Chain.

        num_samples: int
            The number of samples to generate.
        """
        self.current_state = initial_state
        samples = []

        for _ in range(num_samples):
            proposed_state = self.proposal_distribution(self.current_state)
            acceptance_probability = min(
                1,
                (self.target_distribution(proposed_state) * self.proposal_distribution(proposed_state, self.current_state)) /
                (self.target_distribution(self.current_state) * self.proposal_distribution(self.current_state, proposed_state))
            )

            if np.random.uniform() < acceptance_probability:
                self.current_state = proposed_state

            samples.append(self.current_state)

        return samples
```
## Explain the Code in Simple Terms

The code defines two classes: `DQN` and `MCMC`. Each class serves a specific function in modeling decision-making processes or statistical distributions. Here’s a breakdown of each part:

- **DQN Class**: This class implements a deep Q-network, a type of neural network used in reinforcement learning to approximate the Q-value function, which helps an agent learn which actions to take in which states. The structure includes:
  - **Fully Connected Layers**: Layers where each neuron is connected to every neuron in the previous layer, processing the input data to capture complex patterns.
  - **Dropout**: This is a regularization technique where randomly selected neurons are ignored during training to prevent overfitting and to promote model generalization.
  - **ReLU Activation**: A nonlinear operation used after each fully connected layer except the last one to introduce nonlinearity into the model, helping it to learn nonlinear relationships in the data.
  
- **MCMC Class**: This class is designed to perform the Markov Chain Monte Carlo using the Metropolis-Hastings algorithm to sample from a probability distribution (target distribution) that is difficult to sample from directly.
  - **Initialization**: It starts with defining the proposal distribution and the target distribution. Both must be callable functions.
  - **Metropolis-Hastings Algorithm**:
    - **Proposal Step**: Proposes a new state in the Markov chain.
    - **Acceptance Probability**: Calculates the probability of moving to the new state based on the ratio of the target distributions and the proposal distributions at the new and current states.
    - **Accept or Reject**: Decides whether to move to the new state based on the acceptance probability using a random comparison.

This setup effectively allows sampling from complex distributions by navigating the state space using the proposal distribution and making acceptance decisions that aim to simulate the equilibrium distribution of the Markov chain (target distribution).

## Conclusion

The `MCMC` implementation provided facilitates sampling from complex probability distributions, which is particularly useful in contexts such as Bayesian statistics where direct sampling is impractical. The Metropolis-Hastings algorithm, a cornerstone of this implementation, ensures that the chain converges to the desired target distribution, allowing for the estimation of distribution characteristics or model parameters that are otherwise difficult to ascertain. This model is crucial for performing statistical inference in various scientific fields, demonstrating the power and versatility of MCMC methods.

