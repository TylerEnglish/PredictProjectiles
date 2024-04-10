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

    def proposal_distribution(self, current_state, action):
        # This is just a placeholder function. You'll need to replace this with your actual logic.
        # For example, you might add the action to the current state to get the proposed state.
        return current_state + action
    
    def calculate_proposed_state(self, current_state, action):
        if current_state is None:
            current_state = 0  # Or some other appropriate value
        return current_state + action

    def transition(self, current_state, action):
        proposed_state = self.calculate_proposed_state(current_state, action)
        if proposed_state is not None:
            self.current_state = proposed_state if np.random.uniform() < self.target_distribution(proposed_state) else current_state
        else:
            raise ValueError("Proposed state is None.")