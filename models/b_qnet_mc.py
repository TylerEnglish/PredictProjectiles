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
    
class MarkovChain:
    def __init__(self, state_size, action_size):
        """
        Initialize the MarkovChain instance.

        Parameters
        ----------
        state_size: int
            The number of states in the Markov Chain.

        action_size: int
            The number of actions in the Markov Chain.
        """
        self.states = [i for i in range(state_size)]
        self.actions = [i for i in range(action_size)]
        # Initialize the transition matrix with uniform probabilities
        self.transition_matrix = np.full((state_size, action_size, state_size), 1/state_size)
        self.index_dict = {self.states[index]: index for index in range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in range(len(self.states))}
        self.current_state = self.states[0]

    def next_state(self, current_state, action):
        """
        Returns the state of the random variable at the next time 
        instance.

        Parameters
        ----------
        current_state: int
            The current state of the system.

        action: int
            The action taken by the agent.
        """
        return np.random.choice(
         self.states, 
         p=self.transition_matrix[current_state, action, :]
        )

    def generate_states(self, current_state, action, no=10):
        """
        Generates the next states of the system.

        Parameters
        ----------
        current_state: int
            The state of the current random variable.

        action: int
            The action taken by the agent.

        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state, action)
            future_states.append(next_state)
            current_state = next_state
        return future_states
    
    def transition(self, current_state, action):
        """
        Transition to the next state based on the current state and action.

        Parameters
        ----------
        current_state: int
            The current state of the system.

        action: int
            The action taken by the agent.
        """
        # Normalize the transition probabilities
        transition_probs = self.transition_matrix[current_state, action, :]
        transition_probs /= (transition_probs.sum() + 1e-10)  # Add a small constant to avoid division by zero

        self.current_state = np.random.choice(
            self.states, 
            p=transition_probs
        )
        return self.current_state