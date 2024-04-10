# Markov Chain Model Implementation

## Introduction to Markov Chains

A Markov Chain is a statistical model that predicts a sequence of possible events based on the probabilities of previous events. It assumes the Markov Property, where the probability of moving to the next state depends only on the present state and not on the previous states. This property simplifies the computation and is particularly useful in the field of reinforcement learning, where future states depend only on the current state and the decision-maker's action.

## Key Concepts

- **States**: These are distinct conditions or positions in which the system or agent can be. In a reinforcement learning context, a state represents every possible scenario that the agent might observe from the environment.
- **Actions**: Decisions or moves made by the agent that result in transitioning from one state to another.
- **State Transition Matrix**: A square matrix used to describe the probabilities of moving from one state to another state in a given action. Each entry in the matrix represents a probability conditioned on actions taken in the current state.

## Model Definition

[Model](../../models/b_qnet_mc.py)

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
```

## Explain the Code in Simple Terms

The code defines two classes: `DQN` and `MarkovChain`. Let's break down what each part of the code does in simple terms:

- **DQN Class**: This is a type of neural network designed to help in decision-making processes. It has layers (think of them as steps) where each layer processes the information a bit more. The `dropout` parts are like saying, "Let's not rely too much on any one piece of information," preventing the model from memorizing and instead helping it to generalize better. The `forward` method is where the actual decision-making process happens based on the inputs it receives.

- **MarkovChain Class**: This class simulates a simple decision-making process where each decision leads to a new situation (state), and this outcome only depends on the current situation, not the past ones. It's like choosing a path in a maze based only on where you are right now, not how you got there. The `transition_matrix` is a way of saying, "If I am in this situation and make this decision, here are the chances of each possible outcome." The methods `next_state`, `generate_states`, and `transition` are tools for moving through our decision maze, deciding what to do next, predicting future paths, or changing our current situation based on our actions.

## Conclusion

In essence, the provided code is a toolkit for making and predicting decisions in a structured way, where our options (actions) in any situation (state) can lead to new situations. The `DQN` class uses a sophisticated form of decision-making that can learn from complex environments, making it ideal for situations where decisions are not straightforward. The `MarkovChain` class offers a simpler, probabilistic way of making decisions, where the outcome is based on known chances. Both can be used in artificial intelligence to navigate through and make choices in environments that mimic real-world complexity, helping to solve problems where predicting the best next move is crucial.
