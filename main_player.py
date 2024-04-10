import torch
from Game.PlayerSection.basic import *


def run_game(agent, model, model_path, num_episodes, training_mode):
    physics = Physics()
    collision_manager = CollisionManager()
    player = Player(PLAYER_INITIAL_HEALTH)

    if training_mode:
        for i in range(num_episodes):
            print(f'Episode num {i+1}/{num_episodes}')
            game = Game(physics, collision_manager, player, agent, training_mode, 60)
            game.run()

        # Save the trained model if in training mode
        torch.save(model.state_dict(), model_path)
        print(f"Model saved successfully at {model_path}")
    else:
        game = Game(physics, collision_manager, player, agent, training_mode, 6000)
        game.run()

state_size = 4
action_size = 3
model = DQN(state_size, action_size)
agent = Agent(state_size, action_size, model)
model_path = 'data/models/trained_model.pth'
run_game(agent, model, model_path, 0, False)
