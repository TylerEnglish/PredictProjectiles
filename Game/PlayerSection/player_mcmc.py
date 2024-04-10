import sys
import os

# Adjust the path for module imports to reach the "Folder" level
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This navigates up to "Game"
grandparent_dir = os.path.dirname(parent_dir)  # This navigates up to "Folder"
sys.path.append(grandparent_dir)

# Model pathing
model_directory = os.path.join(grandparent_dir, 'data', 'models')
model_filename = 'mcmc_trained_model.pth'
model_path = os.path.join(model_directory, model_filename)

# Create the model path if not made
# Ensure the 'data/models' directory exists
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    print(f"Created model directory at: {model_directory}")

import pygame
import random
import numpy as np
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from models.b_qnet_mcmc import DQN, MCMC
from collections import deque




# Screen and Grid Setup
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
GRID_SIZE = 100
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

# Colors and Physics Constants
BLACK = (0, 0, 0)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0)]
WHITE = (255, 255, 255)
PLAYER_SPEED = 5
PROJECTILE_SPEED = 10
INITIAL_SPAWN_RATE = .1
SPAWN_ACCELERATION = 0.995
PLAYER_SIZE = 2
SHOOTING_COOLDOWN = 2
PLAYER_INITIAL_HEALTH = 100

class Vector2D(np.ndarray):
    """Simple Vector2D class extending numpy ndarray for physics calculations."""
    def __new__(cls, x, y):
        obj = np.asarray([float(x), float(y)]).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

class Physics:
    """A simple physics class for handling gravity and wind."""
    def __init__(self, gravity=0.1, wind=0.1, max_velocity=5):
        self.gravity = Vector2D(0, gravity)
        self.wind = Vector2D(wind, 0)
        self.max_velocity = max_velocity

    def apply_gravity(self, velocity):
        """Apply gravity to the y component of velocity."""
        velocity += self.gravity
        velocity[1] = min(velocity[1], self.max_velocity)
        return velocity

    def apply_wind(self, velocity):
        """Apply wind to the x component of velocity randomly."""
        wind_effect = self.wind if random.random() < 0.5 else -self.wind
        new_velocity = velocity + wind_effect
        return new_velocity.astype(float)  # Ensure the type is float

    def calculate_damage(self, size):
        """Calculate damage based on the size of the object and physics constants."""
        base_damage = 25  # This is a new base damage that you can adjust
        return base_damage * size * self.gravity[1]  # Now damage scales with object size and gravity

class GameObject:
    """Base class for all game objects with common properties."""
    def __init__(self, position, velocity, color, size):
        self.position = Vector2D(*position)
        self.velocity = Vector2D(*velocity)
        self.color = color
        self.size = size

    def update(self, physics):
        """Update object position based on its velocity and apply physics."""
        self.velocity = physics.apply_gravity(self.velocity)
        self.position += self.velocity
        self.position[0] = np.clip(self.position[0], 0, SCREEN_WIDTH - self.size * CELL_SIZE)

    def draw(self, screen):
        """Draw the object on the screen."""
        pygame.draw.rect(screen, self.color, (self.position[0], self.position[1], CELL_SIZE * self.size, CELL_SIZE * self.size))

    def is_out_of_bounds(self):
        """Check if the object is out of screen bounds."""
        return self.position[1] >= SCREEN_HEIGHT

class FallingObject(GameObject):
    """FallingObject inherits from GameObject and represents falling objects."""
    def __init__(self, position, velocity, color, size):
        super().__init__(position, velocity, color, size)
        self.damage = 0

    def update(self, physics):
        """Update falling object position, apply physics and wind."""
        self.velocity = physics.apply_wind(self.velocity)
        # The damage calculation could potentially be moved here if it depends on other dynamic factors
        self.damage = physics.calculate_damage(self.size)
        super().update(physics)
    
    def is_out_of_bounds(self):
        """Check if the object has fallen off the bottom of the screen."""
        # Just compare the position directly without multiplying by CELL_SIZE
        return self.position[1] >= SCREEN_HEIGHT

class Projectile(GameObject):
    """Projectile inherits from GameObject and represents fired projectiles."""
    def __init__(self, position):
        super().__init__(position, Vector2D(0, -PROJECTILE_SPEED), WHITE, 0.5)

    def update(self, physics):
        """Update projectile position without applying gravity."""
        self.position += self.velocity

class Player(GameObject):
    """Player inherits from GameObject and represents the player character."""
    def __init__(self, health):
        super().__init__(Vector2D(SCREEN_WIDTH // 2, SCREEN_HEIGHT - CELL_SIZE), Vector2D(0, 0), WHITE, PLAYER_SIZE)
        self.health = health
        self.objects_avoided = 0
        self.last_shot_time = 0
        self.objects_hit = 0

    def move(self, direction):
        """Move the player horizontally with bounds checking."""
        # Update the x position directly through indexing
        self.position[0] += direction * PLAYER_SPEED
        self.position[0] = np.clip(self.position[0], 0, SCREEN_WIDTH - CELL_SIZE * self.size)

    def shoot(self, projectiles, current_time):
        """Create a new projectile if cooldown has passed."""
        if current_time - self.last_shot_time >= SHOOTING_COOLDOWN:
            projectile_position = self.position + Vector2D(CELL_SIZE * self.size // 2 - CELL_SIZE * 0.25, -CELL_SIZE)
            projectiles.append(Projectile(projectile_position))
            self.last_shot_time = current_time

    def take_damage(self, damage):
        """Reduce player health by the amount of damage."""
        self.health -= damage
        print(f"Player took {damage} damage, health is now {self.health}")  # For debugging

    def is_alive(self):
        """Check if the player is still alive."""
        return self.health > 0

    def draw(self, screen):
        """Draw the player with health bar."""
        super().draw(screen)
        # Draw health bar
        health_bar_width = (self.health / PLAYER_INITIAL_HEALTH) * CELL_SIZE * self.size
        health_bar_rect = pygame.Rect(self.position[0], self.position[1] - 10, health_bar_width, 5)
        pygame.draw.rect(screen, (255, 0, 0), health_bar_rect)

class CollisionManager:
    """A manager for handling collisions between different game objects."""
    @staticmethod
    def check_collision(obj1, obj2):
        """Check if two objects are colliding."""
        rect1 = pygame.Rect(obj1.position[0], obj1.position[1], CELL_SIZE * obj1.size, CELL_SIZE * obj1.size)
        rect2 = pygame.Rect(obj2.position[0], obj2.position[1], CELL_SIZE * obj2.size, CELL_SIZE * obj2.size)
        return rect1.colliderect(rect2)

class GameState:
    RUNNING = 1
    GAME_OVER = 2

class Game:
    """The main game class that ties together game mechanics and interactions."""
    def __init__(self, physics_engine, collision_manager, player, agent, mcmc, training_mode=True, seconds_time=60):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.physics_engine = physics_engine
        self.collision_manager = collision_manager
        self.player = player
        self.projectiles = []
        self.falling_objects = []
        self.spawn_rate = INITIAL_SPAWN_RATE
        self.last_spawn_time = time.time()
        self.score = 0
        self.running = True
        self.state = GameState.RUNNING
        self.agent = agent
        self.mcmc = mcmc
        self.training_mode = training_mode
        self.seconds_time = seconds_time
        self.current_time = time.time()

    def spawn_falling_objects(self):
        current_time = time.time()
        if current_time - self.last_spawn_time >= self.spawn_rate:
            x_position = random.uniform(0, SCREEN_WIDTH - CELL_SIZE)
            position = [x_position, 0]
            color = random.choice(COLORS)
            size = random.uniform(1, 3)
            velocity = [0, random.uniform(1, self.physics_engine.max_velocity)]
            new_object = FallingObject(position, velocity, color, size)
            self.falling_objects.append(new_object)
            self.last_spawn_time = current_time
            self.spawn_rate *= SPAWN_ACCELERATION
            self.score += 1

    def get_current_state(self):
        return self.mcmc.current_state
    
    def handle_collisions(self):
        for falling_object in self.falling_objects[:]:
            if self.collision_manager.check_collision(self.player, falling_object):
                damage = self.physics_engine.calculate_damage(falling_object.size)
                self.player.take_damage(damage)
                self.falling_objects.remove(falling_object)
                if not self.player.is_alive():
                    self.show_game_over()
                    break

    def is_game_over(self):
        return self.state == GameState.GAME_OVER
    
    def show_game_over(self):
        self.state = GameState.GAME_OVER
        self.screen.fill(BLACK)
        game_over_text = self.font.render("Game Over", True, WHITE)
        score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        game_over_text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        score_text_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(game_over_text, game_over_text_rect)
        self.screen.blit(score_text, score_text_rect)
        pygame.display.flip()
        time.sleep(3)
        self.running = False

    def update_game_objects(self):
        for falling_object in self.falling_objects:
            falling_object.update(self.physics_engine)
            if falling_object.is_out_of_bounds():
                self.falling_objects.remove(falling_object)
        for projectile in self.projectiles:
            projectile.update(self.physics_engine)
            if projectile.is_out_of_bounds():
                self.projectiles.remove(projectile)

    def draw_game(self):
        if not self.training_mode:  # Only draw when not in training mode
            self.screen.fill(BLACK)
            for obj in self.falling_objects + self.projectiles:
                obj.draw(self.screen)
            self.player.draw(self.screen)
            score_text = self.font.render(f"Score: {self.score}", True, WHITE)
            self.screen.blit(score_text, (10, 10))
            pygame.display.flip()


    def get_current_state(self):
        player_position = np.array([self.player.position[0], self.player.position[1]])
        nearest_object_position = np.array([self.falling_objects[0].position if self.falling_objects else [0, 0]])
        nearest_object_position = nearest_object_position.flatten()  # Ensure it's a 1D array
        state = np.concatenate((player_position, nearest_object_position))
        return np.reshape(state, (1, -1))

    def apply_action(self, action):
        current_state = self.mcmc.current_state
        self.mcmc.transition(current_state, action)

        # Apply the action to the player
        if action == 0:  # Move left
            self.player.move(-1)
        elif action == 1:  # Move right
            self.player.move(1)
        elif action == 2:  # Shoot
            self.player.shoot(self.projectiles, self.current_time)

    def calculate_reward(self):
        reward = 0.1  # Small reward for staying alive each frame

        # Reward based on player's health
        reward += self.player.health * 0.01

        for obj in self.falling_objects:
            distance = np.linalg.norm(self.player.position - obj.position)
            if distance < 50:  # The player is close to a falling object
                reward -= 1
            else:  # The player is far from a falling object
                reward += 0.5

        # Reward based on the number of objects avoided or hit
        reward += self.player.objects_avoided * 0.5
        reward -= self.player.objects_hit * 1

        return reward

    def check_if_done(self):
        return not self.player.is_alive()

    def reset_game(self):
        """
        Reset the game to its initial state.
        
        This method should reset all game components to start a new game episode for the AI to play again.
        """
        # Reset player, falling objects, score, etc.
        self.player = Player(PLAYER_INITIAL_HEALTH)
        self.falling_objects.clear()
        self.projectiles.clear()
        self.score = 0
        self.last_spawn_time = time.time()
    
    def run(self, external_agent=None):
        total_reward = 0
        start_time = time.time()
        while self.running:
            self.current_time = time.time()
            if (time.time() - start_time) > self.seconds_time:  
                break
            self.spawn_falling_objects()
            self.update_game_objects()
            self.handle_collisions()

            # Use the internal agent if no external agent is provided
            agent_to_use = self.agent if external_agent is None else external_agent

            # Decision making
            state = self.mcmc.current_state
            action = agent_to_use.act(state, self.training_mode)

            # Apply the action and calculate the reward
            self.apply_action(action)
            reward = self.calculate_reward()
            total_reward += reward

            self.draw_game()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.clock.tick(1000)  # Increased frame rate for faster execution

        pygame.quit()
        return total_reward  # Return the total reward at the end

        pygame.quit()
        return total_reward  # Return the total reward at the end
    
    def get_state(self):
        # This is just an example. You will need to replace this with your actual implementation.
        state = {
            'player_position': self.player.position,
            'player_velocity': self.player.velocity,
            'player_health': self.player.health,
            'objects': [(obj.position, obj.velocity) for obj in self.falling_objects],
        }
        return state
    
        

class Agent:
    def __init__(self, state_size, action_size, model, mcmc):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.mcmc = mcmc  # New attribute for the MCMC
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state, training_mode):
        if not training_mode and np.random.rand() > self.epsilon:
            # Use the MCMC to predict the future states
            predicted_states = self.mcmc.predict(state)
            # Model is used here for decision-making when not training
            with torch.no_grad():  # Ensuring no gradient computation
                state = torch.FloatTensor(state).unsqueeze(0)
                predicted_states = torch.FloatTensor(predicted_states).unsqueeze(0)
                action_values = self.model(state, predicted_states)
                return torch.argmax(action_values, dim=1).item()
        else:
            # Random action taken during exploration or training
            return random.randrange(self.action_size)

def proposal_distribution(current_state):
    proposed_state = current_state.copy()
    proposed_state['player_position'] = [random.uniform(0, SCREEN_WIDTH), random.uniform(0, SCREEN_HEIGHT)]
    return proposed_state

def target_distribution(state):
    total_distance = 0
    # Check if state is a dictionary and contains 'objects' key
    if isinstance(state, dict) and 'objects' in state:
        for obj in state['objects']:
            distance = np.linalg.norm(np.array(state['player_position']) - np.array(obj[0]))
            total_distance += distance
    else:
        # print("Warning: Invalid state. Expected a dictionary with 'objects' key.")
        pass
    return total_distance / (SCREEN_WIDTH * SCREEN_HEIGHT) if total_distance else 0


def main_mcmc(training_mode, num_episodes):
    # Define game and DQN parameters
    state_size = 4
    action_size = 3
    batch_size = 64
    # Initialize DQN model and agent
    model = DQN(state_size, action_size)
    
    # Initialize Markov Chain
    mcmc = MCMC(proposal_distribution, target_distribution)

    agent = Agent(state_size, action_size, model, mcmc)

    # Initialize the optimizer and the learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    # Initialize game components
    physics = Physics()
    collision_manager = CollisionManager()
    player = Player(PLAYER_INITIAL_HEALTH)

    if not training_mode:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode
            print("Loaded trained model from:", model_path)
            agent.model = model  # Make sure the agent uses the loaded model
        else:
            print("No trained model found, please check the path or train the model first.")

    best_reward = -np.inf

    if training_mode:
        for i in range(0, num_episodes):
            # Call the game's run method to start the game loop
            print(f'Episode num {i+1}/{num_episodes}')
            physics = Physics()
            collision_manager = CollisionManager()
            player = Player(PLAYER_INITIAL_HEALTH)
            game = Game(physics, collision_manager, player, agent, mcmc, training_mode, 60)
            reward = game.run()

            # Save the best model
            if reward > best_reward:
                best_reward = reward
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved with reward: {best_reward} at {model_path}")

            # Step the learning rate scheduler
            scheduler.step()

    game_over = False
    collision_manager = CollisionManager()
    player = Player(PLAYER_INITIAL_HEALTH)
    game = Game(physics, collision_manager, player, agent, mcmc, training_mode, 60)

    if not training_mode:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode
            print("Loaded trained model from:", model_path)
            agent.model = model  # Make sure the agent uses the loaded model

            # Start the game loop
            while not game_over:
                # Get the current state of the game
                state = game.get_current_state()

                # Let the agent decide on an action
                action = agent.act(state, game.training_mode)

                # Update the game
                game.run()

                break
        else:
            print("No trained model found, please check the path or train the model first.")

# Initializing game components
if __name__ == "__main__":
    main_mcmc(True, 50)
    main_mcmc(False, 0)