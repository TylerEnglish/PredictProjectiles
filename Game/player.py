import pygame
import random
import numpy as np
import time

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
INITIAL_SPAWN_RATE = 1
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
        self.last_shot_time = 0

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
    def __init__(self, physics_engine, collision_manager, player):
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

    def spawn_falling_objects(self):
        """Spawn falling objects at random positions and intervals."""
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

    def handle_collisions(self):
        """Check and handle collisions between objects and the player."""
        for falling_object in self.falling_objects[:]:
            if self.collision_manager.check_collision(self.player, falling_object):
                damage = self.physics_engine.calculate_damage(falling_object.size)
                self.player.take_damage(damage)
                self.falling_objects.remove(falling_object)
                if not self.player.is_alive():
                    self.show_game_over()
                    break  # Exit the loop if the player is dead

        for projectile in self.projectiles[:]:
            for falling_object in self.falling_objects[:]:
                if self.collision_manager.check_collision(projectile, falling_object):
                    self.projectiles.remove(projectile)
                    self.falling_objects.remove(falling_object)
                    break

    def show_game_over(self):
        """Display the game over screen and halt updates."""
        self.state = GameState.GAME_OVER
        self.screen.fill(BLACK)
        game_over_text = self.font.render("Game Over", True, WHITE)
        score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        game_over_text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        score_text_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(game_over_text, game_over_text_rect)
        self.screen.blit(score_text, score_text_rect)
        pygame.display.flip()
        time.sleep(3)  # Display the message for 3 seconds
        self.running = False  # Stop the game loop

    def update_game_objects(self):
        """Update all game objects positions and handle out of bounds objects."""
        if self.state == GameState.GAME_OVER:
            return  # Skip updates if the game is over

        # Update falling objects
        for falling_object in self.falling_objects[:]:
            falling_object.update(self.physics_engine)
            if falling_object.is_out_of_bounds():
                self.falling_objects.remove(falling_object)

        # Update projectiles
        for projectile in self.projectiles[:]:
            projectile.update(self.physics_engine)
            if projectile.is_out_of_bounds():
                self.projectiles.remove(projectile)

    def handle_input(self):
        """Handle user input for player movement and shooting."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.move(-1)
        elif keys[pygame.K_RIGHT]:
            self.player.move(1)
        if keys[pygame.K_SPACE]:
            self.player.shoot(self.projectiles, time.time())

    def draw_game(self):
        """Draw all game objects and update the display."""
        self.screen.fill(BLACK)
        for obj in self.falling_objects + self.projectiles:
            obj.draw(self.screen)
        self.player.draw(self.screen)
        # Draw the score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()



    def run(self):
        while self.running:
            if self.state == GameState.RUNNING:
                self.handle_input()
                self.spawn_falling_objects()
                self.update_game_objects()
                self.handle_collisions()
                self.draw_game()

                if not self.player.is_alive():
                    self.show_game_over()
                    break 
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.clock.tick(60)

        pygame.quit()

# Initializing the game components and starting the game loop
if __name__ == "__main__":
    physics = Physics()
    collision_manager = CollisionManager()
    player = Player(PLAYER_INITIAL_HEALTH)
    game = Game(physics, collision_manager, player)
    game.run()