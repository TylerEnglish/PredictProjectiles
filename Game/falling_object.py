# imports
import pygame
import random

# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
# Define grid dimensions (number of cells in grid)
GRID_SIZE = 100
# Calculate cell size based on screen width and grid size
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

# Define color constants
BLACK = (0, 0, 0)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0)]

# Physics constants
GRAVITY = 0.5  # Acceleration due to gravity, affects how fast objects fall
WIND = 0.1  # Wind resistance, affects lateral movement of objects
MAX_VEL = 5  # Maximum falling speed of objects

class GameObject:
    def __init__(self, object_type, position, velocity, color, hp, damage, speed, size):
        """
        Initialize a new game object with properties.
        :param object_type: Type of the object (e.g., 'type1', 'type2'). Can be used for different behaviors.
        :param position: Tuple representing the object's position on the grid (y, x).
        :param velocity: Tuple representing the object's velocity (vy, vx).
        :param color: Color of the object.
        :param hp: Health points of the object.
        :param damage: Damage the object can inflict.
        :param speed: Speed of the object, affecting its falling speed.
        :param size: Size of the object.
        """
        self.object_type = object_type
        self.position = position
        self.velocity = velocity
        self.color = color
        self.hp = hp
        self.damage = damage
        self.speed = speed
        self.size = size  # New attribute for the object's size

    def update(self):
        """
        Update the object's position and velocity based on its speed, gravity, and wind.
        """
        # Apply speed to velocity
        self.velocity = (self.velocity[0] + self.speed + GRAVITY, self.velocity[1] + (WIND if random.random() < 0.5 else -WIND))
        # Update position based on velocity
        self.position = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])
        # Limit the velocity to MAX_VEL
        if self.velocity[0] > MAX_VEL:
            self.velocity = (MAX_VEL, self.velocity[1])

    def is_out_of_bounds(self, grid_size):
        """
        Check if the object is out of the grid boundaries.
        :param grid_size: Size of the grid.
        :return: Boolean indicating whether the object is out of bounds.
        """
        return self.position[0] >= grid_size or self.position[1] < 0 or self.position[1] >= grid_size

class FallingObjectsGame:
    def __init__(self):
        """
        Initialize the game, setting up the screen, clock, and other game properties.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.objects = []  # List to hold all game objects
        self.spawn_rate = 0.1  # Initial spawn rate for objects
        self.game_steps = 0  # Counter to keep track of game steps

    def draw_objects(self):
        """
        Draw all objects on the screen based on their properties.
        """
        for obj in self.objects:
            x = int(obj.position[1] * CELL_SIZE)
            y = int(obj.position[0] * CELL_SIZE)
            # Adjust drawing to use object's size
            pygame.draw.rect(self.screen, obj.color, (x, y, obj.size * CELL_SIZE, obj.size * CELL_SIZE))

    def add_new_objects(self):
        """
        Spawn new objects based on the current spawn rate.
        """
        if random.random() < self.spawn_rate:
            i = random.randint(0, GRID_SIZE - 1)
            object_type = random.choice(['type1', 'type2', 'type3'])
            color = random.choice(COLORS)
            initial_velocity = (0, random.uniform(-1, 1))
            hp = random.randint(1, 10)
            damage = random.randint(1, 5)
            speed = random.uniform(0.1, 0.3)
            size = random.randint(1, 3)  # Randomize object size
            new_object = GameObject(object_type, (0, i), initial_velocity, color, hp, damage, speed, size)
            self.objects.append(new_object)

    def update_objects(self):
        """
        Update all objects' positions, apply physics, and remove any that are out of bounds.
        """
        for obj in self.objects:
            obj.update()
        self.objects = [obj for obj in self.objects if not obj.is_out_of_bounds(GRID_SIZE)]

    def run_step(self):
        """
        Run one step of the game: update game state, draw the screen, and handle game events.
        """
        self.screen.fill(BLACK)  # Fill background with black
        self.add_new_objects()
        self.update_objects()
        self.draw_objects()
        pygame.display.flip()  # Update the full display Surface to the screen
        self.clock.tick(10)  # Control the game's framerate

        # Increase spawn rate gradually
        self.game_steps += 1
        if self.game_steps % 100 == 0:
            self.spawn_rate += 0.01
            if self.spawn_rate > 0.5:
                self.spawn_rate = 0.5

    def close(self):
        """
        Clean up and close the game.
        """
        pygame.quit()

if __name__ == "__main__":
    game = FallingObjectsGame()
    running = True
    while running:
        game.run_step()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    game.close()
