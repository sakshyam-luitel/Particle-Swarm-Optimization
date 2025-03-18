import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import time
from io import BytesIO

pygame.init()

class Fish:
 def __init__(self, dimension, bounds):
    self.position = np.random.uniform(bounds[0], bounds[1], dimension)
    self.velocity = np.random.uniform(-1, 1, dimension)
    self.best_position = np.copy(self.position)
    self.best_value = float('inf')

def food_function(x):
    return np.sum((x - np.array([5, 5])) ** 2) 

def PSO_fish(num_fish, dimension, bounds, max_iterations, w=0.7, c1=0.01, c2=0.01):
    swarm = [Fish(dimension, bounds) for _ in range(num_fish)]
    global_best_position = np.array([5, 5])  
    global_best_value = float('inf')
    
    history = []
    positions = []
    
    for iteration in range(max_iterations):
        current_positions = []
        all_reached = True
        for fish in swarm:
            fish.velocity = (w * fish.velocity +
                             c1 * np.random.rand() * (fish.best_position - fish.position) +
                             c2 * np.random.rand() * (global_best_position - fish.position))
            fish.position += fish.velocity
            
            value = food_function(fish.position)
            if value < fish.best_value:
                fish.best_value = value
                fish.best_position = fish.position
            
            if value < global_best_value:
                global_best_value = value
                
            current_positions.append(fish.position)
            
            if np.linalg.norm(fish.position - global_best_position) > 0.1:
                all_reached = False
        
        positions.append(np.array(current_positions))
        history.append(global_best_value)
        print(f"Iteration {iteration+1}: Best Value = {global_best_value}")
        
        if all_reached:
            print("All fish have reached the food!")
            break
    
    return global_best_position, global_best_value, history, positions



def visualize_combined(positions, history, screen_size=600, bounds=(-10, 10)):
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Fish Movement & Optimization Progress")
    clock = pygame.time.Clock()
    running = True
    
    font = pygame.font.Font(None, 24)
    
    for i, pos in enumerate(positions):
        screen.fill((255, 255, 255)) 
        
        # Draw fish
        for fish in pos:
            x = int((fish[0] - bounds[0]) / (bounds[1] - bounds[0]) * screen_size)
            y = int((fish[1] - bounds[0]) / (bounds[1] - bounds[0]) * screen_size)
            pygame.draw.circle(screen, (0, 0, 255), (x, y), 7)  # Blue fish
        
        # Draw food
        food_x = int((5 - bounds[0]) / (bounds[1] - bounds[0]) * screen_size)
        food_y = int((5 - bounds[0]) / (bounds[1] - bounds[0]) * screen_size)
        pygame.draw.circle(screen, (255, 0, 0), (food_x, food_y), 10)  # Red food
        
        # Render optimization progress plot
        fig, ax = plt.subplots()
        ax.plot(history[:i+1], marker='o', linestyle='-', color='b')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Value Found')
        ax.set_title('Optimization Progress')
        plt.grid()
        
        # Convert Matplotlib figure to Pygame surface
        buf = BytesIO()
        plt.savefig(buf, format="PNG")
        buf.seek(0)
        plot_surface = pygame.image.load(buf)
        buf.close()
        plot_surface = pygame.transform.scale(plot_surface, (300, 200))
        screen.blit(plot_surface, (screen_size - 310, 10))
        plt.close(fig)
        
        # Display iteration info
        text = font.render(f"Iteration: {i+1}/{len(positions)}", True, (0, 0, 0))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(5)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
    
    pygame.quit()

if __name__ == "__main__":
    num_fish = 30
    dimension = 2
    bounds = (-10, 10)
    max_iterations = 1000
    
    best_position, best_value, history, positions = PSO_fish(num_fish, dimension, bounds, max_iterations)
    
    print(f"Best Position: {best_position}")
    print(f"Best Value: {best_value}")
    
    visualize_combined(positions, history)

