import pygame
import numpy as np
import pandas as pd
import random

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Boid Parameters
NUM_BOIDS = 50
MAX_SPEED = 4
PERCEPTION_RADIUS = 50

# Data Storage
data_records = []

class Boid:
    def __init__(self):
        self.position = np.array([random.uniform(0, WIDTH), random.uniform(0, HEIGHT)])
        angle = random.uniform(0, 2 * np.pi)
        self.velocity = np.array([np.cos(angle), np.sin(angle)]) * MAX_SPEED
        self.acceleration = np.zeros(2)

    def edges(self):
        if self.position[0] > WIDTH:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = WIDTH
        if self.position[1] > HEIGHT:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = HEIGHT

    def align(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)
        for other in boids:
            if np.linalg.norm(other.position - self.position) < PERCEPTION_RADIUS:
                avg_vector += other.velocity
                total += 1
        if total > 0:
            avg_vector /= total
            avg_vector = avg_vector / np.linalg.norm(avg_vector) * MAX_SPEED
            steering = avg_vector - self.velocity
        return steering

    def cohesion(self, boids):
        steering = np.zeros(2)
        total = 0
        center_of_mass = np.zeros(2)
        for other in boids:
            if np.linalg.norm(other.position - self.position) < PERCEPTION_RADIUS:
                center_of_mass += other.position
                total += 1
        if total > 0:
            center_of_mass /= total
            vec_to_com = center_of_mass - self.position
            if np.linalg.norm(vec_to_com) > 0:
                vec_to_com = vec_to_com / np.linalg.norm(vec_to_com) * MAX_SPEED
            steering = vec_to_com - self.velocity
        return steering

    def separation(self, boids):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)
        for other in boids:
            distance = np.linalg.norm(other.position - self.position)
            if distance < PERCEPTION_RADIUS and distance > 0:
                diff = self.position - other.position
                diff /= distance  # Weight by distance
                avg_vector += diff
                total += 1
        if total > 0:
            avg_vector /= total
            if np.linalg.norm(avg_vector) > 0:
                avg_vector = avg_vector / np.linalg.norm(avg_vector) * MAX_SPEED
            steering = avg_vector - self.velocity
        return steering

    def flock(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohesion(boids)
        separation = self.separation(boids)

        # Weights for behaviors
        self.acceleration += alignment * 1.0
        self.acceleration += cohesion * 1.0
        self.acceleration += separation * 1.5

    def update(self):
        self.velocity += self.acceleration
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity = (self.velocity / speed) * MAX_SPEED
        self.position += self.velocity
        self.acceleration = np.zeros(2)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), self.position.astype(int), 3)

# Initialize Boids
boids = [Boid() for _ in range(NUM_BOIDS)]

# Main Loop
running = True
frame = 0
while running and frame < 1000:  # Limit to 1000 frames for data collection
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    for boid in boids:
        boid.flock(boids)
        boid.update()
        boid.edges()
        boid.draw(screen)
        
        # Record data
        data_records.append({
            'frame': frame,
            'boid_id': boids.index(boid),
            'x': boid.position[0],
            'y': boid.position[1],
            'vx': boid.velocity[0],
            'vy': boid.velocity[1]
        })

    pygame.display.flip()
    clock.tick(60)
    frame += 1

pygame.quit()

# Convert data to DataFrame
df = pd.DataFrame(data_records)

# Save to CSV
df.to_csv('boid_simulation_data.csv', index=False)

print("Simulation completed and data saved to 'boid_simulation_data.csv'")
