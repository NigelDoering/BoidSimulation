import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
import pandas as pd

# Boid Class
class Boid:
    def __init__(self, boid_id, position, velocity, max_speed=4, max_force=0.1):
        self.id = boid_id
        self.position = np.array(position, dtype='float64')  # 2D position
        self.velocity = np.array(velocity, dtype='float64')  # 2D velocity
        self.acceleration = np.zeros(2, dtype='float64')    # 2D acceleration
        self.max_speed = max_speed
        self.max_force = max_force

    def apply_force(self, force):
        self.acceleration += force

    def update(self):
        self.velocity += self.acceleration
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        self.position += self.velocity
        # Reset acceleration
        self.acceleration = np.zeros(2, dtype='float64')

    def edges(self, width, height):
        # Wrap around the edges
        if self.position[0] > width:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = width

        if self.position[1] > height:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = height

    def flock(self, boids, separation_radius=25, alignment_radius=50, cohesion_radius=50):
        separation = self.separation(boids, separation_radius)
        alignment = self.alignment(boids, alignment_radius)
        cohesion = self.cohesion(boids, cohesion_radius)

        # Weights for behaviors
        separation_weight = 1.5
        alignment_weight = 1.0
        cohesion_weight = 1.0

        self.apply_force(separation * separation_weight)
        self.apply_force(alignment * alignment_weight)
        self.apply_force(cohesion * cohesion_weight)

    def separation(self, boids, radius):
        steering = np.zeros(2, dtype='float64')
        total = 0
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if boid.id != self.id and distance < radius:
                diff = self.position - boid.position
                if distance > 0:
                    diff /= distance  # Weight by distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            # Implement Reynolds' Steering Formula
            if np.linalg.norm(steering) > 0:
                steering = self.steer(steering, self.max_speed)
        return steering

    def alignment(self, boids, radius):
        steering = np.zeros(2, dtype='float64')
        total = 0
        avg_velocity = np.zeros(2, dtype='float64')
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if boid.id != self.id and distance < radius:
                avg_velocity += boid.velocity
                total += 1
        if total > 0:
            avg_velocity /= total
            steering = self.steer(avg_velocity, self.max_speed)
        return steering

    def cohesion(self, boids, radius):
        steering = np.zeros(2, dtype='float64')
        total = 0
        center_of_mass = np.zeros(2, dtype='float64')
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if boid.id != self.id and distance < radius:
                center_of_mass += boid.position
                total += 1
        if total > 0:
            center_of_mass /= total
            vec_to_com = center_of_mass - self.position
            steering = self.steer(vec_to_com, self.max_speed)
        return steering

    def steer(self, vector, target):
        desired = vector / np.linalg.norm(vector) * target
        steer = desired - self.velocity
        # Limit to max force
        if np.linalg.norm(steer) > self.max_force:
            steer = (steer / np.linalg.norm(steer)) * self.max_force
        return steer

# Simulation Class
class Simulation:
    def __init__(self, num_boids=50, width=800, height=600):
        self.width = width
        self.height = height
        self.boids = []
        self.num_boids = num_boids
        self.initialize_boids()
        self.data_records = []  # To store simulation data

    def initialize_boids(self):
        for i in range(self.num_boids):
            position = [np.random.uniform(0, self.width), np.random.uniform(0, self.height)]
            velocity = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
            boid = Boid(boid_id=i, position=position, velocity=velocity)
            self.boids.append(boid)

    def update(self):
        # Update flocking behavior
        for boid in self.boids:
            boid.flock(self.boids,
                       separation_radius=control_gui.separation_radius.get(),
                       alignment_radius=control_gui.alignment_radius.get(),
                       cohesion_radius=control_gui.cohesion_radius.get())
        # Update boids' positions
        for boid in self.boids:
            boid.update()
            boid.edges(self.width, self.height)
        # Record data
        self.record_data()

    def record_data(self):
        frame = len(self.data_records)
        for boid in self.boids:
            self.data_records.append({
                'frame': frame,
                'boid_id': boid.id,
                'x': boid.position[0],
                'y': boid.position[1],
                'vx': boid.velocity[0],
                'vy': boid.velocity[1]
            })

    def export_to_csv(self, filename='boid_simulation_data.csv'):
        df = pd.DataFrame(self.data_records)
        df.to_csv(filename, index=False)
        print("Data exported to {}".format(filename))

# GUI Class
class BoidGUI:
    def __init__(self, root, simulation):
        self.root = root
        self.simulation = simulation
        self.running = False

        # Set up the main window
        self.root.title("Boid Simulation with GUI")
        self.root.geometry("{}x{}".format(simulation.width, simulation.height + 150))  # Extra space for controls

        # Create Canvas for visualization
        self.canvas = tk.Canvas(self.root, width=simulation.width, height=simulation.height, bg='black')
        self.canvas.pack()

        # Create frame for controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Separation Radius Slider
        ttk.Label(control_frame, text="Separation Radius:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.separation_radius = tk.DoubleVar(value=25)
        separation_slider = ttk.Scale(control_frame, from_=5, to=50, variable=self.separation_radius, command=self.update_parameters)
        separation_slider.grid(row=0, column=1, sticky=tk.EW, pady=5)

        # Alignment Radius Slider
        ttk.Label(control_frame, text="Alignment Radius:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.alignment_radius = tk.DoubleVar(value=50)
        alignment_slider = ttk.Scale(control_frame, from_=10, to=100, variable=self.alignment_radius, command=self.update_parameters)
        alignment_slider.grid(row=1, column=1, sticky=tk.EW, pady=5)

        # Cohesion Radius Slider
        ttk.Label(control_frame, text="Cohesion Radius:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.cohesion_radius = tk.DoubleVar(value=50)
        cohesion_slider = ttk.Scale(control_frame, from_=10, to=100, variable=self.cohesion_radius, command=self.update_parameters)
        cohesion_slider.grid(row=2, column=1, sticky=tk.EW, pady=5)

        # Max Speed Slider
        ttk.Label(control_frame, text="Max Speed:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.max_speed = tk.DoubleVar(value=4)
        speed_slider = ttk.Scale(control_frame, from_=1, to=10, variable=self.max_speed, command=self.update_parameters)
        speed_slider.grid(row=3, column=1, sticky=tk.EW, pady=5)

        # Configure grid weights
        control_frame.columnconfigure(1, weight=1)

        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.export_button = ttk.Button(button_frame, text="Export to CSV", command=self.export_data)
        self.export_button.pack(side=tk.LEFT, padx=5)

        # Initialize boid representations on the canvas
        self.boid_reprs = {}
        for boid in self.simulation.boids:
            x, y = boid.position
            boid_id = boid.id
            # Draw as a small circle (oval)
            oval = self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='white', outline='')
            self.boid_reprs[boid_id] = oval

    def update_parameters(self, event=None):
        # Update simulation parameters based on slider values
        separation_radius = self.separation_radius.get()
        alignment_radius = self.alignment_radius.get()
        cohesion_radius = self.cohesion_radius.get()
        max_speed = self.max_speed.get()

        # Update all boids with new parameters
        for boid in self.simulation.boids:
            boid.max_speed = max_speed
            # If you want to dynamically change perception radii, you can store them in the boid
            # Here, we'll pass them directly in the flock method

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            threading.Thread(target=self.run_simulation, daemon=True).start()

    def pause_simulation(self):
        if self.running:
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)

    def run_simulation(self):
        while self.running:
            # Update simulation parameters
            separation_radius = self.separation_radius.get()
            alignment_radius = self.alignment_radius.get()
            cohesion_radius = self.cohesion_radius.get()
            max_speed = self.max_speed.get()

            # Update boid parameters
            for boid in self.simulation.boids:
                boid.max_speed = max_speed

            # Update simulation
            self.simulation.update()
            self.update_canvas()
            time.sleep(0.05)  # Control simulation speed (20 FPS)

    def update_canvas(self):
        for boid in self.simulation.boids:
            x, y = boid.position
            boid_id = boid.id
            oval = self.boid_reprs[boid_id]
            self.canvas.coords(oval, x-3, y-3, x+3, y+3)

    def export_data(self):
        self.simulation.export_to_csv()

# Main Function
def main():
    simulation = Simulation(num_boids=100, width=800, height=600)
    root = tk.Tk()
    global control_gui  # To access from Simulation.update()
    control_gui = BoidGUI(root, simulation)
    root.mainloop()

if __name__ == "__main__":
    main()
