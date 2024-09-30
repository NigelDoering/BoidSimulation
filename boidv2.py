import tkinter as tk
from tkinter import ttk
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
        self.cluster_id = 0  # Initialize cluster ID

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
        # Fixed space: Boids bounce off the edges
        if self.position[0] > width:
            self.position[0] = width
            self.velocity[0] *= -1
        elif self.position[0] < 0:
            self.position[0] = 0
            self.velocity[0] *= -1

        if self.position[1] > height:
            self.position[1] = height
            self.velocity[1] *= -1
        elif self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] *= -1

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
        neighbors = [boid for boid in boids if boid.id != self.id and np.linalg.norm(self.position - boid.position) < radius]
        total = len(neighbors)
        if total > 0:
            diffs = self.position - np.array([boid.position for boid in neighbors])
            distances = np.linalg.norm(diffs, axis=1).reshape(-1, 1)
            diffs = diffs / distances  # Normalize
            steering = diffs.sum(axis=0) / total
            if np.linalg.norm(steering) > 0:
                steering = self.steer(steering, self.max_speed)
        return steering

    def alignment(self, boids, radius):
        steering = np.zeros(2, dtype='float64')
        neighbors = [boid for boid in boids if boid.id != self.id and np.linalg.norm(self.position - boid.position) < radius]
        total = len(neighbors)
        if total > 0:
            avg_velocity = np.mean([boid.velocity for boid in neighbors], axis=0)
            steering = self.steer(avg_velocity, self.max_speed)
        return steering

    def cohesion(self, boids, radius):
        steering = np.zeros(2, dtype='float64')
        neighbors = [boid for boid in boids if boid.id != self.id and np.linalg.norm(self.position - boid.position) < radius]
        total = len(neighbors)
        if total > 0:
            center_of_mass = np.mean([boid.position for boid in neighbors], axis=0)
            vec_to_com = center_of_mass - self.position
            steering = self.steer(vec_to_com, self.max_speed)
        return steering

    def steer(self, vector, target):
        if np.linalg.norm(vector) == 0:
            return vector
        desired = vector / np.linalg.norm(vector) * target
        steer = desired - self.velocity
        # Limit to max force
        if np.linalg.norm(steer) > self.max_force:
            steer = (steer / np.linalg.norm(steer)) * self.max_force
        return steer

# Simulation Class
class Simulation:
    def __init__(self, num_boids=50, width=600, height=600, grid_rows=3, grid_cols=3):
        self.width = width
        self.height = height
        self.boids = []
        self.num_boids = num_boids
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
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
        # Assign clusters based on fixed spatial ranges
        self.assign_clusters()
        # Record data
        self.record_data()

    def assign_clusters(self):
        # Define grid boundaries
        row_height = self.height / self.grid_rows
        col_width = self.width / self.grid_cols

        for boid in self.boids:
            row = int(boid.position[1] // row_height)
            col = int(boid.position[0] // col_width)
            # Ensure row and col are within grid bounds
            row = min(max(row, 0), self.grid_rows - 1)
            col = min(max(col, 0), self.grid_cols - 1)
            cluster_id = row * self.grid_cols + col  # Unique cluster ID based on grid position
            boid.cluster_id = cluster_id

    def record_data(self):
        frame = len(self.data_records)
        for boid in self.boids:
            self.data_records.append({
                'frame': frame,
                'boid_id': boid.id,
                'cluster_id': boid.cluster_id,
                'x': boid.position[0],
                'y': boid.position[1],
                'vx': boid.velocity[0],
                'vy': boid.velocity[1]
            })

    def export_to_csv(self, filename='boid_simulation_data.csv'):
        df = pd.DataFrame(self.data_records)
        df.to_csv(filename, index=False)
        print("Data exported to {}".format(filename))

# GUI Class with Side-by-Side Layout and Fixed Spatial Clusters
class BoidGUI:
    def __init__(self, root, simulation):
        self.root = root
        self.simulation = simulation
        self.running = False

        # Set up the main window with a fixed size
        window_width = simulation.width + 350  # Extra width for control panel
        window_height = simulation.height + 100   # Extra height for padding
        self.root.geometry("{}x{}".format(window_width, window_height))
        self.root.resizable(False, False)  # Prevent resizing to maintain layout

        # Create a main frame to hold control panel and canvas side by side
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=10)

        # Create Control Panel Frame on the Left
        control_panel = ttk.Frame(main_frame, width=300)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Configure grid for control panel
        control_panel.columnconfigure(1, weight=1)

        # Separation Radius Slider
        ttk.Label(control_panel, text="Separation Radius:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.separation_radius = tk.DoubleVar(value=25)
        separation_slider = ttk.Scale(control_panel, from_=5, to=50, variable=self.separation_radius, command=self.update_parameters)
        separation_slider.grid(row=0, column=1, sticky=tk.EW, pady=5)

        # Alignment Radius Slider
        ttk.Label(control_panel, text="Alignment Radius:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.alignment_radius = tk.DoubleVar(value=50)
        alignment_slider = ttk.Scale(control_panel, from_=10, to=100, variable=self.alignment_radius, command=self.update_parameters)
        alignment_slider.grid(row=1, column=1, sticky=tk.EW, pady=5)

        # Cohesion Radius Slider
        ttk.Label(control_panel, text="Cohesion Radius:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.cohesion_radius = tk.DoubleVar(value=50)
        cohesion_slider = ttk.Scale(control_panel, from_=10, to=100, variable=self.cohesion_radius, command=self.update_parameters)
        cohesion_slider.grid(row=2, column=1, sticky=tk.EW, pady=5)

        # Max Speed Slider
        ttk.Label(control_panel, text="Max Speed:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.max_speed = tk.DoubleVar(value=4)
        speed_slider = ttk.Scale(control_panel, from_=1, to=10, variable=self.max_speed, command=self.update_parameters)
        speed_slider.grid(row=3, column=1, sticky=tk.EW, pady=5)

        # Grid Rows Slider
        ttk.Label(control_panel, text="Grid Rows:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.grid_rows = tk.IntVar(value=3)
        grid_rows_slider = ttk.Scale(control_panel, from_=1, to=10, variable=self.grid_rows, command=self.update_parameters)
        grid_rows_slider.grid(row=4, column=1, sticky=tk.EW, pady=5)

        # Grid Columns Slider
        ttk.Label(control_panel, text="Grid Columns:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.grid_cols = tk.IntVar(value=3)
        grid_cols_slider = ttk.Scale(control_panel, from_=1, to=10, variable=self.grid_cols, command=self.update_parameters)
        grid_cols_slider.grid(row=5, column=1, sticky=tk.EW, pady=5)

        # Buttons Frame
        button_frame = ttk.Frame(control_panel)
        button_frame.grid(row=6, column=0, columnspan=2, pady=(20, 0))

        # Start (Play) Button
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Pause Button
        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        # Export to CSV Button
        self.export_button = ttk.Button(button_frame, text="Export CSV", command=self.export_data)
        self.export_button.pack(side=tk.LEFT, padx=5)

        # Create Canvas for visualization on the Right
        self.canvas = tk.Canvas(main_frame, width=simulation.width, height=simulation.height, bg='black')
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        # Initialize boid representations on the canvas with colors
        self.boid_reprs = {}
        total_clusters = simulation.grid_rows * simulation.grid_cols
        self.cluster_colors = self.generate_cluster_colors(total_clusters)
        for boid in self.simulation.boids:
            x, y = boid.position
            boid_id = boid.id
            # Assign initial color based on cluster_id
            color = self.cluster_colors[boid.cluster_id % len(self.cluster_colors)]
            oval = self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=color, outline='')
            self.boid_reprs[boid_id] = oval

    def generate_cluster_colors(self, num_clusters):
        # Generate distinct colors for clusters
        base_colors = [
            'red', 'blue', 'green', 'yellow', 'purple',
            'orange', 'cyan', 'magenta', 'lime', 'pink'
        ]
        if num_clusters <= len(base_colors):
            return base_colors[:num_clusters]
        else:
            # Generate additional distinct colors if needed
            extra_colors = ['#%06x' % np.random.randint(0, 0xFFFFFF) for _ in range(num_clusters - len(base_colors))]
            return base_colors + extra_colors

    def update_parameters(self, event=None):
        # Update simulation parameters based on slider values
        separation_radius = self.separation_radius.get()
        alignment_radius = self.alignment_radius.get()
        cohesion_radius = self.cohesion_radius.get()
        max_speed = self.max_speed.get()
        grid_rows = max(1, self.grid_rows.get())  # Ensure at least 1 row
        grid_cols = max(1, self.grid_cols.get())  # Ensure at least 1 column

        # Update grid settings in simulation
        self.simulation.grid_rows = grid_rows
        self.simulation.grid_cols = grid_cols

        # Regenerate cluster colors based on new grid size
        total_clusters = self.simulation.grid_rows * self.simulation.grid_cols
        self.cluster_colors = self.generate_cluster_colors(total_clusters)

        # Update all boids with new parameters
        for boid in self.simulation.boids:
            boid.max_speed = max_speed
            # Perception radii are handled in the flock method

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.run_simulation()  # Start the simulation loop

    def pause_simulation(self):
        if self.running:
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)

    def run_simulation(self):
        if self.running:
            start_time = time.time()

            # Update simulation
            self.simulation.update()
            self.update_canvas()

            # Calculate elapsed time and schedule next update
            elapsed_time = time.time() - start_time
            delay = max(0, int((1/30 - elapsed_time) * 1000))  # 30 FPS -> ~33ms
            self.root.after(delay, self.run_simulation)

    def update_canvas(self):
        # Update boid positions and colors based on cluster_id
        for boid in self.simulation.boids:
            x, y = boid.position
            boid_id = boid.id
            oval = self.boid_reprs[boid_id]
            # Update color based on cluster_id
            color = self.cluster_colors[boid.cluster_id % len(self.cluster_colors)]
            self.canvas.itemconfig(oval, fill=color)
            # Update position
            self.canvas.coords(oval, x-3, y-3, x+3, y+3)

    def export_data(self):
        try:
            self.simulation.export_to_csv()
        except Exception as e:
            print("Error exporting data:", e)

# Main Function
def main():
    root = tk.Tk()
    simulation = Simulation(num_boids=50, width=600, height=600, grid_rows=3, grid_cols=3)  # Adjusted canvas size and grid
    global control_gui  # To access from Simulation.update()
    control_gui = BoidGUI(root, simulation)
    root.mainloop()

if __name__ == "__main__":
    main()
