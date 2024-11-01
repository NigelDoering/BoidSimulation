import os
import tkinter as tk
from tkinter import ttk, colorchooser, messagebox, simpledialog
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ------------------------------
# Boid Class
# ------------------------------
class Boid:
    def __init__(self, boid_id, position, velocity, flock):
        self.id = boid_id
        self.position = np.array(position, dtype='float64')  # 2D position
        self.velocity = np.array(velocity, dtype='float64')  # 2D velocity
        self.acceleration = np.zeros(2, dtype='float64')    # 2D acceleration
        self.flock = flock                                  # Instance of Flock

    def apply_force(self, force):
        self.acceleration += force

    def update(self):
        self.velocity += self.acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > self.flock.max_speed:
            self.velocity = (self.velocity / speed) * self.flock.max_speed
        self.position += self.velocity
        self.acceleration = np.zeros(2, dtype='float64')

    def edges(self, width, height):
        # Bounce off the edges
        if self.position[0] >= width:
            self.position[0] = width
            self.velocity[0] *= -1
        elif self.position[0] <= 0:
            self.position[0] = 0
            self.velocity[0] *= -1

        if self.position[1] >= height:
            self.position[1] = height
            self.velocity[1] *= -1
        elif self.position[1] <= 0:
            self.position[1] = 0
            self.velocity[1] *= -1

    def flocking(self, boids, separation_radius, alignment_radius, cohesion_radius, obstacles):
        separation = self.separation(boids, separation_radius)
        alignment = self.alignment(boids, alignment_radius)
        cohesion = self.cohesion(boids, cohesion_radius)
        avoid = self.avoid_obstacles(obstacles)

        # Weights for behaviors
        separation_weight = 1.5
        alignment_weight = 1.0
        cohesion_weight = 1.0
        avoid_weight = 3.0  # Higher weight for obstacle avoidance

        self.apply_force(separation * separation_weight)
        self.apply_force(alignment * alignment_weight)
        self.apply_force(cohesion * cohesion_weight)
        self.apply_force(avoid * avoid_weight)

    def separation(self, boids, radius):
        steering = np.zeros(2, dtype='float64')
        total = 0
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if boid != self and distance < radius:
                diff = self.position - boid.position
                if distance > 0:
                    diff /= distance  # Weight by distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            if np.linalg.norm(steering) > 0:
                steering = self.steer(steering, self.flock.max_speed)
        return steering

    def alignment(self, boids, radius):
        steering = np.zeros(2, dtype='float64')
        total = 0
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if boid != self and distance < radius:
                steering += boid.velocity
                total += 1
        if total > 0:
            steering /= total
            if np.linalg.norm(steering) > 0:
                steering = self.steer(steering, self.flock.max_speed)
        return steering

    def cohesion(self, boids, radius):
        steering = np.zeros(2, dtype='float64')
        total = 0
        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if boid != self and distance < radius:
                steering += boid.position
                total += 1
        if total > 0:
            steering /= total
            steering -= self.position
            if np.linalg.norm(steering) > 0:
                steering = self.steer(steering, self.flock.max_speed)
        return steering

    def avoid_obstacles(self, obstacles):
        steering = np.zeros(2, dtype='float64')
        for obstacle in obstacles:
            distance = np.linalg.norm(self.position - obstacle.position)
            buffer_distance = obstacle.radius + self.flock.size + 20  # Buffer distance
            if distance < buffer_distance:
                diff = self.position - obstacle.position
                if distance > 0:
                    diff /= distance
                steering += diff
        if np.linalg.norm(steering) > 0:
            steering = self.steer(steering, self.flock.max_speed)
        return steering

    def steer(self, vector, target):
        desired = vector / np.linalg.norm(vector) * target
        steer = desired - self.velocity
        if np.linalg.norm(steer) > self.flock.max_force:
            steer = (steer / np.linalg.norm(steer)) * self.flock.max_force
        return steer

# ------------------------------
# Flock Class
# ------------------------------
class Flock:
    def __init__(self, flock_id, color, max_speed=4, max_force=0.05, size=3):
        self.flock_id = flock_id
        self.color = color
        self.max_speed = max_speed
        self.max_force = max_force
        self.size = size
        self.boids = []

    def add_boid(self, boid):
        self.boids.append(boid)

# ------------------------------
# Obstacle Class
# ------------------------------
class Obstacle:
    def __init__(self, position, radius, color='brown'):
        self.position = np.array(position, dtype='float64')  # 2D position
        self.radius = radius
        self.color = color

# ------------------------------
# Simulation Class
# ------------------------------
class Simulation:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.flocks = []
        self.boids = []
        self.obstacles = []  # List to hold obstacles
        self.next_flock_id = 1
        self.data_records = []  # List to hold snapshot data

    def add_flock(self, color, num_boids=30, max_speed=4, max_force=0.05, size=3):
        flock = Flock(flock_id=self.next_flock_id, color=color, max_speed=max_speed, max_force=max_force, size=size)
        self.flocks.append(flock)
        for _ in range(num_boids):
            boid_id = len(self.boids)
            position = [np.random.uniform(0, self.width), np.random.uniform(0, self.height)]
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = [np.cos(angle), np.sin(angle)]
            velocity = np.array(velocity) * np.random.uniform(1, max_speed)
            boid = Boid(boid_id=boid_id, position=position, velocity=velocity, flock=flock)
            flock.add_boid(boid)
            self.boids.append(boid)
        self.next_flock_id += 1

    def add_obstacle(self, position, radius, color='brown'):
        obstacle = Obstacle(position, radius, color)
        self.obstacles.append(obstacle)

    def update(self, separation_radius, alignment_radius, cohesion_radius):
        for flock in self.flocks:
            for boid in flock.boids:
                boid.flocking(flock.boids, separation_radius, alignment_radius, cohesion_radius, self.obstacles)
        for boid in self.boids:
            boid.update()
            boid.edges(self.width, self.height)

    def record_data(self, frame_number):
        # Record the state of all boids at the current frame
        for boid in self.boids:
            record = {
                'frame': frame_number,
                'boid_id': boid.id,
                'flock_id': boid.flock.flock_id,
                'x': boid.position[0],
                'y': boid.position[1],
                'vx': boid.velocity[0],
                'vy': boid.velocity[1]
            }
            self.data_records.append(record)

    def export_to_csv(self):
        # Define the data directory
        data_dir = 'data'
        # Create the data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        # Construct the full file path
        filename = 'boid_simulation_data.csv'
        file_path = os.path.join(data_dir, filename)
        # Convert data_records to DataFrame and export to CSV
        df = pd.DataFrame(self.data_records)
        df.to_csv(file_path, index=False)
        print("Data exported to {}".format(file_path))

# ------------------------------
# GUI Class
# ------------------------------
class BoidGUI:
    def __init__(self, root, simulation):
        self.root = root
        self.simulation = simulation
        self.running = False
        self.start_time = None  # To track when the simulation starts
        self.frame_number = 0   # To track the current frame for data recording

        # Set up the main window with a fixed size
        window_width = simulation.width + 400  # Extra width for control panel
        window_height = simulation.height + 50   # Adjusted for optimal layout
        self.root.geometry("{}x{}".format(window_width, window_height))
        self.root.resizable(False, False)  # Prevent resizing to maintain layout

        # Create a main frame to hold simulation and control panel
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=10)

        # Create Canvas for visualization on the Left
        self.canvas = tk.Canvas(main_frame, width=simulation.width, height=simulation.height, bg='black')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Create Control Panel Frame on the Right
        control_panel = ttk.Frame(main_frame, width=350)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Configure grid for control panel
        for i in range(20):
            control_panel.rowconfigure(i, weight=1)
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

        # Separator
        separator = ttk.Separator(control_panel, orient='horizontal')
        separator.grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)

        # Add Flock Controls
        ttk.Label(control_panel, text="Add New Flock").grid(row=4, column=0, columnspan=2, pady=5)

        # Flock Color
        ttk.Label(control_panel, text="Flock Color:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.new_flock_color = tk.StringVar(value="blue")
        color_button = ttk.Button(control_panel, text="Choose Color", command=lambda: self.choose_color(color_button))
        color_button.grid(row=5, column=1, sticky=tk.EW, pady=5)

        # Number of Boids
        ttk.Label(control_panel, text="Number of Boids:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.new_flock_num_boids = tk.StringVar(value="30")
        num_boids_entry = ttk.Entry(control_panel, textvariable=self.new_flock_num_boids)
        num_boids_entry.grid(row=6, column=1, sticky=tk.EW, pady=5)

        # Max Speed
        ttk.Label(control_panel, text="Max Speed:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.new_flock_max_speed = tk.StringVar(value="4.0")
        max_speed_entry = ttk.Entry(control_panel, textvariable=self.new_flock_max_speed)
        max_speed_entry.grid(row=7, column=1, sticky=tk.EW, pady=5)

        # Max Force
        ttk.Label(control_panel, text="Max Force:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.new_flock_max_force = tk.StringVar(value="0.05")
        max_force_entry = ttk.Entry(control_panel, textvariable=self.new_flock_max_force)
        max_force_entry.grid(row=8, column=1, sticky=tk.EW, pady=5)

        # Boid Size
        ttk.Label(control_panel, text="Boid Size:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.new_flock_size = tk.StringVar(value="3")
        size_entry = ttk.Entry(control_panel, textvariable=self.new_flock_size)
        size_entry.grid(row=9, column=1, sticky=tk.EW, pady=5)

        # Add Flock Button
        add_flock_button = ttk.Button(control_panel, text="Add Flock", command=self.add_flock)
        add_flock_button.grid(row=10, column=0, columnspan=2, pady=15, sticky=tk.EW)

        # Separator
        separator2 = ttk.Separator(control_panel, orient='horizontal')
        separator2.grid(row=11, column=0, columnspan=2, sticky='ew', pady=10)

        # Add Obstacle Controls
        ttk.Label(control_panel, text="Add Obstacles").grid(row=12, column=0, columnspan=2, pady=5)

        # Add Single Obstacle Button
        add_obstacle_button = ttk.Button(control_panel, text="Add Obstacle", command=self.add_obstacle_dialog)
        add_obstacle_button.grid(row=13, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # Add Multiple Obstacles Button
        add_multiple_obstacles_button = ttk.Button(control_panel, text="Add Multiple Obstacles", command=self.add_multiple_obstacles_dialog)
        add_multiple_obstacles_button.grid(row=14, column=0, columnspan=2, pady=5, sticky=tk.EW)

        # Separator
        separator3 = ttk.Separator(control_panel, orient='horizontal')
        separator3.grid(row=15, column=0, columnspan=2, sticky='ew', pady=10)

        # Simulation Control Buttons
        button_frame = ttk.Frame(control_panel)
        button_frame.grid(row=16, column=0, columnspan=2, pady=10)

        # Start (Play) Button
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Pause Button
        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        # Export to CSV Button
        self.export_button = ttk.Button(button_frame, text="Export CSV", command=self.export_data, state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=5)

        # Reset Button
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_simulation, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Add Timer Label
        self.timer_label = ttk.Label(control_panel, text="Elapsed Time: 00:00:00")
        self.timer_label.grid(row=17, column=0, columnspan=2, pady=10)

        # Status Label
        self.status_label = ttk.Label(control_panel, text="Status: Ready")
        self.status_label.grid(row=18, column=0, columnspan=2, pady=5)

        # Initialize boid representations on the canvas with colors
        self.boid_reprs = {}
        for boid in self.simulation.boids:
            x, y = boid.position
            boid_id = boid.id
            # Assign initial color based on flock
            color = self.get_boid_color(boid)
            oval = self.canvas.create_oval(
                x - boid.flock.size, y - boid.flock.size,
                x + boid.flock.size, y + boid.flock.size,
                fill=color, outline=''
            )
            self.boid_reprs[boid_id] = oval

    def get_boid_color(self, boid):
        # Return boid's flock color
        return boid.flock.color

    def update_parameters(self, event=None):
        # Update simulation parameters based on slider values
        separation_radius = self.separation_radius.get()
        alignment_radius = self.alignment_radius.get()
        cohesion_radius = self.cohesion_radius.get()

    def choose_color(self, button):
        # Open color chooser and set the chosen color as the button's text
        color_code = colorchooser.askcolor(title="Choose Flock Color")
        if color_code[1]:
            button.config(text=color_code[1])
            self.new_flock_color.set(color_code[1])

    def add_flock(self):
        # Retrieve flock parameters with validation
        color = self.new_flock_color.get()
        if not color or color == "Choose Color":
            color = "blue"  # Default color

        try:
            num_boids = int(self.new_flock_num_boids.get())
            max_speed = float(self.new_flock_max_speed.get())
            max_force = float(self.new_flock_max_force.get())
            size = int(self.new_flock_size.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for boid parameters.")
            return

        if num_boids <= 0 or max_speed <= 0 or max_force <= 0 or size <= 0:
            messagebox.showerror("Input Error", "Please enter positive values for boid parameters.")
            return

        # Add flock to simulation
        self.simulation.add_flock(color=color, num_boids=num_boids, max_speed=max_speed, max_force=max_force, size=size)

        # Create visual representations for new boids
        for boid in self.simulation.flocks[-1].boids[-num_boids:]:
            x, y = boid.position
            oval = self.canvas.create_oval(
                x - boid.flock.size, y - boid.flock.size,
                x + boid.flock.size, y + boid.flock.size,
                fill=boid.flock.color, outline=''
            )
            self.boid_reprs[boid.id] = oval

    def initialize_obstacles(self):
        # Removed default obstacles as per user request
        pass  # No default obstacles are added

    def add_obstacle_dialog(self):
        # Prompt user for obstacle parameters and add it
        try:
            prompt_x = "Enter X position (0 to {}):".format(self.simulation.width)
            x = simpledialog.askinteger("Input", prompt_x,
                                        parent=self.root, minvalue=0, maxvalue=self.simulation.width)
            if x is None:
                return
            prompt_y = "Enter Y position (0 to {}):".format(self.simulation.height)
            y = simpledialog.askinteger("Input", prompt_y,
                                        parent=self.root, minvalue=0, maxvalue=self.simulation.height)
            if y is None:
                return
            radius = simpledialog.askinteger("Input", "Enter obstacle radius (e.g., 30):",
                                            parent=self.root, minvalue=10, maxvalue=100)
            if radius is None:
                return
            color_code = colorchooser.askcolor(title="Choose Obstacle Color")
            if color_code[1]:
                color = color_code[1]
            else:
                color = 'brown'  # Default color

            # Add obstacle to simulation
            self.simulation.add_obstacle(position=(x, y), radius=radius, color=color)

            # Draw obstacle on canvas
            circle = self.canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                fill=color, outline='grey', width=2
            )
        except Exception as e:
            messagebox.showerror("Error", "An error occurred while adding obstacle:\n{}".format(e))

    def add_multiple_obstacles_dialog(self):
        # Prompt user for number of obstacles and add them with random positions
        try:
            num = simpledialog.askinteger("Input", "Enter number of obstacles to add:",
                                         parent=self.root, minvalue=1, maxvalue=100)
            if num is None:
                return
            for _ in range(num):
                # Random position avoiding overlaps (simple approach)
                attempts = 0
                max_attempts = 100
                while attempts < max_attempts:
                    x = np.random.randint(50, self.simulation.width - 50)
                    y = np.random.randint(50, self.simulation.height - 50)
                    radius = np.random.randint(20, 60)
                    color = 'brown'  # Default color or random colors
                    # Simple overlap check
                    overlap = False
                    for obstacle in self.simulation.obstacles:
                        distance = np.linalg.norm(np.array([x, y]) - obstacle.position)
                        if distance < (radius + obstacle.radius + 20):
                            overlap = True
                            break
                    if not overlap:
                        break
                    attempts += 1
                if attempts >= max_attempts:
                    messagebox.showwarning("Warning", "Could not place all obstacles without overlap.")
                    break
                self.simulation.add_obstacle(position=(x, y), radius=radius, color=color)
                # Draw obstacle on canvas
                circle = self.canvas.create_oval(
                    x - radius, y - radius,
                    x + radius, y + radius,
                    fill=color, outline='grey', width=2
                )
        except Exception as e:
            messagebox.showerror("Error", "An error occurred while adding multiple obstacles:\n{}".format(e))

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Running")
            # Record the start time
            if not self.start_time:
                self.start_time = time.time()
            self.run_simulation()
            self.update_timer()

    def pause_simulation(self):
        if self.running:
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Paused")

    def run_simulation(self):
        if self.running:
            start_time_loop = time.time()

            # Update simulation with current parameters
            separation_radius = self.separation_radius.get()
            alignment_radius = self.alignment_radius.get()
            cohesion_radius = self.cohesion_radius.get()
            self.simulation.update(separation_radius, alignment_radius, cohesion_radius)

            # Update canvas with new boid positions
            self.update_canvas()

            # Record data for export
            self.frame_number += 1
            self.simulation.record_data(self.frame_number)

            # Calculate elapsed time and schedule next update
            elapsed_time = time.time() - start_time_loop
            target_delay = 1.0 / 60.0  # Aim for ~60 FPS
            delay = max(1, int((target_delay - elapsed_time) * 1000))  # in milliseconds
            self.root.after(delay, self.run_simulation)  # Update approximately every 16ms

    def update_canvas(self):
        for boid in self.simulation.boids:
            x, y = boid.position
            oval = self.boid_reprs.get(boid.id)
            if oval:
                # Update position
                self.canvas.coords(
                    oval,
                    x - boid.flock.size, y - boid.flock.size,
                    x + boid.flock.size, y + boid.flock.size
                )
                # Update color in case it changes (optional)
                self.canvas.itemconfig(oval, fill=boid.flock.color)

    def export_data(self):
        try:
            self.simulation.export_to_csv()
            messagebox.showinfo("Export Successful", "Simulation data has been exported successfully.")
            self.export_button.config(state=tk.DISABLED)
        except Exception as e:
            print("Error exporting data:", e)
            messagebox.showerror("Export Error", "An error occurred while exporting data:\n{}".format(e))

    def reset_simulation(self):
        if messagebox.askyesno("Reset Simulation", "Are you sure you want to reset the simulation?"):
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Ready")

            # Clear simulation data
            self.simulation.flocks.clear()
            self.simulation.boids.clear()
            self.simulation.obstacles.clear()
            self.simulation.data_records = []
            self.simulation.next_flock_id = 1
            self.frame_number = 0

            # Clear canvas except obstacles
            self.canvas.delete("all")
            self.boid_reprs.clear()

            # Re-initialize obstacles (none, as per user request)
            self.initialize_obstacles()

            # Reset timer
            self.start_time = None
            self.timer_label.config(text="Elapsed Time: 00:00:00")

    def update_timer(self):
        if self.running and self.start_time:
            current_time = time.time()
            elapsed_seconds = int(current_time - self.start_time)
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            seconds = elapsed_seconds % 60
            time_string = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
            self.timer_label.config(text="Elapsed Time: {}".format(time_string))
            # Schedule the next timer update after 1 second
            self.root.after(1000, self.update_timer)

# ------------------------------
# Main Function
# ------------------------------
def main():
    root = tk.Tk()
    root.title("Boid Simulation with Export Functionality")

    # Initialize simulation
    simulation = Simulation(width=800, height=600)

    # Add initial flock
    simulation.add_flock(color="blue", num_boids=30, max_speed=4.0, max_force=0.05, size=3)

    # Initialize GUI
    gui = BoidGUI(root, simulation)

    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
