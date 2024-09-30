# Boid Simulation
A Python-based Boid Simulation demonstrating flocking behaviors using Tkinter, NumPy, and Pandas. This simulation allows users to create multiple flocks with customizable parameters and export simulation data for further analysis.

# Features
<ul>
<li>

**Multiple Flocks:** Add and customize multiple flocks with different colors, sizes, and behaviors.
**Movement:** Boids exhibit separation, alignment, and cohesion behaviors.
</li>

<li>

**Edge Bouncing**: Boids bounce off the edges of the simulation window for realistic boundary interactions.

</li>
<li>

**Data Export:** Export simulation data to CSV files for analysis.
</li>
<li>

**User-Friendly GUI:** Intuitive interface built with Tkinter for easy interaction and customization.
</li>

</ul>

# Prerequisite
<ul>
<li>

**Conda**

</li>

<li>

**Python**
</li>
</ul>

## Setup
<ol>
<li>

**Clone the repository.**

</li>

`git clone <repository url>`
<li>

**Navigate to project directory**

</li>

`cd bold_simulation`

<li>

**Create Conda Environment**

</li>

`conda env create -f environment.yml`

or if(above doesn't work)

`conda create -n <environment-name> --file req.txt`
<li>

**Activate the environment**

</li>

`conda activate boid-env`

<li>

**Run the environment.**

`python boidv4.py`
</li>
</ol>

# Export Data

To export the simulation data to a CSV file within the data folder:

<ol>
<li> Run the simulation.
</li>
<li>
Once you've configured your flocks and parameters, click the "Export CSV" button in the GUI.
</li>
<li>
The data will be saved as boid_simulation_data.csv inside the data directory of your project.

</ol>

# Resetting the Simulation

To reset the simulation to its initial state:
<ol>
<li>
Click the "Reset" button in the GUI.
</li>
<li>
Confirm the action when prompted.
</li>
<li>
All existing flocks will be cleared, and the simulation will be ready for new configurations.
</li>
</ol>