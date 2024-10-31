# Boid Simulation Data Analysis
Data analysis based on boid simulation to see if we can identify flocks using tracks (posioning, positioning + velocity, and other features). 

# Models
<ul>
<li>

**K-means:** K-Means clustering by finding the cluster with the nearest means.
**DBScans:** Density-Based Spacial Clustering (and Space-Temporal DBscan)
</li>

<li>

**Data used**: data_analysis.ipynb uses boid_simulation_data.scv and DBScan_analysis.ipynb uses boid_simulation_datav2.csv

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

`conda activate <environment-name>>`

<li>

**Run the environment.**

`jupyter notebook`

</li>
</ol>

