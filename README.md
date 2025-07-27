# Enhanced Sparrow Search Algorithm for 5G Base Station Placement

A Streamlit web application for optimizing 5G base station placement using Random Walk Sparrow Search Algorithm.

## Abstract:

The objective of this project is to implement and analyze an enhanced version of the Sparrow Search Algorithm (SSA) for optimizing 5G base station placement in a
constrained environment. This involves simulating real-world deployment scenarios where the goal is to maximize signal coverage over a predefined area while avoiding
overlap and adhering to various spatial constraints. The algorithm incorporates realistic communication parameters like signal radius and effective range, and includes anti-clustering penalties and smooth convergence mechanisms. The enhanced SSA introduces role-based behaviors—producers and scroungers—as well as anti-predator movements to prevent local minima trapping. Additional features such as grid-based fitness evaluation, spatial visualization of coverage, and heatmap-based performance metrics were implemented. 

## Implimentation
The sparrows (representing 5G tower configurations) use a **fitness function** to evaluate the effectiveness of tower placement.

1. **Fitness Function Definition:**
    - The fitness function measures how effectively the current tower configuration covers the mine area.
    -  It uses Rcov as one of its fitness dimensions which indicates the proportion of mine regions (pixels) covered by the current tower placement.
2. **Global Exploration Strategy:**
    - First, sparrows (tower configurations) are randomly placed throughout the mine.
    - Each configuration's fitness value is then calculated.
    - Sparrows "fly" to new positions using **random steps** and insights from previous iterations (as defined in formula 7).
3. **How Does the Sparrow Move?**
    - A sparrow explores the mine by moving to a **new random location**, guided by:
        - A **random direction** for exploring unexplored areas
        - Influences from the best-performing sparrow (the one with optimal fitness)
    - This random movement enables efficient exploration of **large areas**.
4. **Identifying a Good Region:**
    - Sparrows evaluate and compare fitness values at different positions.
    - Areas with the **highest fitness value** (best coverage ratio) are marked as promising.
    - The search then intensifies in these promising regions during **subsequent iterations**

Sparrow Roles (Discoverers vs. Joiners):
    - The sparrow population is divided into two groups:
        - **Discoverers (top-performing sparrows):**
        These sparrows have better fitness values—they represent tower configurations with higher coverage. They **explore globally**, guiding the search process.
        - **Joiners (remaining sparrows):**
        These sparrows have lower fitness values. They **follow discoverers** and perform **local exploration** around promising regions.
        - Joiners continuously observe discoverers to track high-fitness regions.
        - If a discoverer finds better food, joiners compete for it.
        - If a joiner wins the competition (achieves a higher fitness), it replaces the discoverer at that position.

RESULTS:

Through analysis, among the solution results of the four
algorithms, the result of RWSSA is better than the other three
algorithms.And it can be determined that the greater the number
of macrobase stations, the better the signal coverage, but the
greater the construction cost of the 5G network. Therefore,
while meeting the signal coverage requirements, the smaller the
number of base stations, the lower the construction cost of 5G
private networks.
In summary,RWSSA has achieved a higher optimal coverage rate, a more
uniform5Gbase station distribution and fewer network coverage
blind spots, which validates the algorithm it has better network
coverage optimization performance.


## Prerequisites
- Python 3.7+
- pip package manager

## Installation
1. Install dependencies:
   pip install streamlit numpy matplotlib scipy pandas

2. Run the application:
   streamlit run run.py

## Application Modes

### Single Run Mode
Purpose: Optimize base station placement for a single configuration (as implemented in the research paper).

Parameters (Left Sidebar):
- Area Configuration:
  - Width (5-20 units)
  - Height (5-20 units)
  - Boundary Buffer (0.5-3.0 units)
  
- Base Station Setup:
  - Number of Stations (1-10)
  - Coverage Radius (0.5-5.0 units)

- Target Areas:
  - Number of Pit Areas (1-5)
  - Individual Pit Sizes (0.5-3.0 units)

- Algorithm Settings:
  - Sparrow Population (10-200)
  - Max Iterations (50-500)
  - Early Stop (10-100 iterations)
  - Grid Resolution X (50-200 points)
  - Grid Resolution Y (50-200 points)

### Parameter Optimization Mode
Purpose: Compare multiple configurations with additional parameters (an improved version of the paper’s implementation).

Additional Parameters:
- Station Range:
  - Minimum Stations (1-10)
  - Maximum Stations (5-15)

- Radius Analysis:
  - Minimum Radius (0.5-5.0)
  - Maximum Radius (1.0-10.0)
  - Radius Step Size (0.1-1.0)

## Usage Guide
1. Select mode using the sidebar radio buttons
2. Adjust parameters using sliders/number inputs
3. Click "Run Optimization" (Single Mode) or "Run Parameter Optimization" (Parameter Mode if you want optimize using additional parameters)
4. Analyze results through:
   - Interactive heatmaps
   - 3D parameter surfaces
   - Coverage probability visualizations
   - Station placement coordinates
   - Convergence graphs

## Key Features
- Dynamic Visualization: Real-time coverage maps with pit/avoidance zones
- Multi-Objective Optimization: Balances coverage, cost, and constraints
- Adaptive Resolution: Automatically adjusts grid density for accuracy
- Export Capabilities: Save results as JSON/CSV
- Comparative Analysis: Side-by-side parameter comparisons

Optimization Tips:
- Start with wider parameter ranges
- Balance station count vs coverage radius
- Use smaller radius steps for precision
- Monitor convergence graphs for stability

Computational Notes
- Parameter Optimization mode may take several minutes depending on parameter ranges and grid resolution.

##Inputs for the uploaded outputs

##Single Run Mode Configuration

Area Configuration:
-Width: 10 units
-Height: 10 units
-Boundary Buffer: 2.0 units

Base Station Setup:
-Number of Stations: 2
-Coverage Radius (Rs): 2.0 units

Target Areas:
-Number of Pit Areas: 2
-Pit Sizes: 1.0 units (for both pits)

Algorithm Settings:
-Sparrow Population: 100
-Max Iterations: 300
-Early Stop Iterations: 50
-Grid Resolution X: 100 points
-Grid Resolution Y: 100 points
-Random Seed: 42

## Parameter Optimization Mode

Parameter Ranges:
-Minimum Stations: 2
-Maximum Stations: 2
-Minimum Coverage Radius: 1.5
-Maximum Coverage Radius: 3.00
-Radius Step Size: 0.5

Area Configuration:
-Width: 10 units
-Height: 9 units
-Boundary Buffer: 0.9 units

Target Areas:
-Number of Pit Areas: 2
-Pit Sizes: 1.00 units (for both pits)

Algorithm Settings:
-Sparrow Population: 49
-Max Iterations: 200
-Early Stop Iterations: 40
-Grid Resolution X: 100 points
-Grid Resolution Y: 100 points
-Random Seed: 42


