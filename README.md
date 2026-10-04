# Robot Pathfinding and Multi-Robot Path Planning

Implementation and experimental comparison of A* heuristic search for single-robot grid pathfinding and multi-robot path planning using Independent A*, Prioritized Planning, Conflict-Based Search (CBS), and a proposed Hybrid Prioritized Planning + CBS (H-PCBS) approach.

## Project Overview

This project explores pathfinding and multi-robot coordination in a 2D grid environment containing obstacles.

The project is divided into two major problems:

- **Q1 — Single-Robot Pathfinding:** A* search using different heuristic functions.
- **Q2 — Multi-Robot Pathfinding:** Comparison of four approaches for planning collision-free paths for multiple robots.

The algorithms are evaluated using metrics such as:

- Path cost
- Sum of Costs (SOC)
- Makespan
- Number of expanded nodes
- Number of collisions
- Runtime
- Success rate

---

# Q1 — Single-Robot Pathfinding

## Problem Definition

A robot must navigate from a specified start cell to a goal cell in a 2D grid containing randomly generated obstacles.

The A* algorithm is implemented using four different heuristics:

1. **Manhattan Distance**
2. **Euclidean Distance**
3. **Chebyshev Distance**
4. **Zero Heuristic (Dijkstra's Algorithm)**

The different heuristics are compared in terms of search efficiency and solution quality.

## Heuristics

### Manhattan Distance

\[
h(n) = |x_n-x_g| + |y_n-y_g|
\]

Suitable for four-directional grid movement.

### Euclidean Distance

\[
h(n) = \sqrt{(x_n-x_g)^2+(y_n-y_g)^2}
\]

Measures the straight-line distance between the current node and goal.

### Chebyshev Distance

\[
h(n) = \max(|x_n-x_g|,|y_n-y_g|)
\]

### Zero Heuristic

\[
h(n)=0
\]

This reduces A* to Dijkstra's algorithm.

---

# Q2 — Multi-Robot Path Planning

## Problem Definition

Multiple robots must simultaneously navigate through the same grid from individual start positions to individual goal positions.

The main challenge is avoiding conflicts between robots while maintaining efficient paths.

The following algorithms are implemented and compared:

### 1. Independent A*

Each robot independently computes an A* path without considering other robots.

**Advantages**
- Simple
- Fast
- Easy to implement

**Limitation**
- Paths may collide with each other.

### 2. Prioritized Planning

Robots are assigned an ordering. Each robot plans its path while considering the paths already planned for higher-priority robots.

**Advantages**
- Computationally efficient
- Simpler than CBS

**Limitation**
- Strongly dependent on robot ordering
- May fail even when a collision-free solution exists.

### 3. Conflict-Based Search (CBS)

CBS is a two-level multi-agent pathfinding algorithm.

The high-level search detects conflicts between robot paths and creates constraints to resolve them.

The low-level search uses Space-Time A* to find paths satisfying the generated constraints.

**Advantages**
- Can produce collision-free solutions
- Explicitly resolves conflicts

**Limitation**
- Computational cost can increase as the number of robots and conflicts increases.

### 4. Hybrid Prioritized Planning + CBS (H-PCBS)

A hybrid approach proposed and implemented in this project.

H-PCBS first attempts **Prioritized Planning** to obtain a solution quickly.

If the prioritized solution is successful and collision-free, it is directly returned.

If conflicts remain or prioritized planning fails, the solution is used as the starting point for targeted CBS-style conflict resolution.

The hybrid approach attempts to combine:

- The speed of Prioritized Planning
- The conflict-resolution capability of CBS
- Targeted replanning instead of immediately performing a full CBS search

> H-PCBS is an implementation proposed for this project and is evaluated experimentally against the other approaches.

---

# Algorithms Implemented

```text
                    Pathfinding
                         |
              +----------+----------+
              |                     |
        Single Robot           Multi Robot
              |                     |
             A*              +------+------+
              |              |             |
        +-----+-----+    Independent      PP
        |     |     |          A*          |
    Manhattan Euclidean        CBS        H-PCBS
    Chebyshev  Zero
```

---

# Evaluation Metrics

## Single-Robot

### Path Cost

Number of movements required to reach the goal.

### Expanded Nodes

Number of grid nodes expanded by A* during the search.

### Runtime

Time required to compute the path.

## Multi-Robot

### Sum of Costs (SOC)

Total path cost of all robots:

\[
SOC = \sum_{i=1}^{N} Cost_i
\]

### Makespan

Time required for the last robot to reach its goal:

\[
Makespan = \max_i Cost_i
\]

### Collisions

Number of detected conflicts between robots.

### Runtime

Time required by the planning algorithm.

### Success

Whether the algorithm successfully generated valid paths for all robots.

---

# Project Structure

```text
robot-pathfinding-algorithms/
|
├── astar.py
├── benchmarks.py
├── cbs.py
├── grid.py
├── heuristics.py
├── hybrid_pp_cbs.py
├── main.py
├── prioritized_planning.py
├── simulation.py
├── space_time_astar.py
├── visualize.py
|
├── README.md
|
└── results/
    ├── q1/
    └── q2/
```

## File Description

| File | Description |
|---|---|
| `grid.py` | Grid generation, obstacle generation, start/goal generation |
| `heuristics.py` | Manhattan, Euclidean, Chebyshev and Zero heuristics |
| `astar.py` | Standard A* implementation |
| `space_time_astar.py` | A* search with time and collision constraints |
| `prioritized_planning.py` | Independent A* and Prioritized Planning |
| `cbs.py` | Conflict-Based Search implementation |
| `hybrid_pp_cbs.py` | Proposed H-PCBS implementation |
| `benchmarks.py` | Experimental benchmarking |
| `visualize.py` | Result visualization and graphs |
| `simulation.py` | Interactive Q1 and Q2 simulations |
| `main.py` | Main benchmark execution interface |

---

# Installation

## Requirements

- Python 3.9+
- NumPy
- Matplotlib

Install the required packages:

```bash
pip install numpy matplotlib
```

Clone the repository:

```bash
git clone https://github.com/<your-username>/robot-pathfinding-algorithms.git
cd robot-pathfinding-algorithms
```

---

# Running the Simulations

## Q1 — A* Simulation

Run the four heuristic simulations:

```bash
python simulation.py --q1
```

This demonstrates:

- Manhattan
- Euclidean
- Chebyshev
- Zero heuristic

## Q1 — Comparison

To compare the final paths obtained using different heuristics:

```bash
python simulation.py --q1compare
```

## Q2 — Multi-Robot Simulation

Run all four multi-robot algorithms:

```bash
python simulation.py --q2
```

The simulation compares:

```text
Independent A*
Prioritized Planning
CBS
H-PCBS
```

## H-PCBS Simulation

To specifically demonstrate the proposed hybrid algorithm:

```bash
python simulation.py --q2hybrid
```

## Q2 — Four Algorithm Comparison

```bash
python simulation.py --q2compare
```

This displays the solutions obtained by:

```text
Independent A*
Prioritized Planning
CBS
H-PCBS
```

on the same grid and robot configuration.

---

# Benchmarking

The benchmarking framework evaluates the algorithms over multiple:

- Grid sizes
- Obstacle densities
- Number of agents
- Random trials

The benchmark results can be used to compare:

- Runtime
- Success rate
- Sum of Costs
- Makespan
- Collision rate
- Search effort

Run the benchmark using:

```bash
python main.py --q1
```

and:

```bash
python main.py --q2
```

---

# Simulation Environment

The environment is represented as a 2D grid.

Example:

```text
S . . # . . .
. # . # . # .
. # . . . # .
. . . # . . .
# . . . # . G
```

Where:

```text
S = Start
G = Goal
# = Obstacle
. = Free cell
```

For multi-robot scenarios, each robot has its own start and goal position.

---

# Experimental Methodology

For a fair comparison, different algorithms are evaluated on the same generated grid configurations.

The experiments vary:

- Grid dimensions
- Obstacle density
- Number of robots
- Random seeds

Each configuration can be evaluated over multiple trials.

This allows the algorithms to be compared under different levels of environmental complexity and robot density.

---

# Expected Observations

### A*

Different heuristics can result in different numbers of expanded nodes while still producing optimal paths when the heuristic is admissible for the given problem formulation.

### Independent A*

Generally has low computational cost but does not explicitly account for interactions between robots.

### Prioritized Planning

Can be significantly faster than more complex multi-agent algorithms but may depend heavily on the chosen priority ordering.

### CBS

Explicitly resolves conflicts and can produce collision-free solutions, but its computational cost can increase as the number of agents and conflicts increases.

### H-PCBS

Attempts to exploit the speed of Prioritized Planning while using targeted CBS-style conflict resolution when necessary.

The experimental results determine whether this trade-off is beneficial for the tested environments.

---

# Simulation

The `simulation.py` file provides animated visualizations of the algorithms.

The simulation windows are maximized automatically to make them suitable for demonstrations and screen recording.

Example:

```bash
python simulation.py --q1
```

or:

```bash
python simulation.py --q2
```

---

# Technologies Used

- **Python**
- **A* Search**
- **Dijkstra's Algorithm**
- **Space-Time A***
- **Prioritized Planning**
- **Conflict-Based Search (CBS)**
- **Multi-Agent Path Finding**
- **Matplotlib**
- **Algorithmic Benchmarking**

---

# Academic Context

This project was developed as part of the **CSMI17 — Artificial Intelligence** coursework.

The project focuses on applying classical and multi-agent search techniques to grid-based robotic pathfinding problems and experimentally comparing their performance.

---

# Authors

**Rupesh G**
**Sameer Basha**
**Sandheep Muthiah Suresh**
B.Tech Mechanical Engineering  
National Institute of Technology, Tiruchirappalli (NIT Trichy)

Interests:

- Robotics
- Autonomous Systems
- Motion Planning
- Control Systems
- Mechanical Design
- Artificial Intelligence

---

# License

This project was developed for academic purposes.

If you use or modify this repository, please provide appropriate attribution.
