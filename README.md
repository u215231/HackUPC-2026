# HackUPC 2026: Warehouse Optimizer for Mecalux

- School: Universitat Politècnica de Catalunya
- Business: Mecalux S.A.
- Authors:
    - Marc Bosch Manzano
    - Christian Clemente García
    - Pinda Li
    - Iman Shahzad Shafiq
- Creation: 2026/04/25

---

## 🏭 Overview
This project solves a **2D Bin Packing and Warehouse Layout Optimization** challenge proposed by **Mecalux S.A.** at HackUPC 2026. 
The goal is to automatically generate optimal layouts for warehouse storage bays that minimize the cost-per-load ratio while maximizing the space utilized inside the warehouse footprint, considering structural constraints and real-world logistics rules.

## 🎯 Optimization Problem

### Objective Function
The optimizer minimizes a highly non-linear objective score calculated as:
`Score = (totalPrice / totalLoads) ^ (2 - areaPct)`
- A lower score is better.
- It rewards maximizing total loads and minimizing total price.
- `areaPct` is the percentage of usable area filled by bays. Filling more of the warehouse drives the exponent towards 1, reducing the penalty (since price/load > 1 generally).

### Parameters & Characteristics
- **Bays (Candidates):** Different models of storage bays, each with distinct dimensions (width, depth, height), total loads capacity, price, and required front access gap.
- **Warehouse Footprint:** Modeled as a 2D polygon.
- **Obstacles:** Static rectangular areas (e.g., pillars) that cannot be occupied.
- **Ceiling Height:** The warehouse has varying ceiling heights defined by step-functions across the X-axis.

### Constraints & Restrictions
- **Inside the Warehouse:** All storage bays and their required operational gaps must reside entirely within the warehouse polygon.
- **No Internal Collisions:** The physical footprint of a bay cannot overlap with other bays or obstacles. (Border touching is allowed).
- **Front Gap Rules:**
  - Bays need an empty rectangular gap in front of them to access loads.
  - The gap area cannot intersect with the footprint of any bay or obstacle.
  - However, the front gaps of multiple bays **can** overlap with each other (aisle sharing).
- **Height Constraints:** The height of a placed bay must not exceed the minimum ceiling height in the span it occupies.

## ⚙️ Algorithm Architecture

Our approach combines a deterministic heuristic constructor with a parallelized stochastic improver:

1. **Candidate Generation & Anchor Grid:**
   Instead of purely continuous placement, we construct a discrete grid using `gridStep` (default: 500mm), `angleStep` (default: 10°), and crucial geometric anchors (corners of the warehouse, obstacles, and ceiling breaks) to generate millions of feasible bay placements.
   
2. **Deterministic Greedy Construction:**
   - Evaluates feasible candidates and prioritizes them based on efficiency (loads/price), bay area, and spatial heuristics like corner-hugging and wall proximity.
   - It utilizes OpenMP parallelization to evaluate compatibility and awards bonuses for shared front gaps and touching neighbors to cluster bays efficiently.
   - It performs passes for cardinal angles (0°, 90°, 180°, 270°) before attempting oblique placements.

3. **Parallel Simulated Annealing (SA):**
   - Applies an SA optimization loop to iteratively refine the greedy solution.
   - Explores the neighborhood by randomly inserting, removing, and swapping bays.
   - Leverages OpenMP parallel threads to evaluate multiple mutation branches per iteration and accepts the best non-deteriorating move or allows worse moves probabilistically based on the temperature parameter.
   - Prunes redundant bays that negatively impact the non-linear objective score.

## 🛠️ Requirements

- **C++17 Compiler** with OpenMP support (e.g., GCC/g++)
- Unix-like environment (Linux / macOS) for executing shell scripts.
- Python 3.x (optional, for running auxiliary Python tools if provided).

## 🚀 How to Execute

### 1. Build the solver
Navigate to the `mecalux/Scripts` directory and run the build script. This will compile the C++ source code with `-O3` optimizations and OpenMP, and place the executable `solver.out` in the `bin` directory.

```bash
cd mecalux/Scripts
./build.sh
```

### 2. Run the solver
You can use the provided run script which points to a test case. You can use both the run.sh script or the python.sh script. The first one will run just the algorithm and the second one will run the algorithm and generate a visual representation of the solution.

```bash
./run.sh
```

or visualize the results

```bash
./python.sh
```

**Manual Execution Example:**
If you want to manually run the solver with custom parameters, the usage format is:
`./solver.out <case_dir> <output_csv_path> <run_sa_bool> [gridStep] [angleStep]`

```bash
cd mecalux/bin
# Fast test (Simulated Annealing enabled, 500mm grid, 10° rotation step)
./solver.out ../PublicTestCases/Case0 solution.csv 1 500 10

# High-quality test (Simulated Annealing disabled, finer 250mm grid, 5° rotation step)
./solver.out ../PublicTestCases/Case0 solution.csv 0 250 5
```

## 📊 Expected Output & Behavior

- **Terminal Output:** The solver outputs the warehouse properties (vertices, area, etc.), candidate generation statistics, and progress of the Greedy and Simulated Annealing algorithms. It will report the final objective score.
- **Result File:** The solution will be written to `solution.csv` (or the path you specified) containing the final chosen bays in the required format: `Id, X, Y, Rotation`.
- The program completes by printing the total execution time to the standard error output.