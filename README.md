# Cell Chemotaxis Distortion Function

# Introduction
This repository contains simulation and analysis tools to study **cellular decision-making** in the context of chemotaxis and apoptosis, using an **information-theoretic framework**.  

The workflow consists of three main components:  
1. **Simulation (`simulation.py`)**  
   - Generates synthetic datasets of cell movement choices under varying **Hill coefficients** by the LEGI model.  
   - Saves the results in `./data/` as CSV files.  

2. **Chemotaxis Analysis (`chemotaxis.py`)**  
   - Processes the simulation outputs.  
   - Computes **decision strategies**, **distortion matrices**, and **cosine-based distortion comparisons**.  
   - Produces plots (`.pdf`), CSV exports, and LaTeX tables in `./chemotaxis_figures/`.  

3. **Apoptosis Rate–Distortion Analysis (`apoptosis.py`)**  
   - Implements the **Blahut–Arimoto algorithm**.  
   - Compares different **distortion functions** (hamming-like vs quadratic).  
   - Produces rate–distortion curves, decision strategies, and distortion plots in `./aptosis_figures/`. 
   Sometimes the hamming-like distortion function is called 

Together, these tools allow the exploration of how cells encode and process directional or survival signals, and how **information theory** can quantify the efficiency and reliability of these biological processes.  

---

# Chemotaxis Distortion Analysis

## Overview
This script analyzes **chemotaxis decision strategies** by reading simulation data, computing **distortion matrices**, and generating both plots and CSV data exports. It is designed to evaluate how movement direction distributions depend on source direction under different **Hill coefficients**, and compares the distortion to a cosine-based distortion function.  

The workflow:
1. Load simulation data from `./data/hill_{hill}.csv`.  
2. Construct histograms of chosen vs. true directions.  
3. Compute conditional probability matrices and distortion matrices.  
4. Generate multiple plots (`.pdf`) and export their data (`.csv`).  
5. Create a LaTeX table summarizing mean squared error (MSE) between distortion curves and the cosine distortion function.  

All results are stored inside the `./chemotaxis_figures` directory.  

---

## Requirements
- Python 3.x  
- NumPy  
- Matplotlib  

## Input Data
The script expects CSV files in the following format:


- **step**: simulation step index  
- **agent_id**: ID of the simulated agent  
- **chosen_theta**: chosen movement direction (radians, [0, 2π))  
- **true_theta**: true source direction (radians, [0, 2π))  
- **old_direction**: previous direction (not directly used here)  
- **chosen_theta_idx**: index in discretized bins (0 … num_bins_choosen-1)  
- **true_theta_idx**: index in discretized bins (0 … num_bins_true-1)  

File naming convention:

./data/hill_{hill}.csv


where `{hill}` is a Hill coefficient value (e.g., `1`, `3`, `5`, `7`, `9`, `15`).  

---

## Output
All results are stored in:


### Plots (`.pdf`)
- **Decision strategy heatmap**: `decisionstrategy_{hill}.pdf`  
- **Distortion heatmap**: `distortion_{hill}.pdf`  
- **Shifted distortion plots**: `distortion2dshifted_{hill}.pdf`  
- **Comparison of mean distortions across Hill coefficients**:  
  - `totalcosineabsolut.pdf` (raw mean distortions)  
  - `totalcosine.pdf` (compared to cosine distortion)  

### Data (`.csv`)
For each plot, the underlying numerical data is exported as `.csv` with the same base name as the figure.  

### Table (`.tex`)
- `hillerror.tex`: LaTeX table summarizing the **MSE** between the computed distortion and cosine distortion function for each Hill coefficient.  

---

## Usage
Run the script directly:

python chemotaxis.py

# Apoptosis Rate–Distortion Analysis

## Overview
This script implements a **rate–distortion analysis** for apoptosis-like binary decision problems using the **Blahut–Arimoto algorithm**. It evaluates different distortion functions (Hamming-like and quadratic) to study how reliably a system can encode decisions about molecule counts given noise constraints.  

The script:  
1. Defines probability distributions over molecule counts.  
2. Computes rate–distortion curves for different distortion functions.  
3. Selects operating points at chosen distortion levels.  
4. Evaluates the resulting **decision strategies** and **distortion matrices**.  
5. Produces plots (`.pdf`) and exports underlying data (`.csv`).  

All results are saved in `./aptosis_figures`.  

---

## Requirements
- Python 3.x  
- NumPy  
- Matplotlib  
- tqdm  

---

## Methodology
- **Blahut–Arimoto Algorithm** (`bablahutAlgo`):  
  Iteratively computes the optimal conditional distribution that minimizes distortion for a given Lagrange multiplier `λ`.  
- **Distortion Functions**:  
  - *Indicator (Hamming-like)*: Penalizes incorrect binary classification around a threshold.  
  - *Quadratic*: Penalizes squared deviations from a threshold.  
- **Rate–Distortion Curve**:  
  Plots the tradeoff between distortion `D` and mutual information `R`.  
- **Decision Strategy Evaluation**:  
  At specific distortion levels, evaluates and plots optimal decision strategies and resulting distortion matrices.  

---

## Input
This script is **self-contained** and does not require external data files.  
- Molecule counts are defined as `x ∈ [0, N]` with exponential prior distribution `px`.  
- Distortion functions are defined internally (`indicator`, `quadraticDistortion`).  

---

## Output
All outputs are saved to:

./aptosis_figures/


### Plots (`.pdf`)
- **Distortion functions**:  
  - `binarydistortionorignal_{name}.pdf` (original distortion functions)  
  - `binarydistortioncalc_{name}.pdf` (calculated distortion functions)  
- **Decision strategies**:  
  - `binary_strategy_{name}.pdf`  
- **Rate–distortion curves**:  
  - `ratedistortioncurvetotal.pdf`  

### Data (`.csv`)
For each plot, the underlying data is exported as `.csv` with the same basename.  

### Tables (`.tex`)
- `ratedistortioncurve_{name}.tex`: Contains the **maximum distortion** value in LaTeX format.  

---

## Usage
Run the script directly:

python aptosis.py


# Chemotaxis Hill Coefficient Simulation

## Overview
This script simulates **cellular chemotaxis** under different **Hill coefficients**, generating datasets that describe how cells choose movement directions in response to a chemical source.  

The output of this simulation (`hill_{hill}.csv`) serves as the **input data** for the analysis scripts (e.g., distortion/decision strategy analysis).  

---

## Requirements
- Python 3.x  
- NumPy  
- Matplotlib (only used indirectly for imports, not plotting here)  
- tqdm (progress bars)  
- scikit-learn (for `mutual_info_score`, though not directly used in the main loop)  

---

## Simulation Details
- **Cells**: `num_cells = 500`  
- **Velocity**: `1e-5 µm/min`  
- **Time step**: `10 s`  
- **Total simulation time**: `200,000 s`  
- **Receptors**: `RT = 1000`  
- **Ligand parameters**: `a = 220.0`, `b = 20.0`, `Kd = 200.0`  
- **Angles**: `N = 100` uniformly spaced between `0` and `2π`.  

At each time step, for each cell:  
1. Compute the **source direction** (`θ_s`).  
2. Calculate ligand concentration profile `L(θ)`.  
3. Sample receptor occupancy via a **binomial distribution**.  
4. Apply Hill transformation with the given Hill coefficient.  
5. Generate a **probability distribution over movement directions**.  
6. Sample the chosen direction (`θ_m`).  
7. Save the result as a row in the output file.  

---

## Output
Simulation results are stored in:

./data/hill_{hill}.csv

Each row has the format:


- **step**: currently fixed to `0`  
- **agent_id**: currently fixed to `0`  
- **chosen_theta**: chosen movement direction (radians, [0, 2π))  
- **true_theta**: actual source direction at the chosen index  
- **old_direction**: fixed `0` in this implementation  
- **chosen_theta_idx**: discretized index of chosen direction  
- **true_theta_idx**: discretized index of source direction  

---

## Usage
Run the simulation:

python simulation.py
