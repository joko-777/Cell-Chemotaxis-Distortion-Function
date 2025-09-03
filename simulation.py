# -----------------------------------------------------------------------------
# Author: Johannes Konrad
# Affiliation: Institute for Digital Communications, Friedrich-Alexander-Universität Erlangen-Nürnberg]
# Email: johannes.konrad@fau.de
#
# This code is associated with the manuscript:
# "Cell Chemotaxis Distortion Function" (in preparation)
#
# The code implements the methods and algorithms described in the manuscript.
# Please refer to the paper for a detailed explanation of the methodology, 
# experimental design, and results.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def simulate_for_hill(hill_coefficient, seed):
    rng = np.random.default_rng(seed)

    logfile = open(f"./data/hill_{hill_coefficient}.csv", "w")

    # Fixe Parameter
    num_cells  = 500
    velocity   = 1e-5/60       # µm/min
    time_step  = 10.0       # s
    #step_len   = velocity * (time_step / 60.0)  # µm/Schritt
    N          = 100
    total_time = 200000.0      # s
    steps      = int(total_time / time_step)
    epsiolon = 0
    tetaBias = 0

    a, b, Kd, RT = 220.0, 20.0, 200.0, 1000

    hi = np.linspace(0, 2*np.pi, N, endpoint=False)
    hi_idx = np.arange(0, N)

    # feste Quelle (kann hier später ebenfalls zufällig gezogen werden)
    source = np.array([0.0, 0.0])

    # Simulation
    def createPos():
        r  = np.random.uniform(0, 100, (num_cells,))
        angle = np.random.uniform(0, 2 * np.pi, (num_cells,))
        positions = np.stack([r * np.cos(angle), r * np.sin(angle)])
        return positions.swapaxes(0, 1)
    positions = createPos()

    for _ in tqdm(range(steps), desc=f"Simulatino for {hill_coefficient}"):
        tetaS = np.arctan2(source[1] - positions[:,1], source[0] - positions[:,0])
        tetaS = np.where(tetaS < 0, tetaS + 2 * np.pi, tetaS)
        
        L = a - b * (1 - np.cos(hi[None, :] - tetaS[:, None]))


        p = L / (Kd + L)

        for i in range(num_cells):
            C = np.random.binomial(RT, p[i])
            Y = C.copy()

            Y = Y * (1 + epsiolon * np.cos(hi - tetaBias))
            
            Yp = Y - Y.min()
            r = (Yp ** hill_coefficient)
            r /= r.mean()
            
            P = r / r.sum()
            idx = np.random.choice(hi_idx, p=P)

            hr = hi[idx]

            true_idx = np.argmin(np.abs((tetaS[i] - hi  + 2 * np.pi) % (2 * np.pi)))

            logfile.write(f"0,0,{hr},{tetaS[idx]},0,{idx},0,{true_idx}\n")

            positions = createPos()
    logfile.close()
    return True
if __name__ == "__main__":
    hillcoef = [1, 3, 5, 7, 9, 15]

    # Boolean flag to determine parallel execution
    parallel_execution = True

    # Generate a seed for each Hill coefficient
    seeds = np.random.SeedSequence(1234).spawn(len(hillcoef))

    if parallel_execution:
        # Use ProcessPoolExecutor to run simulations in parallel
        with ProcessPoolExecutor(max_workers=2) as executor:
            executor.map(simulate_for_hill, hillcoef, seeds)
    else:
        # Run simulations sequentially
        for hill, s in zip(hillcoef, seeds):
            simulate_for_hill(hill, s)