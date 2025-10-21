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
from tqdm import tqdm
import csv
import os

BASE_PATH = "./aptosis_figures"

os.makedirs(BASE_PATH, exist_ok=True)

FONT = 13
FONTBIG = 24
LINEWIDTH = 2.5

plt.rcParams.update({
    "font.size": FONT,          # default font size for all text
    "axes.labelsize": FONT,     # x and y labels
    "xtick.labelsize": FONT,    # x ticks
    "ytick.labelsize": FONT,    # y ticks
    "legend.fontsize": FONT,    # legend
    "axes.titlesize": FONT,     # title
    "text.usetex": True,

})


#### FOR EDITING PLOTS SET THIS VALUE TO 10. FOR FINAL VERSION INCREASE AGAIN TO 10000 ####
resolution = 10000
N = 1200

ext = "pdf"

def export_all_plot_data_to_csv(base_filename, all_xaxis=False):
    """
    Exports all line and image data from the current matplotlib Axes.
    - If all_xaxis is False:
        - Lines are saved as: base_filename (x + multiple y columns)
    - If all_xaxis is True:
        - Each line's x and y data are saved in separate columns (x1, y1, x2, y2, ...)
    - Image (imshow) data are saved as: base_filename + '_image_<n>.csv'
    """
    ax = plt.gca()
    lines = ax.get_lines()
    images = ax.get_images()

    # Export line data
    if lines:
        line_filename = base_filename

        headers = []
        rows = []

        if all_xaxis:
            # Gather each x and y as separate column
            all_data = []
            max_len = 0

            for idx, line in enumerate(lines):
                x = line.get_xdata()
                y = line.get_ydata()

                if len(x) != len(y):
                    print(f"Error: Line {idx + 1} has mismatched x and y data lengths.")
                    return

                label = line.get_label()
                if label.startswith('_child'):
                    label = f'y{idx + 1}'

                headers.extend([f'x{idx + 1}', label])
                all_data.append((x, y))
                max_len = max(max_len, len(x))

            # Build row-wise data
            for i in range(max_len):
                row = []
                for x, y in all_data:
                    row.append(x[i] if i < len(x) else '')
                    row.append(y[i] if i < len(y) else '')
                rows.append(row)
        else:
            # Shared x axis
            x_data = lines[0].get_xdata()

            if len(x_data) == 0:
                print("Error: x_data is empty.")
                return

            headers = ['x']
            y_data_columns = []

            for idx, line in enumerate(lines):
                label = line.get_label()
                if label.startswith('_child'):
                    label = f'y{idx + 1}'
                headers.append(label)
                y_data_columns.append(line.get_ydata())

            if any(len(y_col) != len(x_data) for y_col in y_data_columns):
                print("Error: y_data_columns lengths do not match x_data.")
                return

            for i in range(len(x_data)):
                row = [x_data[i]] + [y_col[i] for y_col in y_data_columns]
                rows.append(row)

        # Write to CSV
        with open(line_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"Line data exported to '{line_filename}'")

    else:
        print("No line plots found.")

    # Export image (imshow) data
    if images:
        for idx, img in enumerate(images):
            img_data = img.get_array().data
            img_filename = f"{base_filename}_image_{idx + 1}.csv"
            np.savetxt(img_filename, img_data, delimiter=',')
            print(f"Image data exported to '{img_filename}'")
    else:
        print("No image data (imshow) found.")

## BLAHUT ##

def bablahutAlgo(px, d, beta, tol=1e-12,max_iter=1000):
    epsilon = 1e-10

    assert px.shape[0] == d.shape[0], "px and d row mismatch"


    n_y, n_z = d.shape
    pz = np.full(n_z, 1.0 / n_z)

    assert np.sum(pz) == 1

    for i in range(max_iter):
        exp_term = np.exp(-beta * d) * pz  # shape: (n_y, n_z)
        denominator = np.sum(exp_term, axis=1, keepdims=True)

        pzy = exp_term / denominator
        pz_new = np.sum(px[:, np.newaxis] * pzy, axis=0)

        if np.linalg.norm(pz_new - pz, 1) < tol:
            pz = pz_new
            break
        
        pz = pz_new
    # Berechne Rate

    def safe_log2(x):
        return np.log2(np.clip(x, 1e-12, None))

    ratio = pzy / (pz + epsilon)  # |Y| x |Z|
    R = np.sum(px[:, np.newaxis] * pzy * safe_log2(ratio))

    # Durchschnittliche Verzerrung
    D_avg = np.sum(px[:, np.newaxis] * pzy * d)

    return R, D_avg, pzy, pz
lamb = 0.005
xvals = np.linspace(0, N, 1000)
px = lamb * np.exp(-lamb * xvals)
px /= np.sum(px)

def maineval(distortionFunction, name, lam):

    distortionMatrix = distortionFunction()

    R, D_avg, py_x, py = bablahutAlgo(px, distortionMatrix, lam)

    labels = [r"$y=1$", r"$y=0$"]

    #### FIG 5 (a) and (b) ####
    plt.figure(figsize=(6, 5))
    for y in range(distortionMatrix.shape[1]):
        plt.plot(xvals, distortionMatrix[:,y], label=labels[y], linewidth=LINEWIDTH)
    #plt.title("Distortion Matrix (Original)")
    plt.xlabel(r"Number of molecules $x$", fontsize=FONTBIG)
    plt.ylabel(r"Distortion", fontsize=FONTBIG)
    plt.legend(fontsize=FONTBIG)
    plt.tick_params(axis='both', labelsize=FONTBIG)
    plt.xticks([0, 300, 600, 900, 1200])
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/binarydistortionorignal_{name}.{ext}")
    export_all_plot_data_to_csv(f"{BASE_PATH}/binarydistortionorignal_{name}.csv")
    plt.close()
    ## END ##

    #### FIG 7 (a), (b), (c) and (d) ####
    plt.figure(figsize=(6, 5))
    for y in range(py_x.shape[1]):
        plt.plot(xvals, py_x[:,y], label=labels[y], linewidth=LINEWIDTH)
    plt.xlabel(r"Number of molecules $x$", fontsize=FONTBIG)
    plt.ylabel(r"Decision strategy $P_{Y|X}(y|x)$", fontsize=FONTBIG)
    plt.legend(fontsize=FONTBIG)
    plt.tick_params(axis='both', labelsize=FONTBIG)
    plt.xticks([0, 300, 600, 900, 1200])
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/binary_strategy_{name}.{ext}")
    export_all_plot_data_to_csv(f"{BASE_PATH}/binary_strategy_{name}.csv")
    plt.close()
    ## END ##

    print("CHOOSE PROB", py_x[-1, :])


    distortionMatrixCalc = py_x / py[np.newaxis,:]
    distortionMatrixCalc = np.where(np.isnan(distortionMatrixCalc), 10**12, distortionMatrixCalc)
    distortionMatrixCalc = -np.log(distortionMatrixCalc)
    distortionMatrixCalc = distortionMatrixCalc - np.min(distortionMatrixCalc, axis=1)[:,np.newaxis]


    # === Plot === #
    #### FIG 8 (a), (b), (c) and (d) ####
    plt.figure(figsize=(6, 5))
    for y in range(distortionMatrix.shape[1]):
        plt.plot(xvals, distortionMatrixCalc[:,y], label=labels[y], linewidth=LINEWIDTH)
    #plt.title("Distortion Matrix (Claculated)")
    plt.legend(fontsize=FONTBIG)
    plt.ylabel(r"Distortion", fontsize=FONTBIG)
    plt.xlabel(r"Number of molecules $x$", fontsize=FONTBIG)
    plt.tick_params(axis='both', labelsize=FONTBIG)
    plt.xticks([0, 300, 600, 900, 1200])
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/binarydistortioncalc_{name}.{ext}")
    export_all_plot_data_to_csv(f"{BASE_PATH}/binarydistortioncalc_{name}.csv")
    plt.close()

    ## END ##

distortionCurves = []
def ratedistortioncurve(dFunc, name, start=0.0001, stop=14000):

    distortionMatrix = dFunc()

    rate = []
    dist = []

    x1 = np.linspace(start, 5, resolution)
    x2 = np.linspace(5, stop, resolution)

    lambdas = np.concatenate((x1, x2), axis=0)

    for l in tqdm(lambdas, desc="Rate dist curve"):
        R, D_avg, _, _ = bablahutAlgo(px, distortionMatrix, l)
        rate.append(R)
        dist.append(D_avg)

    dist = np.asarray(dist)
    rate = np.asarray(rate)

    distortionCurves.append((dist, rate, name))

    idx = np.argmin(dist[rate < 10**-5])
    print("Max distortion", dist[idx])

    with open(f"{BASE_PATH}/ratedistortioncurve_{name}.tex", "w") as f:
        f.write(f"${round(dist[idx], 3)}$")
    
    return lambdas

if __name__ == "__main__":


    def quadraticDistortion():
        d = np.ones((xvals.shape[0], 2)) 
        yth = 600
        # Berechne die Cosinus-Distanz für jedes Paar (i, j)
        ## HIGH ##
        for i in range(xvals.shape[0]):
            if xvals[i] > yth:
                d[i, 0] = 0
            else:
                d[i, 0] = np.abs(yth - xvals[i])**2
        
        ## LOW ##
        for i in range(0, xvals.shape[0]):
            if xvals[i] < yth:
                d[i, 1] = 0
            else:
                d[i, 1] = np.abs(xvals[i] - yth)**2
        return 0.00005 * d
    
    def indicator():
        d = np.ones(shape=(xvals.shape[0], 2))
        yth = 600
        yth_idx = np.argmin((xvals - yth)**2)
        d[:yth_idx,1] = 0
        d[yth_idx:,0] = 0
        return d
    
    lamd1 = ratedistortioncurve(indicator, "hamming-like", start=0.0025)
    lamd2 = ratedistortioncurve(quadraticDistortion, "quadrat", start=0.0007)

    d1 = 0.031

    idx1 = np.argmin(np.abs(distortionCurves[0][0] - d1))
    idx2 = np.argmin(np.abs(distortionCurves[1][0] - d1))
    print(distortionCurves[0][1][idx1]) # 0.07997302736212292
    print(distortionCurves[1][1][idx2]) # 0.08007199585694154
    print(lamd1[idx1]) # 2.442713567839196
    print(lamd2[idx2]) # 0.9243718592964825

    d2 = 0.01
    iidx1 = np.argmin(np.abs(distortionCurves[0][0] - d2))
    iidx2 = np.argmin(np.abs(distortionCurves[1][0] - d2))
    print(distortionCurves[0][1][iidx1]) # 0.07997302736212292
    print(distortionCurves[1][1][iidx2]) # 0.08007199585694154
    print(lamd1[iidx1]) # 2.442713567839196
    print(lamd2[iidx2]) # 0.9243718592964825
    print(iidx1)
    print(iidx2)

    funcs = [(indicator, "indicatorlow", lamd1[idx1]), (indicator, "indicatorhigh", lamd1[iidx1]), (quadraticDistortion, "quadratlow", lamd2[idx2]), (quadraticDistortion, "quadrathigh", lamd2[iidx2])] # 3 ud 0.4

    for f, n, lam in funcs:
        maineval(f, n, lam)

    #### FIG 6 ####
    plt.figure(figsize=(6, 5))
    for dist, rate, name in distortionCurves:
        plt.plot(dist, rate, label="Rectified squared" if name == "quadrat" else "Hamming-like")
    plt.xlabel(r"$D$")
    plt.ylabel(r"Mutal Information")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/ratedistortioncurvetotal.{ext}")
    export_all_plot_data_to_csv(f"{BASE_PATH}/ratedistortioncurvetotal.csv", all_xaxis=True)
    plt.close()
    ## END ##
