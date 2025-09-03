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
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

BASE_PATH = "./chemotaxis_figures"

os.makedirs(BASE_PATH, exist_ok=True)

FONT = 13
plt.rcParams.update({
    "font.size": FONT,          # default font size for all text
    "axes.labelsize": FONT,     # x and y labels
    "xtick.labelsize": FONT,    # x ticks
    "ytick.labelsize": FONT,    # y ticks
    "legend.fontsize": FONT,    # legend
    "axes.titlesize": FONT,     # title
    "text.usetex": True,

})

def export_all_plot_data_to_csv(base_filename):
    """
    Exports all line and image data from the current matplotlib Axes.
    - Lines are saved as: base_filename + '_lines.csv'
    - Image (imshow) data are saved as: base_filename + '_image.csv'
    """
    ax = plt.gca()
    lines = ax.get_lines()
    images = ax.get_images()

    # Export line data
    if lines:
        line_filename = f"{base_filename}"
        x_data = lines[0].get_xdata()
        header = ['x']
        y_data_columns = []

        for idx, line in enumerate(lines):
            label = line.get_label()
            if label.startswith('_child'):
                label = f'y{idx + 1}'
            header.append(label)
            y_data_columns.append(line.get_ydata())

        with open(line_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(x_data)):
                row = [x_data[i]] + [y_col[i] for y_col in y_data_columns]
                writer.writerow(row)

        print(f"Line data exported to '{line_filename}'")

    else:
        print("No line plots found.")

    # Export image (imshow) data
    if images:
        for idx, img in enumerate(images):
            img_data = img.get_array().data
            img_filename = f"{base_filename}"
            np.savetxt(img_filename, img_data, delimiter=',')
            print(f"Image data exported to '{img_filename}'")
    else:
        print("No image data (imshow) found.")

ext = "pdf"

rolledMeanList = [] 
rolledMeanFlattendList = []    

num_bins_true = 100
num_bins_choosen = 100

edgespx = np.linspace(0, 2*np.pi, num_bins_choosen, endpoint=False)
edgespy = np.linspace(0, 2*np.pi, num_bins_true, endpoint=False)

xtick_vals = np.linspace(0, 100, 5)
xtickpi_vals = np.linspace(0, 2 * np.pi, 5)
xtick_labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

def createPlotsForDistortion(dMNormal, hill, appendix):
    
    #### FIG 11 ####
    plt.figure(figsize=(6, 5))
    im = plt.imshow(dMNormal, origin="lower",
                    cmap="viridis", aspect="auto")

    plt.colorbar(im, label=r"Distortion")
    plt.xlabel(r"Source direction $\theta_s$")
    plt.ylabel(r"Movement direction $\theta_m$")
    #plt.title(f"Normalized Distortion Function D($\\theta_y$, $\\theta_x$) for h={hill}")
    plt.xticks(xtick_vals, labels=xtick_labels)
    plt.yticks(xtick_vals, labels=xtick_labels)
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/distortion{appendix}_{hill}.{ext}")
    export_all_plot_data_to_csv(f"{BASE_PATH}/distortion{appendix}_{hill}.csv")
    plt.close()
    ### END ###

    ## shifted ##

    #### FIG 12 ####
    plt.figure(figsize=(6, 5))
    rolledList = []
    for x in range(num_bins_choosen):
        rolled = np.roll(dMNormal[:, x], -x)
        rolledList.append(rolled)
        plt.plot(edgespy, rolled)

    rolledStacked = np.stack(rolledList)
    rolledMean = np.mean(rolledStacked, axis=0)

    plt.plot(edgespy, rolledMean, linewidth=4, color="black", label="Mean")
    plt.xlabel(r"Shifted source direction $(\theta_m - \theta_s)+\pi$")
    plt.ylabel(r"Distortion")
    plt.xticks(xtickpi_vals, labels=xtick_labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/distortion2dshifted{appendix}_{hill}.{ext}")
    export_all_plot_data_to_csv(f"{BASE_PATH}/distortion2dshifted{appendix}_{hill}.csv")
    plt.close()
    ## END ##
    return rolledMean

def main(hill):
    filename = f"./data/hill_{hill}.csv"
    try:
        data = np.loadtxt(filename, delimiter=",")
    except Exception as e:
        print(f"Error during reading the file: {e}")
        exit(1)


    steps         = data[:, 0].astype(int)
    agent_ids     = data[:, 1].astype(int)
    chosen_theta  = data[:, 2]
    true_theta    = data[:, 3]
    old_direction = data[:, 4]

    true_theta_idx = data[:, 7].astype(np.int16)
    choosen_theta_idx = data[:,5].astype(np.int16)

    old_idx = np.roll(choosen_theta_idx, shift=1)
    old_idx[0] = 0

    # Bin-Indices berechnen
    assert ((0 <= true_theta) & (true_theta < 2 * np.pi)).all()
    assert ((0 <= chosen_theta) & (chosen_theta < 2 * np.pi)).all()
    assert ((0 <= true_theta_idx) & (true_theta_idx < num_bins_true)).all(), "true_theta_idx out of bounds"

    # Histogramm initialisieren
    hist2d = np.zeros((num_bins_true, num_bins_choosen), dtype=int) 

    hist2d += 1

    # Inkrementiere Histogramm für jedes Wertepaar
    for i, j in zip(true_theta_idx, choosen_theta_idx): 
        hist2d[i, j] += 1  # wird 


    pyx = hist2d / np.sum(hist2d)

    row_sums = hist2d.sum(axis=1, keepdims=True)
    col_sums = hist2d.sum(axis=0, keepdims=True)
    px_y = np.divide(hist2d, row_sums, where=row_sums != 0)
    py_x = np.divide(hist2d, col_sums, where=col_sums != 0)

    py = np.sum(pyx, axis=1)
    px = np.sum(pyx, axis=0)

    #### FIG 10 ####
    plt.figure(figsize=(6, 5))
    plt.imshow(py_x, origin='lower',
            aspect='auto', cmap='viridis')
    plt.colorbar(label=r"Decision strategy $P_{\Theta_m|\Theta_s}(\theta_m|\theta_s)$")
    plt.xlabel(r"Source direction $\theta_s$")
    plt.ylabel(r"Movement direction $\theta_m$")
    plt.xticks(xtick_vals, labels=xtick_labels)
    plt.yticks(xtick_vals, labels=xtick_labels)
    #plt.title(r"Decision strategy $P(\mathrm{chosen} \mid \mathrm{true})$")
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/decisionstrategy_{hill}.{ext}")
    export_all_plot_data_to_csv(f"{BASE_PATH}/decisionstrategy_{hill}.csv")
    plt.close()
    ### END ###
    assert not np.isnan(px_y).any()

    ## clac distortion matrix

    assert (py != 0).all()
    assert (px_y != 0).all()


    distortionMatrix = py_x / py[:, np.newaxis]
    distortionMatrix = distortionMatrix
    distortionMatrix = -np.log(distortionMatrix)
    distortionMatrix -= np.min(distortionMatrix) 

    rolledMean = createPlotsForDistortion(distortionMatrix, hill, "")
    rolledMeanList.append((rolledMean, hill))


if __name__=='__main__':
    hillcoef = [1, 3, 5, 7, 9, 15]

    for h in hillcoef:
        main(h)

    def createTotalPlot(meanList, name):
        #### FIG 13 ####
        for rolled, hill in meanList:
            plt.plot(edgespy, rolled, label=rf"$h={hill}$")

        plt.xlabel(r"Shifted source direction $(\theta_m - \theta_s)+\pi$")
        plt.ylabel(r"Distortion")
        plt.legend()
        plt.xticks(xtickpi_vals, labels=xtick_labels)
        plt.tight_layout()
        plt.savefig(f"{BASE_PATH}/totalcosineabsolut{name}.{ext}")
        export_all_plot_data_to_csv(f"{BASE_PATH}/totalcosineabsolut{name}.csv")
        plt.close()
        ### END ###

        cosdist = 0.5 * (1 - np.cos(edgespy))

        #### Table with mse error to cosine distortion function ####
        with open(f"{BASE_PATH}/hillerror.tex", "w") as f:
            f.write("\\begin{tabular}{|c|c|} \\hline \n \\textbf{Hill coefficent} & \\textbf{MSE} \\\\ \\hline \n")

            for rolled, hill in meanList:
                a = np.dot(rolled, cosdist) / np.dot(rolled, rolled)
                plt.plot(edgespy, a * rolled, label=rf"$h={hill}$; $a={round(a, 2)}$")
                
                err = np.mean((cosdist - a * rolled) ** 2)
                print("hill=", hill, "Error = ", err)
                f.write(f"{hill} & {err:.5f} \\\\ \\hline \n")
            f.write("\\end{tabular}\n")

        plt.figure(figsize=(6, 5))
        plt.plot(edgespy, cosdist, label="Cosine distortion", linewidth=4, color="black")
        plt.xlabel(r"Shifted source direction $(\theta_m - \theta_s)+\pi$")
        plt.ylabel(r"Distortion")
        plt.legend()
        plt.xticks(xtickpi_vals, labels=xtick_labels)
        plt.tight_layout()
        plt.savefig(f"{BASE_PATH}/totalcosine{name}.{ext}")
        export_all_plot_data_to_csv(f"{BASE_PATH}/totalcosine{name}.csv")
        plt.close()

    createTotalPlot(rolledMeanList, "")
