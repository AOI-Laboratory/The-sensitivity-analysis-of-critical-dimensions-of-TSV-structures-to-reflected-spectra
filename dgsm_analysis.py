import argparse
from os.path import abspath

import numpy as np
import pandas as pd
from SALib.analyze import dgsm
from matplotlib import pyplot as plt

CENTRAL_ANGLE = 0
problem = {
    'num_vars': 6,
    'names': ['retch', 'rtop', 'rbot', 'n', 'height', 'b'],
    'bounds': [[0.02, 0.1], [0.27, 0.33], [0.2, 0.27], [100, 600], [2.7, 3.3], [0.05, 0.1]]
}


def main(ocd_path, spectrum_path):
    # Load data from the ocd_path file with ',' as the delimiter and store it in X.
    X = np.loadtxt(ocd_path, delimiter=',')

    # Read the spectrum data from the spectrum_path using pandas.
    spectrum = pd.read_csv(spectrum_path)

    # Extract the frequency values from the first row of the spectrum data.
    freqs = spectrum.head().columns[:]
    freq_size = len(freqs)

    # Initialize arrays to store various analysis results for each frequency.
    vi = np.zeros(shape=(freq_size, problem['num_vars']), dtype=float)
    vi_std = np.zeros(shape=(freq_size, problem['num_vars']), dtype=float)
    dgsm_value = np.zeros(shape=(freq_size, problem['num_vars']), dtype=float)
    dgsm_conf = np.zeros(shape=(freq_size, problem['num_vars']), dtype=float)

    start_pos = problem['num_vars']

    # Iterate over each frequency in the spectrum data.
    for i in range(start_pos, start_pos + freq_size):
        # Extract the Y values for the current frequency.
        Y = spectrum.iloc[:, i:i + 1].values.flatten()

        # Analyze the sensitivity using dgsm.analyze function and store the results.
        Si = dgsm.analyze(problem, X, Y, print_to_console=False)
        vi[i - start_pos] = Si['vi']
        vi_std[i - start_pos] = Si['vi_std']
        dgsm_value[i - start_pos] = Si['dgsm']
        dgsm_conf[i - start_pos] = Si['dgsm_conf']

    # Iterate over the problem variables.
    for i in range(problem['num_vars']):
        names = problem['names'][i]

        # Create subplots for VI, VI std, DGSM, and DGSM conf.
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(19, 10))
        axs = axs.flatten()

        # Set the title for the current subplot.
        fig.suptitle('{} , center angle={}'.format(names, CENTRAL_ANGLE), fontsize=16)

        # Plot VI values.
        axs[0].plot(freqs, vi[:, i])
        axs[0].set_ylabel('VI')

        # Plot VI std values.
        axs[1].plot(freqs, vi_std[:, i])
        axs[1].set_ylabel('VI std')

        # Plot DGSM values.
        axs[2].plot(freqs, dgsm_value[:, i])
        axs[2].set_ylabel('DGSM')

        # Plot DGSM conf values.
        axs[3].plot(freqs, dgsm_conf[:, i])
        axs[3].set_ylabel('DGSM conf')

        # Display the plots.
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A program that performs DGSM analysis')

    # add arguments
    parser.add_argument('-o', '--ocd',
                        help='The csv file contains OCD data',
                        required=True)

    parser.add_argument('-s', '--spectrum',
                        help='The csv file contains spectrum data',
                        required=True)

    args = parser.parse_args()
    print('OCD CSV file = {}'.format(abspath(args.ocd)))
    print('Spectrum CSV file = {}'.format(abspath(args.spectrum)))
    main(abspath(args.ocd), abspath(args.spectrum))
