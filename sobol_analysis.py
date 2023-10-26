import argparse
from os.path import abspath

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

CENTRAL_ANGLE = 0
problem = {
    'num_vars': 6,
    'names': ['retch', 'rtop', 'rbot', 'n', 'height', 'b'],
    'bounds': [[0.02, 0.1], [0.27, 0.33], [0.2, 0.27], [100, 600], [2.7, 3.3], [0.05, 0.1]]
}


def main(spectrum_path):
    # Read the spectrum data from the spectrum_path using pandas.
    spectrum = pd.read_csv(spectrum_path)

    # Extract the frequency values from the first row of the spectrum data.
    freqs = spectrum.head().columns[:]
    freq_size = len(freqs)

    # Initialize arrays to store results
    s1 = np.zeros(shape=(freq_size, problem['num_vars']), dtype=float)
    s1_conf = np.zeros(shape=(freq_size, problem['num_vars']), dtype=float)
    st = np.zeros(shape=(freq_size, problem['num_vars']), dtype=float)
    st_conf = np.zeros(shape=(freq_size, problem['num_vars']), dtype=float)

    start_pos = problem['num_vars']

    # Analyze the data for each frequency value
    for i in range(start_pos, start_pos + freq_size):
        Y = spectrum.iloc[:, i:i + 1].values.flatten()
        Si = sobol.analyze(problem, Y, calc_second_order=True)
        s1[i - start_pos] = Si['S1']
        s1_conf[i - start_pos] = Si['S1_conf']
        st[i - start_pos] = Si['ST']
        st_conf[i - start_pos] = Si['ST_conf']

    # Generate plots for each variable
    for i in range(problem['num_vars']):
        names = problem['names'][i]

        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(19, 10))
        axs = axs.flatten()

        # Set the title of the figure
        fig.suptitle('{} , center angle={}'.format(names, CENTRAL_ANGLE), fontsize=16)

        # Plot the data and configure axes
        axs[0].plot(freqs, s1[:, i])
        axs[0].set_ylabel('wavelength')
        axs[0].set_ylabel('S1')
        axs[0].xaxis.set_major_locator(MaxNLocator(prune='both'))

        axs[1].plot(freqs, s1_conf[:, i])
        axs[1].set_ylabel('wavelength')
        axs[1].set_ylabel('S1 conf')
        axs[1].xaxis.set_major_locator(MaxNLocator(prune='both'))

        axs[2].plot(freqs, st[:, i])
        axs[2].set_ylabel('wavelength')
        axs[2].set_ylabel('ST')
        axs[2].xaxis.set_major_locator(MaxNLocator(prune='both'))

        axs[3].plot(freqs, st_conf[:, i])
        axs[3].set_ylabel('wavelength')
        axs[3].set_ylabel('ST_conf')
        axs[3].xaxis.set_major_locator(MaxNLocator(prune='both'))

        # Show the plots
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A program that performs Sobol analysis')

    # add arguments
    parser.add_argument('-s', '--spectrum',
                        help='The csv file contains spectrum data',
                        required=True)

    args = parser.parse_args()
    print('Spectrum CSV file = {}'.format(abspath(args.spectrum)))
    main(abspath(args.spectrum))
