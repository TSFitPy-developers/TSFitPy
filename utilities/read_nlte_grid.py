# script to read the binary NLTE grid at a pointer (specified in the auxiliary file)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.use('MacOSX')

def read_binary_grid(grid_file, pointer=1):
    """
    # author: @Katerina Magg

    Reads a record at specified position from the binary NLTE grid
    (grid of departure coefficients)

    Parameters
    ----------
    grid_file : str
        path to the binary NLTE grid
    pointer : int
        bitwise start of the record as read from the auxiliarly file

    Returns
    -------
    ndep : int
        number of depth points in the model atmosphere used to solve for NLTE
    nk : int
        number of energy levels in the model atom used to solved for NLTE
    depart : array
        NLTE departure coefficients of shape (ndep, nk)
    tau : array
        depth scale (e.g. TAU500nm) in the model atmosphere
        used to solve for NLTE of shape (ndep)
    """
    with open(grid_file, 'rb') as f:
        # -1 since Python stars at 0
        pointer = pointer - 1

        f.seek(pointer)
        atmosStr = f.readline(500).decode('utf-8', 'ignore').strip()
        ndep = int.from_bytes(f.read(4), byteorder='little')
        nk = int.from_bytes(f.read(4), byteorder='little')
        tau  = np.log10(np.fromfile(f, count = ndep, dtype='f8'))
        depart = np.fromfile(f, dtype='f8', count=ndep*nk).reshape(nk, ndep)

    return ndep, nk, depart, tau, atmosStr


def plot_departure_coefficients(depart, tau, atmosStr, levels_to_plot):
    # plot departure coefficients

    # if levels_to_plot is int, convert to list
    if isinstance(levels_to_plot, int):
        levels_to_plot = [levels_to_plot]

    fig, ax = plt.subplots()
    for level in levels_to_plot:
        ax.plot(tau, depart[level - 1], label=f"level {level - 1}")
    ax.set_xlabel(r"$\log_{10}(\tau)$")
    ax.set_ylabel(r"$b_{\rm NLTE}$")
    ax.set_title(f"Departure coefficients for {atmosStr}")
    ax.legend()
    plt.show()



if __name__ == '__main__':
    pointer = 1001
    grid_file = 'binary_grid.bin'
    ndep, nk, depart, tau, atmosStr = read_binary_grid(grid_file, pointer=pointer)

    print(f"atmosStr={atmosStr}")  # atmosStr is the model atmosphere used to solve for NLTE, can compare to the model atmosphere in the auxiliary file
    print(f"ndep={ndep} nk={nk}")  # ndep are number of depth points, nk are number of energy levels
    print(f"tau={(tau)}")          # converted to log10(tau)
    print(f"depart={depart}")      # departure coefficients as a function of depth and energy level, 2D array.
    # depart[level] is the departure coefficients for a given energy level, at all depths

    # levels_to_plot can be a list of levels to plot, or a single level
    # levels are indexed from 1
    plot_departure_coefficients(depart, tau, atmosStr, levels_to_plot=[1, 2])