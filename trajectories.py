"""

Created on November 10, 2022

Author: isegarra@comillas.edu

"""

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib.cm     as cm

from matplotlib   import rcParams
from scipy.linalg import solve

def simulate_arithmetic_BM(S0: float = 0, n: float =100, time_steps: float = 10) -> np.ndarray:
    """
    Returns a np.ndarray with shape (time_steps + 1, n), each column is a trajectory of the BM

    Arguments:
        S0:         initial value of the underlying
        n:          number of trajectories simulated
        time_steps: number of time steps during the simulation

    Output:
        trajectories: matrix with the trajectories and each value at each time step
    """
    sd_step = 1/np.sqrt(time_steps)

    trajectories    = sd_step * np.random.randn(time_steps + 1 , n)
    trajectories[0] = S0

    trajectories = np.cumsum(trajectories, axis = 0)

    return np.transpose(trajectories)

def fancy_plot(ts: np.ndarray , n: float = 100, time_steps: float = 10, title: str = 'A very cool plot',
                     display: int = 100, xlabel: str = 'time', ylabel: str = 'value',
                     savefig: bool = False, figdir: str = 'fancy_plot.pdf') -> None:
    """
    Given trajectories it plots the trajectories and the distribution at
    the ending point

    Arguments:
        ts:         Trajectories matrix, 1 row per simulation
        n:          number of trajectories simulated
        time_steps: number of time steps during the simulation (ignoring the first)
        title:      title of the plot
        display:    number of trajectories shown in the plot
        xlabel:     xlabel of the figure
        ylabel:     ylabel of the figure
        savefig:    True or False to save fig or not
        figdir:     Name of the saved figure

    Output:
        None

    Example:

        time_steps = 100
        num_sims   = 1000
        trajs = simulate_arithmetic_BM(n = num_sims, time_steps = time_steps)
        fancy_plot(trajs, n = num_sims, time_steps = time_steps)

    """
    # changing index to its xlabel
    ts = ts.transpose()

    index  = np.arange(0,time_steps+1)
    df     = pd.DataFrame(ts, index=index)

    # z is the outcome of the trajectories, it defines the distribution.
    z      = df.iloc[-1]
    color  = plt.cm.get_cmap('turbo')((z - z.min()) / (z.max() - z.min())) # fancy color map for the trajectories :)

    # Displaying few trajectories

    df = df.iloc[:,:display]

    # layout
    fig = plt.figure()
    gs  = fig.add_gridspec(1, 2, wspace=0, width_ratios=[9, 2])
    ax  = gs.subplots(sharey=True)

    # line chart

    df.plot( figsize=(8,6), title=(title),grid = False, legend=False, ax=ax[0],
            color=color)

    # histogram
    n_bins = 20 # number of bins in the histogram
    cnt, bins, patches = ax[1].hist(
        z, np.linspace(z.min(), z.max(), n_bins),
        ec='k', orientation='horizontal', density = True)

    colors = plt.cm.get_cmap('turbo')((bins - z.min()) / (z.max() - z.min())) # fancy color map for the histogram :)
    ax[1].set_xticks([])
    for i, p in enumerate(patches):
        p.set_color(colors[i])

    ax[0].set_xlabel(xlabel, fontsize = 14)
    ax[0].set_ylabel(ylabel, fontsize = 14)
#     ax2 = ax[1].twinx()
#     ax2.set_ylabel(ylabel_right, fontsize = 14, rotation = 270)
#     ax2.yaxis.set_label_coords(1.5,0.5)
#     ax2.set_xticks([])

    plt.tight_layout()

    if savefig:
        plt.savefig(figdir)
    plt.show()

if __name__ == "__main__":
    time_steps = 100
    num_sims   = 1000
    trajs = simulate_arithmetic_BM(n = num_sims, time_steps = time_steps)
    fancy_plot(trajs, n = num_sims, time_steps = time_steps)
