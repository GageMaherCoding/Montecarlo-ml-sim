"""
Professional plotting functions for Monte Carlo GBM simulations.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-darkgrid")  # consistent professional style

def plot_paths(paths: np.ndarray, n_display: int = 50, save_path=None) -> None:
    """
    Plot the first n_display Monte Carlo paths.

    Args:
        paths: ndarray of shape (n_steps, n_paths) or (n_paths, n_steps)
        n_display: number of paths to display
        save_path: if provided, saves figure to this path
    """
    if paths.shape[0] < paths.shape[1]:
        paths = paths.T  # ensure shape (n_paths, n_steps)
    
    plt.figure(figsize=(10,6))
    for i in range(min(n_display, paths.shape[0])):
        plt.plot(paths[i], alpha=0.7)
    
    plt.title(f"Monte Carlo Simulation Paths ({n_display} paths)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()


def plot_terminal_distribution(paths: np.ndarray, bins: int = 50, save_path=None) -> None:
    """
    Plot histogram of terminal prices from Monte Carlo paths.

    Args:
        paths: ndarray of shape (n_paths, n_steps) or (n_steps, n_paths)
        bins: number of histogram bins
        save_path: if provided, saves figure to this path
    """
    if paths.shape[0] < paths.shape[1]:
        paths = paths.T

    terminal_prices = paths[:, -1]
    plt.figure(figsize=(10,6))
    plt.hist(terminal_prices, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title("Terminal Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()
