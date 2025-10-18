# src/plot.py
"""
Plotting functions for Monte Carlo GBM simulation.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_paths(paths: np.ndarray, n_display: int = 50) -> None:
    """
    Plot the first n_display Monte Carlo paths.

    Args:
        paths: ndarray of shape (n_paths, n_steps+1)
        n_display: number of paths to display
    """
    plt.figure(figsize=(10,6))
    for i in range(min(n_display, paths.shape[0])):
        plt.plot(paths[i], alpha=0.7)
    plt.title("Monte Carlo Paths")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


def plot_terminal_distribution(paths: np.ndarray, bins: int = 50) -> None:
    """
    Plot histogram of terminal prices from Monte Carlo paths.

    Args:
        paths: ndarray of shape (n_paths, n_steps+1)
        bins: number of histogram bins
    """
    terminal_prices = paths[:, -1]
    plt.figure(figsize=(10,6))
    plt.hist(terminal_prices, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Terminal Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
