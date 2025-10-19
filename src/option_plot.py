# src/option_plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use Seaborn darkgrid style for professional visuals
sns.set_theme(style="darkgrid", palette="deep")


def plot_paths(paths: np.ndarray, n_display: int = 50, save_path: str = None):
    """Plot first n_display Monte Carlo paths."""
    plt.figure(figsize=(12, 6))
    for i in range(min(n_display, paths.shape[0])):
        plt.plot(paths[i], alpha=0.7)
    plt.title("Monte Carlo Simulated Paths", fontsize=16)
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_terminal_distribution(paths: np.ndarray, bins: int = 50, save_path: str = None):
    """Histogram of terminal prices from Monte Carlo paths."""
    terminal_prices = paths[:, -1]
    plt.figure(figsize=(12, 6))
    plt.hist(terminal_prices, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Terminal Price Distribution", fontsize=16)
    plt.xlabel("Price", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(np.mean(terminal_prices), color='red', linestyle='--', label='Mean')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_option_payoff(
    results: list,
    best_call: dict,
    best_put: dict,
    return_curves: dict = None,
    save_path: str = None
):
    """
    Plot expected option payoff vs strike prices.
    Optionally overlay expected return curves.
    """
    strikes = [r["strike"] for r in results]
    call_prices = [r["call_price"] for r in results]
    put_prices = [r["put_price"] for r in results]

    plt.figure(figsize=(12, 6))
    plt.plot(strikes, call_prices, marker='o', label="Call Expected Payoff", color="blue")
    plt.plot(strikes, put_prices, marker='o', label="Put Expected Payoff", color="orange")
    plt.scatter(best_call["strike"], best_call["call_price"], color="green", s=150, label="Best Call")
    plt.scatter(best_put["strike"], best_put["put_price"], color="red", s=150, label="Best Put")

    # Overlay risk-adjusted expected return curves if provided
    if return_curves is not None:
        plt.plot(strikes, return_curves["call_curve"], linestyle='--', color="blue", alpha=0.5, label="Call Risk-Adjusted")
        plt.plot(strikes, return_curves["put_curve"], linestyle='--', color="orange", alpha=0.5, label="Put Risk-Adjusted")

    plt.title("Expected Option Payoff vs Strike Price", fontsize=16)
    plt.xlabel("Strike Price", fontsize=12)
    plt.ylabel("Expected Payoff ($)", fontsize=12)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
