# src/option_plot.py

import matplotlib.pyplot as plt

def plot_option_payoff(results: list, best_call: dict, best_put: dict):
    """
    Plot expected call/put payoff vs strike prices, highlighting best options.

    Args:
        results: list of dicts returned from evaluate_options()
        best_call: dict of best call option
        best_put: dict of best put option
    """
    strikes = [r["strike"] for r in results]
    call_prices = [r["call_price"] for r in results]
    put_prices = [r["put_price"] for r in results]

    plt.figure(figsize=(10,6))
    plt.plot(strikes, call_prices, marker='o', label="Call Price", color="blue")
    plt.plot(strikes, put_prices, marker='o', label="Put Price", color="orange")

    # Highlight best options
    plt.scatter(best_call["strike"], best_call["call_price"], color="green", s=150, label="Best Call")
    plt.scatter(best_put["strike"], best_put["put_price"], color="red", s=150, label="Best Put")

    plt.title("Expected Option Payoff vs Strike Price")
    plt.xlabel("Strike Price")
    plt.ylabel("Expected Discounted Payoff ($)")
    plt.grid(True)
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Import evaluate_options and paths from your simulation
    # from option_analysis import evaluate_options
    # results, best_call, best_put = evaluate_options(paths, strike_prices, horizon_days)
    # plot_option_payoff(results, best_call, best_put)
    pass
