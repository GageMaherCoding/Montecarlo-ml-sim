# src/option_analysis.py

import numpy as np

def evaluate_options(paths: np.ndarray, strike_prices: list, horizon_days: int,
                     risk_free_rate: float = 0.05, risk_tolerance: str = "moderate"):
    """
    Evaluate call and put options from Monte Carlo paths.

    Args:
        paths: ndarray of shape (n_steps, n_paths) or (n_paths, n_steps)
        strike_prices: list of strikes to evaluate
        horizon_days: simulation horizon in days
        risk_free_rate: annualized risk-free rate
        risk_tolerance: "aggressive", "moderate", or "conservative"

    Returns:
        List of dicts with option metrics for each strike, best call, best put
    """
    # Ensure paths are (n_paths, n_steps)
    if paths.shape[0] < paths.shape[1]:
        paths = paths.T

    T = horizon_days / 252  # convert days to years
    results = []

    for K in strike_prices:
        terminal_prices = paths[:, -1]

        # Call option
        call_payoff = np.maximum(terminal_prices - K, 0)
        call_price = np.exp(-risk_free_rate * T) * np.mean(call_payoff)
        call_pop = np.mean(call_payoff > 0)

        # Put option
        put_payoff = np.maximum(K - terminal_prices, 0)
        put_price = np.exp(-risk_free_rate * T) * np.mean(put_payoff)
        put_pop = np.mean(put_payoff > 0)

        # Adjust based on risk tolerance
        if risk_tolerance == "aggressive":
            call_score = call_price * (1 - call_pop)
            put_score = put_price * (1 - put_pop)
        elif risk_tolerance == "conservative":
            call_score = call_price * call_pop
            put_score = put_price * put_pop
        else:  # moderate
            call_score = call_price * call_pop
            put_score = put_price * put_pop

        results.append({
            "strike": K,
            "call_price": call_price,
            "call_pop": call_pop,
            "call_score": call_score,
            "put_price": put_price,
            "put_pop": put_pop,
            "put_score": put_score
        })

    best_call = max(results, key=lambda x: x["call_score"])
    best_put = max(results, key=lambda x: x["put_score"])

    return results, best_call, best_put


# Example usage
if __name__ == "__main__":
    # Replace this with your Monte Carlo paths
    # paths = np.load("monte_carlo_paths.npy")  
    strike_prices = [660, 664, 668, 670, 675]
    horizon_days = 30
    risk_tolerance = "moderate"

    # results, best_call, best_put = evaluate_options(paths, strike_prices, horizon_days, risk_tolerance=risk_tolerance)
    # print("Best Call:", best_call)
    # print("Best Put:", best_put)
