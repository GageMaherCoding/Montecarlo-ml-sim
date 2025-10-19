# src/option_analysis.py
import numpy as np
from typing import List, Tuple, Dict

def evaluate_options(paths: np.ndarray, strikes: List[float], horizon_days: int,
                     risk_tolerance: str = "moderate") -> Tuple[List[Dict], Dict, Dict]:
    """
    Evaluate European Call and Put options on given Monte Carlo paths.

    Returns:
        results: list of dicts for each strike with call/put prices and POP
        best_call: dict for best call option (risk-adjusted score)
        best_put: dict for best put option (risk-adjusted score)
    """
    dt = 1.0 / 252.0
    discount_factor = np.exp(-0.0 * horizon_days * dt)  # Zero risk-free for simplicity

    terminal_prices = paths[:, -1]
    results = []

    for K in strikes:
        call_payoffs = np.maximum(terminal_prices - K, 0) * discount_factor
        put_payoffs = np.maximum(K - terminal_prices, 0) * discount_factor

        call_price = float(np.mean(call_payoffs))
        put_price = float(np.mean(put_payoffs))

        call_pop = float(np.mean(call_payoffs > 0))  # Probability of profit
        put_pop = float(np.mean(put_payoffs > 0))

        # Risk-adjusted "score" (expected payoff * POP)
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

    # Best options (maximize risk-adjusted score)
    best_call = max(results, key=lambda r: r["call_score"])
    best_put = max(results, key=lambda r: r["put_score"])

    return results, best_call, best_put
