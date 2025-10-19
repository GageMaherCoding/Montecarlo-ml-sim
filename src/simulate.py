import numpy as np
import yfinance as yf
from typing import Tuple, List

from ml_model import QuantVolatilityPredictor
from option_analysis import evaluate_options
from option_plot import plot_option_payoff, plot_paths, plot_terminal_distribution


def run_simulation_for_ticker(
    ticker: str = "SPY",
    period: str = "10y",
    horizon_days: int = 252,
    n_paths: int = 10000,
    seed: int = 42,
    use_quant: bool = True,
    ewma_span: int = 60,
    recent_weight: float = 0.7,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Run vectorized Monte Carlo GBM simulation using quant-grade drift and volatility.

    Returns:
      - paths: ndarray shape (n_paths, n_steps+1)
      - S0: float
      - (mu, sigma): annualized
    """
    np.random.seed(seed)

    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    close = df["Close"].dropna()
    if close.empty:
        raise ValueError("Failed to download close prices for ticker: " + ticker)

    S0 = float(close.iloc[-1].item())

    if use_quant:
        predictor = QuantVolatilityPredictor(ewma_span=ewma_span, recent_weight=recent_weight)
        predictor.fit(close)
        mu, sigma = predictor.predict(close)
    else:
        # Simple historical estimates
        log_returns = np.log(close / close.shift(1)).dropna()
        mu = float(log_returns.mean().item() * 252)
        sigma = float(log_returns.std(ddof=1).item() * np.sqrt(252))

    dt = 1 / 252
    n_steps = int(horizon_days)

    Z = np.random.standard_normal(size=(n_steps, n_paths))
    increments = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.cumprod(increments, axis=0).T

    print("Simulation metadata:")
    print(f" ticker: {ticker}, S0: {S0:.4f}, horizon_days: {horizon_days}, n_paths: {n_paths}, seed: {seed}")
    print(f" mu (annualized): {mu:.6f}, sigma (annualized): {sigma:.6f}")

    return paths, S0, (mu, sigma)


def generate_strike_prices(S0: float, num_strikes: int = 7, step: float = 2) -> List[float]:
    """Generate evenly spaced strike prices around current price"""
    center = S0
    half = num_strikes // 2
    return [center + step * (i - half) for i in range(num_strikes)]


if __name__ == "__main__":
    horizon_days = 30
    n_paths = 10000

    # Monte Carlo simulation
    paths, S0, (mu, sigma) = run_simulation_for_ticker(
        horizon_days=horizon_days,
        n_paths=n_paths,
        use_quant=True,
        ewma_span=60,
        recent_weight=0.7
    )

    strike_prices = generate_strike_prices(S0, num_strikes=7, step=2)

    # Evaluate options
    results, best_call, best_put = evaluate_options(paths, strike_prices, horizon_days, risk_tolerance="moderate")

    # Print textual results
    print("\nOptions Analysis Results:")
    for r in results:
        print(f"Strike {r['strike']:.2f}: Call ${r['call_price']:.2f} (POP {r['call_pop']*100:.1f}%), "
              f"Put ${r['put_price']:.2f} (POP {r['put_pop']*100:.1f}%)")

    print("\nBest Call:", best_call)
    print("Best Put:", best_put)

    # Plotting
    plot_paths(paths, n_display=50, save_path="monte_carlo_paths.png")
    plot_terminal_distribution(paths, bins=50, save_path="terminal_distribution.png")
    plot_option_payoff(results, best_call, best_put, risk_tolerance="moderate", save_path="option_payoff.png")
