# src/simulate.py

import numpy as np
import yfinance as yf

# Local imports (all in the same folder: src/)
from ml_model import MuSigmaPredictor
from option_analysis import evaluate_options
from option_plot import plot_option_payoff

def run_simulation_for_ticker(
    ticker="SPY",
    period="1y",
    horizon_days=30,
    n_paths=10000,
    seed=42,
    use_ml=True
):
    np.random.seed(seed)

    # Download recent prices
    df_recent = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    close = df_recent['Close'].dropna()
    S0 = close.iloc[-1].item()

    # ML predictor
    if use_ml:
        predictor = MuSigmaPredictor(window=5)
        predictor.train(ticker=ticker, period="5y")
        mu, sigma = predictor.predict(close)
    else:
        log_returns = np.log(close / close.shift(1)).dropna()
        mu = log_returns.mean()
        sigma = log_returns.std(ddof=1)

    # Monte Carlo simulation
    dt = 1/252
    paths = np.zeros((horizon_days, n_paths))
    paths[0, :] = S0

    for t in range(1, horizon_days):
        Z = np.random.standard_normal(n_paths)
        paths[t, :] = paths[t-1, :] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

    print(f"Simulation metadata:\nticker: {ticker}\nS0: {S0}\nhorizon_days: {horizon_days}\nn_paths: {n_paths}\nseed: {seed}")
    print(f"Predicted mu: {mu:.5f}, sigma: {sigma:.5f}")

    return paths, S0, (mu, sigma)

def generate_strike_prices(S0, num_strikes=5, step=2):
    """Generate a list of strike prices around current price S0."""
    center = int(round(S0))
    half = num_strikes // 2
    strikes = [center + step*(i-half) for i in range(num_strikes)]
    return strikes

if __name__ == "__main__":
    # Run Monte Carlo simulation
    paths, S0, (mu, sigma) = run_simulation_for_ticker(use_ml=True)
    horizon_days = 30
    risk_tolerance = "moderate"

    # Automatically generate strike prices
    strike_prices = generate_strike_prices(S0, num_strikes=5, step=2)

    # Evaluate options
    results, best_call, best_put = evaluate_options(paths, strike_prices, horizon_days, risk_tolerance=risk_tolerance)

    print("\nOptions Analysis Results:")
    for r in results:
        print(f"Strike {r['strike']}: Call ${r['call_price']:.2f} (POP {r['call_pop']*100:.1f}%), "
              f"Put ${r['put_price']:.2f} (POP {r['put_pop']*100:.1f}%)")

    print("\nBest Call Option:", best_call)
    print("Best Put Option:", best_put)

    # Plot payoff vs strike
    plot_option_payoff(results, best_call, best_put)
