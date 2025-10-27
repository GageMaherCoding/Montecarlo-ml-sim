# Monte Carlo Option Analysis Engine

This project provides a full quantitative Monte Carlo simulation pipeline for pricing European call and put options using a GBM (Geometric Brownian Motion) process. It includes:


* **Monte Carlo price path generation**
* **European option payoff evaluation and ranking**
* **Professional plotting utilities** for diagnostics

---

## Features

### 1) QuantVolatilityPredictor (`ml_model.py`)

Uses log returns to fit annualized drift (mu) and volatility (sigma):

* `sigma` via EWMA of returns (annualized)
* `mu` as a weighted blend of long‐term mean vs recent EWMA

### 2) Monte Carlo Simulation (`run_simulation_for_ticker`)

* Fetches historical prices from Yahoo Finance (`yfinance`)
* Simulates GBM over ( N ) paths and horizon
* Returns full path matrix for downstream pricing

### 3) Option Evaluation (`option_analysis.py`)

For a set of strikes:

* Computes expected call & put prices from terminal payoffs
* Computes POP (Probability of Profit)
* Computes risk‐adjusted score = expected payoff × POP
* Returns ranked best call & put

### 4) Plotting (`option_plot.py`)

* Path trace plot
* Terminal distribution
* Expected payoff vs strike + best option markers

---


The script will:

1. Download price history
2. Fit mu, sigma
3. Run Monte Carlo
4. Evaluate strikes
5. Print results & save plots

---


## 📌 Dependencies

```
pandas
numpy
yfinance
matplotlib
seaborn
```


## ✅Outputs

* Printed simulation metadata (mu, sigma, S0)
* Ranked option candidates
* Saved figures:

  * `monte_carlo_paths.png`
  * `terminal_distribution.png`
  * `option_payoff.png`

---

## Notes & Assumptions

* Uses GBM — ignores jumps, skew, regime shifts
* Risk‐free rate assumed ~0 in current implementation
* Path simulation assumes daily steps (252/year)



## Future Improvements

* Add risk‐free discounting
* Add confidence intervals & risk metrics
* Add implied vol surface support
* Add Greeks estimation
* Add portfolio/strategy layer


## Acknowledgements

Special thanks to Kaya Li for co‑collaboration. She helped with live coding support in VS Code, utility review, and mathematical guidance.
End of README.
