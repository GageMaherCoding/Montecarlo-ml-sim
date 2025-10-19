import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Tuple


class QuantVolatilityPredictor:
    """
    Quant-grade predictor for drift (mu) and volatility (sigma) for Monte Carlo simulations.

    - sigma is estimated via EWMA of log returns (annualized)
    - mu is blended between long-term mean and recent EWMA
    """

    def __init__(self, ewma_span: int = 60, recent_weight: float = 0.7):
        """
        Parameters:
        ewma_span: span for EWMA volatility
        recent_weight: weight given to recent EWMA drift vs long-term mean
        """
        self.ewma_span = ewma_span
        self.recent_weight = recent_weight
        self.sigma_series: Optional[pd.Series] = None
        self.fitted = False

    @staticmethod
    def _log_returns(close: pd.Series) -> pd.Series:
        """Compute log returns from price series"""
        close = close.dropna()
        return np.log(close / close.shift(1)).dropna()

    def fit(self, close: pd.Series):
        """
        Fit the EWMA volatility series.
        """
        log_returns = self._log_returns(close)

        # EWMA volatility, annualized
        self.sigma_series = log_returns.ewm(span=self.ewma_span, adjust=False).std()
        self.sigma_series *= np.sqrt(252)

        self.fitted = True

    def predict(self, close: pd.Series) -> Tuple[float, float]:
        """
        Return (mu, sigma) for next-step Monte Carlo simulation.
        mu: blended drift (annualized)
        sigma: annualized volatility

        Uses:
        - long-term mean of log returns
        - recent EWMA of log returns
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit(close) first.")

        log_returns = self._log_returns(close)
        long_term_mu = float(log_returns.mean().item() * 252)  # annualized

        # Recent EWMA drift
        recent_mu_ewma = float(log_returns.ewm(span=self.ewma_span, adjust=False).mean().iloc[-1].item() * 252)

        mu = self.recent_weight * recent_mu_ewma + (1 - self.recent_weight) * long_term_mu

        # Sigma from EWMA series
        sigma = float(self.sigma_series.dropna().iloc[-1].item())

        return mu, sigma

    def fit_from_ticker(self, ticker: str, period: str = "10y"):
        """Download prices and fit EWMA volatility"""
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        close = df["Close"].dropna()
        self.fit(close)
