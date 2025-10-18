# src/ml_model.py

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

class MuSigmaPredictor:
    def __init__(self, window=20):
        self.window = window
        self.mu_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def _compute_features(self, close: pd.Series):
        """Compute rolling mean and std as features for the last day."""
        if len(close) < self.window:
            raise ValueError(f"Close series too short for window={self.window}")
        ma = float(close.rolling(window=self.window).mean().iloc[-1])
        std = float(close.rolling(window=self.window).std().iloc[-1])
        features = np.array([[ma, std]])  # shape (1,2) for sklearn
        return features

    def train(self, ticker: str, period="5y"):
        """Train mu model using historical stock data with log returns."""
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        close = df["Close"].dropna()

        if len(close) < self.window + 1:
            raise ValueError(f"Not enough data to compute features (need at least {self.window + 1} days).")

        # Compute log returns
        log_returns = np.log(close / close.shift(1)).dropna()

        # Compute rolling features
        ma_series = close.rolling(window=self.window).mean()
        std_series = close.rolling(window=self.window).std()

        # Align features with log returns
        aligned_index = ma_series.index[self.window - 1:]
        ma_aligned = ma_series.loc[aligned_index]
        std_aligned = std_series.loc[aligned_index]
        returns_aligned = log_returns.loc[aligned_index]

        # Drop NaNs
        valid_mask = (~ma_aligned.isna()) & (~std_aligned.isna()) & (~returns_aligned.isna())
        ma_final = ma_aligned[valid_mask]
        std_final = std_aligned[valid_mask]
        returns_final = returns_aligned[valid_mask]

        X = np.column_stack((ma_final.values, std_final.values))
        y_mu = returns_final.values.ravel()  # 1D target

        # Sanity check
        if len(X) != len(y_mu):
            raise ValueError(f"X and y lengths do not match! {len(X)} vs {len(y_mu)}")

        # Train mu model
        self.mu_model.fit(X, y_mu)

        # Compute realistic sigma as rolling std of log returns
        self.sigma_series = log_returns.rolling(window=self.window).std().dropna()

    def predict(self, close: pd.Series):
        """Predict mu using ML, and sigma using rolling volatility (log returns)."""
        # ML-predicted mu
        features = self._compute_features(close)
        mu_pred = self.mu_model.predict(features)[0]

        # Sigma: last available rolling volatility of log returns
        if len(self.sigma_series) == 0:
            raise ValueError("Sigma series is empty. Train the model first.")
        sigma_pred = float(self.sigma_series.iloc[-1])

        return mu_pred, sigma_pred
