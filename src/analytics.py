"""Analytical Black-Scholes solver and 20-step Cox-Ross-Rubinstein Binomial Tree engine."""

import numpy as np
from scipy.stats import norm
from src.config import MarketConfig


class BlackScholesEngine:
    """Closed-form Black-Scholes-Merton option pricing and analytical Greeks."""

    def __init__(self, config: MarketConfig) -> None:
        self.cfg = config

    def d1(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> float:
        return (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (
            sigma * np.sqrt(t)
        )

    def d2(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> float:
        return self.d1(s, k, t, r, q, sigma) - sigma * np.sqrt(t)

    def price_call(self) -> float:
        d1 = self.d1(
            self.cfg.s0,
            self.cfg.k,
            self.cfg.t,
            self.cfg.r,
            self.cfg.q,
            self.cfg.sigma,
        )
        d2 = self.d2(
            self.cfg.s0,
            self.cfg.k,
            self.cfg.t,
            self.cfg.r,
            self.cfg.q,
            self.cfg.sigma,
        )
        return self.cfg.s0 * np.exp(-self.cfg.q * self.cfg.t) * norm.cdf(
            d1
        ) - self.cfg.k * np.exp(-self.cfg.r * self.cfg.t) * norm.cdf(d2)

    def price_put(self) -> float:
        d1 = self.d1(
            self.cfg.s0,
            self.cfg.k,
            self.cfg.t,
            self.cfg.r,
            self.cfg.q,
            self.cfg.sigma,
        )
        d2 = self.d2(
            self.cfg.s0,
            self.cfg.k,
            self.cfg.t,
            self.cfg.r,
            self.cfg.q,
            self.cfg.sigma,
        )
        return self.cfg.k * np.exp(-self.cfg.r * self.cfg.t) * norm.cdf(
            -d2
        ) - self.cfg.s0 * np.exp(-self.cfg.q * self.cfg.t) * norm.cdf(-d1)

    def greeks(self) -> dict:
        """Analytical Black-Scholes sensitivity gradients."""
        s, k, t, r, q, sigma = (
            self.cfg.s0,
            self.cfg.k,
            self.cfg.t,
            self.cfg.r,
            self.cfg.q,
            self.cfg.sigma,
        )
        d1_val = self.d1(s, k, t, r, q, sigma)
        d2_val = self.d2(s, k, t, r, q, sigma)

        delta_call = np.exp(-q * t) * norm.cdf(d1_val)
        delta_put = np.exp(-q * t) * (norm.cdf(d1_val) - 1.0)
        gamma = (
            np.exp(-q * t) * norm.pdf(d1_val) / (s * sigma * np.sqrt(t))
        )
        vega = s * np.exp(-q * t) * norm.pdf(d1_val) * np.sqrt(t)
        theta_call = (
            -(s * sigma * np.exp(-q * t) * norm.pdf(d1_val))
            / (2 * np.sqrt(t))
            - r * k * np.exp(-r * t) * norm.cdf(d2_val)
            + q * s * np.exp(-q * t) * norm.cdf(d1_val)
        )
        theta_put = (
            -(s * sigma * np.exp(-q * t) * norm.pdf(d1_val))
            / (2 * np.sqrt(t))
            + r * k * np.exp(-r * t) * norm.cdf(-d2_val)
            - q * s * np.exp(-q * t) * norm.cdf(-d1_val)
        )
        rho_call = k * t * np.exp(-r * t) * norm.cdf(d2_val)
        rho_put = -k * t * np.exp(-r * t) * norm.cdf(-d2_val)

        return {
            "Call": {
                "Delta": delta_call,
                "Gamma": gamma,
                "Vega": vega / 100.0,
                "Theta": theta_call / 365.0,
                "Rho": rho_call / 100.0,
            },
            "Put": {
                "Delta": delta_put,
                "Gamma": gamma,
                "Vega": vega / 100.0,
                "Theta": theta_put / 365.0,
                "Rho": rho_put / 100.0,
            },
        }


class BinomialTreeEngine:
    """Cox-Ross-Rubinstein (CRR) 20-step Binomial Tree pricing engine for Q1 and Q2."""

    def __init__(self, config: MarketConfig, n_steps: int = 20) -> None:
        self.cfg = config
        self.n = n_steps
        self.dt = self.cfg.t / self.n
        self.u = np.exp(self.cfg.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (np.exp((self.cfg.r - self.cfg.q) * self.dt) - self.d) / (
            self.u - self.d
        )
        self.df = np.exp(-self.cfg.r * self.dt)

    def price_standard_option(
        self, is_call: bool = False, is_american: bool = True
    ) -> float:
        """Prices standard European or American options via CRR tree (Q1)."""
        asset_prices = np.zeros(self.n + 1)
        option_values = np.zeros(self.n + 1)

        for j in range(self.n + 1):
            asset_prices[j] = self.cfg.s0 * (
                self.u ** (self.n - j)
            ) * (self.d**j)
            if is_call:
                option_values[j] = max(0.0, asset_prices[j] - self.cfg.k)
            else:
                option_values[j] = max(0.0, self.cfg.k - asset_prices[j])

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                asset_prices[j] = self.cfg.s0 * (
                    self.u ** (i - j)
                ) * (self.d**j)
                continuation = self.df * (
                    self.p * option_values[j]
                    + (1.0 - self.p) * option_values[j + 1]
                )
                if is_american:
                    exercise = (
                        (asset_prices[j] - self.cfg.k)
                        if is_call
                        else (self.cfg.k - asset_prices[j])
                    )
                    option_values[j] = max(continuation, exercise)
                else:
                    option_values[j] = continuation

        return option_values[0]

    def price_compound_option(
        self,
    ) -> float:
        """Prices Call-on-Call compound option (Q2) using nested 20-step trees."""
        # T1 = 1.0 yr, T2 = 2.0 yr, strike X2 = 20.0
        n1 = self.n  # 20 steps for phase 1 (0 to T1)
        n2 = self.n  # 20 steps for phase 2 (T1 to T2)
        dt1 = self.cfg.t1 / n1
        dt2 = (self.cfg.t2 - self.cfg.t1) / n2

        u1 = np.exp(self.cfg.sigma * np.sqrt(dt1))
        d1 = 1.0 / u1
        p1 = (np.exp((self.cfg.r - self.cfg.q) * dt1) - d1) / (u1 - d1)
        df1 = np.exp(-self.cfg.r * dt1)

        u2 = np.exp(self.cfg.sigma * np.sqrt(dt2))
        d2 = 1.0 / u2
        p2 = (np.exp((self.cfg.r - self.cfg.q) * dt2) - d2) / (u2 - d2)
        df2 = np.exp(-self.cfg.r * dt2)

        # Values of underlying option at T1 across all stock price nodes
        underlying_values_at_t1 = np.zeros(n1 + 1)

        for i in range(n1 + 1):
            s_t1 = self.cfg.s0 * (u1 ** (n1 - i)) * (d1**i)

            # Build 20-step sub-tree from T1 to T2 for underlying European call with strike K
            sub_opt = np.zeros(n2 + 1)
            for j in range(n2 + 1):
                s_t2 = s_t1 * (u2 ** (n2 - j)) * (d2**j)
                sub_opt[j] = max(0.0, s_t2 - self.cfg.k)

            for step in range(n2 - 1, -1, -1):
                for j in range(step + 1):
                    sub_opt[j] = df2 * (
                        p2 * sub_opt[j] + (1.0 - p2) * sub_opt[j + 1]
                    )

            underlying_values_at_t1[i] = sub_opt[0]

        # Terminal payoff of compound option at T1: max(0, V_underlying(T1) - X2)
        compound_values = np.maximum(
            0.0, underlying_values_at_t1 - self.cfg.x2
        )

        # Roll back compound option from T1 to t=0
        for step in range(n1 - 1, -1, -1):
            for j in range(step + 1):
                compound_values[j] = df1 * (
                    p1 * compound_values[j]
                    + (1.0 - p1) * compound_values[j + 1]
                )

        return compound_values[0]