"""Longstaff-Schwartz (2001) American option pricing engine with Laguerre basis polynomials."""

import numpy as np
from src.config import MarketConfig, SimulationConfig


class LongstaffSchwartzEngine:
    """Least-Squares Monte Carlo (LSM) engine using orthogonal Laguerre basis functions."""

    def __init__(self, mkt_cfg: MarketConfig, sim_cfg: SimulationConfig) -> None:
        self.mkt = mkt_cfg
        self.sim = sim_cfg
        self.dt = self.mkt.t / self.sim.n_steps
        self.df = np.exp(-self.mkt.r * self.dt)

    @staticmethod
    def laguerre_basis(x: np.ndarray) -> np.ndarray:
        """Computes weighted orthogonal Laguerre polynomials L0, L1, L2, L3."""
        l0 = np.ones_like(x)
        l1 = 1.0 - x
        l2 = 1.0 - 2.0 * x + 0.5 * (x**2)
        l3 = 1.0 - 3.0 * x + 1.5 * (x**2) - (x**3) / 6.0
        return np.column_stack([l0, l1, l2, l3])

    def evaluate_american(
        self, paths: np.ndarray, is_call: bool = False
    ) -> dict:
        """Executes backward induction regression over ITM paths."""
        m_sim, n_p1 = paths.shape
        n_steps = n_p1 - 1

        # Terminal cash flows
        if is_call:
            cash_flows = np.maximum(paths[:, -1] - self.mkt.k, 0.0)
        else:
            cash_flows = np.maximum(self.mkt.k - paths[:, -1], 0.0)

        stopping_times = np.full(m_sim, n_steps)
        exercise_boundary = {}

        # Backward induction from t_{N-1} down to t_1
        for t in range(n_steps - 1, 0, -1):
            s_t = paths[:, t]

            # Identify In-The-Money (ITM) paths
            itm = (s_t > self.mkt.k) if is_call else (s_t < self.mkt.k)
            if not np.any(itm):
                continue

            s_itm = s_t[itm]
            x_scaled = s_itm / self.mkt.k

            # Discounted future realized cash flows
            time_steps_ahead = stopping_times[itm] - t
            y_continuation = cash_flows[itm] * (self.df**time_steps_ahead)

            # Design matrix with Laguerre basis
            a_matrix = self.laguerre_basis(x_scaled)

            # Least-Squares Regression
            coefficients, *_ = np.linalg.lstsq(
                a_matrix, y_continuation, rcond=None
            )
            predicted_continuation = a_matrix @ coefficients

            # Immediate exercise payoff
            immediate_exercise = (
                (s_itm - self.mkt.k) if is_call else (self.mkt.k - s_itm)
            )

            # Stopping decision rule
            do_exercise = immediate_exercise > predicted_continuation

            idx_itm = np.where(itm)[0]
            exercised_indices = idx_itm[do_exercise]

            # Update cash flows and stopping timestamps
            cash_flows[exercised_indices] = immediate_exercise[do_exercise]
            stopping_times[exercised_indices] = t

            # Record critical exercise threshold for plotting
            if len(exercised_indices) > 0:
                boundary_price = (
                    np.min(s_t[exercised_indices])
                    if is_call
                    else np.max(s_t[exercised_indices])
                )
                exercise_boundary[t * self.dt] = boundary_price

        # Discount cash flows back to t = 0
        discounted_present_value = cash_flows * (self.df**stopping_times)
        price = np.mean(discounted_present_value)
        se = np.std(discounted_present_value, ddof=1) / np.sqrt(m_sim)

        return {
            "Price": price,
            "SE": se,
            "CI": (price - 1.96 * se, price + 1.96 * se),
            "Exercise_Boundary": exercise_boundary,
        }