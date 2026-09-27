"""Vectorised PyTorch / Metal (MPS) & NumPy Monte Carlo path generator with variance reduction."""

import numpy as np
import torch
from scipy.stats import qmc
from src.config import MarketConfig, SimulationConfig


class MonteCarloEngine:
    """High-performance accelerated path generator and European option engine."""

    def __init__(self, mkt_cfg: MarketConfig, sim_cfg: SimulationConfig) -> None:
        self.mkt = mkt_cfg
        self.sim = sim_cfg
        self.dt = self.mkt.t / self.sim.n_steps
        self.device = torch.device(
            self.sim.device_name
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def generate_paths_numpy(
        self,
        m_paths: int,
        use_antithetic: bool = False,
        use_sobol: bool = False,
    ) -> np.ndarray:
        """Generates asset price paths (M, N+1) using NumPy CPU execution."""
        n_steps = self.sim.n_steps
        drift = (self.mkt.r - self.mkt.q - 0.5 * self.mkt.sigma**2) * self.dt
        vol_sq_dt = self.mkt.sigma * np.sqrt(self.dt)

        if use_sobol:
            sampler = qmc.Sobol(
                d=n_steps, scramble=True, seed=self.sim.seed
            )
            # Power of 2 for optimal Sobol balance
            pow_m = int(np.ceil(np.log2(m_paths)))
            u_draws = sampler.random_base2(m=pow_m)[:m_paths]
            # Clip for numerical stability near 0 and 1
            u_draws = np.clip(u_draws, 1e-7, 1 - 1e-7)
            z = norm_ppf_approx(u_draws)
        elif use_antithetic:
            half_m = m_paths // 2
            np.random.seed(self.sim.seed)
            z_half = np.random.standard_normal((half_m, n_steps))
            z = np.vstack([z_half, -z_half])
        else:
            np.random.seed(self.sim.seed)
            z = np.random.standard_normal((m_paths, n_steps))

        log_increments = drift + vol_sq_dt * z
        log_paths = np.cumsum(log_increments, axis=1)

        paths = np.zeros((z.shape[0], n_steps + 1))
        paths[:, 0] = self.mkt.s0
        paths[:, 1:] = self.mkt.s0 * np.exp(log_paths)
        return paths

    def generate_paths_torch_mps(self, m_paths: int) -> torch.Tensor:
        """Generates GPU-accelerated tensor paths on Apple Silicon Metal (MPS)."""
        torch.manual_seed(self.sim.seed)
        n_steps = self.sim.n_steps
        drift = (self.mkt.r - self.mkt.q - 0.5 * self.mkt.sigma**2) * self.dt
        vol_sq_dt = self.mkt.sigma * np.sqrt(self.dt)

        z = torch.randn(
            (m_paths, n_steps), device=self.device, dtype=torch.float32
        )
        log_increments = drift + vol_sq_dt * z
        log_paths = torch.cumsum(log_increments, dim=1)

        paths = torch.zeros(
            (m_paths, n_steps + 1), device=self.device, dtype=torch.float32
        )
        paths[:, 0] = self.mkt.s0
        paths[:, 1:] = self.mkt.s0 * torch.exp(log_paths)
        return paths

    def evaluate_european(
        self,
        m_paths: int,
        use_antithetic: bool = False,
        use_sobol: bool = False,
        use_control_variate: bool = False,
        bs_call_exact: float = 0.0,
        bs_put_exact: float = 0.0,
    ) -> dict:
        """Evaluates European options with Standard Errors and 95% Confidence Intervals."""
        paths = self.generate_paths_numpy(
            m_paths, use_antithetic=use_antithetic, use_sobol=use_sobol
        )
        s_t = paths[:, -1]
        df = np.exp(-self.mkt.r * self.mkt.t)

        call_payoffs = np.maximum(s_t - self.mkt.k, 0.0)
        put_payoffs = np.maximum(self.mkt.k - s_t, 0.0)

        discounted_calls = df * call_payoffs
        discounted_puts = df * put_payoffs

        if use_control_variate:
            # Control Variate adjustment against analytical BS expectation
            c_cov = np.cov(discounted_calls, s_t)[0, 1]
            c_var = np.var(s_t)
            beta_c = c_cov / c_var if c_var > 0 else 0.0
            expected_st = self.mkt.s0 * np.exp(
                (self.mkt.r - self.mkt.q) * self.mkt.t
            )
            discounted_calls = discounted_calls - beta_c * (
                s_t - expected_st
            )

            p_cov = np.cov(discounted_puts, s_t)[0, 1]
            beta_p = p_cov / c_var if c_var > 0 else 0.0
            discounted_puts = discounted_puts - beta_p * (s_t - expected_st)

        c_price = np.mean(discounted_calls)
        p_price = np.mean(discounted_puts)

        c_se = np.std(discounted_calls, ddof=1) / np.sqrt(len(discounted_calls))
        p_se = np.std(discounted_puts, ddof=1) / np.sqrt(len(discounted_puts))

        return {
            "Call_Price": c_price,
            "Call_SE": c_se,
            "Call_CI": (c_price - 1.96 * c_se, c_price + 1.96 * c_se),
            "Put_Price": p_price,
            "Put_SE": p_se,
            "Put_CI": (p_price - 1.96 * p_se, p_price + 1.96 * p_se),
        }


def norm_ppf_approx(u: np.ndarray) -> np.ndarray:
    """High-precision inverse CDF approximation for Sobol sequence mapping."""
    from scipy.stats import norm

    return norm.ppf(u)