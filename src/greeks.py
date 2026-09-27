"""Sensitivity estimation engine (The Greeks) via Bump-and-Revalue Finite Difference schemes."""

from src.analytics import BlackScholesEngine
from src.config import MarketConfig, SimulationConfig
from src.monte_carlo import MonteCarloEngine


class GreeksEngine:
    """Computes Delta, Gamma, Vega, Theta, and Rho using finite differences."""

    def __init__(self, mkt_cfg: MarketConfig, sim_cfg: SimulationConfig) -> None:
        self.mkt = mkt_cfg
        self.sim = sim_cfg

    def compute_mc_greeks(self, is_call: bool = False) -> dict:
        """Calculates Monte Carlo numerical Greeks via symmetric central perturbations."""
        ds = self.mkt.s0 * 0.01  # 1% spot bump
        dvol = 0.01  # 1% volatility bump
        dr = 0.0001  # 1 bp rate bump
        dt_shift = 1.0 / 365.0  # 1 day time decay

        mc_engine = MonteCarloEngine(self.mkt, self.sim)

        base_res = mc_engine.evaluate_european(self.sim.m_simulations)
        base_price = (
            base_res["Call_Price"] if is_call else base_res["Put_Price"]
        )

        cfg_up = MarketConfig(s0=self.mkt.s0 + ds)
        cfg_dn = MarketConfig(s0=self.mkt.s0 - ds)
        p_up = MonteCarloEngine(cfg_up, self.sim).evaluate_european(
            self.sim.m_simulations
        )["Call_Price" if is_call else "Put_Price"]
        p_dn = MonteCarloEngine(cfg_dn, self.sim).evaluate_european(
            self.sim.m_simulations
        )["Call_Price" if is_call else "Put_Price"]

        delta = (p_up - p_dn) / (2.0 * ds)
        gamma = (p_up - 2.0 * base_price + p_dn) / (ds**2)

        cfg_vol = MarketConfig(sigma=self.mkt.sigma + dvol)
        p_vol = MonteCarloEngine(cfg_vol, self.sim).evaluate_european(
            self.sim.m_simulations
        )["Call_Price" if is_call else "Put_Price"]
        vega = (p_vol - base_price) / (dvol * 100.0)

        cfg_r = MarketConfig(r=self.mkt.r + dr)
        p_r = MonteCarloEngine(cfg_r, self.sim).evaluate_european(
            self.sim.m_simulations
        )["Call_Price" if is_call else "Put_Price"]
        rho = (p_r - base_price) / (dr * 10000.0)

        cfg_t = MarketConfig(t=max(1e-4, self.mkt.t - dt_shift))
        p_t = MonteCarloEngine(cfg_t, self.sim).evaluate_european(
            self.sim.m_simulations
        )["Call_Price" if is_call else "Put_Price"]
        theta = p_t - base_price

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Vega": vega,
            "Theta": theta,
            "Rho": rho,
        }