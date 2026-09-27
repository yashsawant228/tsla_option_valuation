"""Configuration dataclasses for market parameters and simulation settings."""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class MarketConfig:
    """Immutable market and option parameters."""

    s0: float = 311.21  # Underlying closed price at 31 July 2026
    k: float = 311.21  # Strike price (ATM)
    r: float = 0.03903  # 6-month US risk-free yield
    sigma: float = 0.466810  # Annualised volatility
    q: float = 0.0  # Dividend yield (TSLA = 0)
    t: float = 0.5  # Time to maturity (6 months)

    # Q2 Compound Option Parameters
    x2: float = 20.0  # Compound option strike
    t1: float = 1.0  # Maturity of first option (1 year)
    t2: float = 2.0  # Maturity of underlying option (2 years)


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable Monte Carlo and execution settings."""

    n_steps: int = 20  # Time steps (matching coursework)
    m_simulations: int = 1000  # Baseline simulation path count
    m_large: int = 100000  # High-precision simulation path count
    seed: int = 42  # Seed for reproducibility
    device_name: str = "mps" if torch.backends.mps.is_available() else "cpu"