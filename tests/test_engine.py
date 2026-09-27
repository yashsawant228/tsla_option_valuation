"""Comprehensive test suite verifying analytical, numerical, and accelerated pricing engines."""

import numpy as np
import pytest
import torch

from src.analytics import BinomialTreeEngine, BlackScholesEngine
from src.config import MarketConfig, SimulationConfig
from src.greeks import GreeksEngine
from src.longstaff_schwartz import LongstaffSchwartzEngine
from src.monte_carlo import MonteCarloEngine


@pytest.fixture
def default_market_config() -> MarketConfig:
    """Provides default market parameter configuration."""
    return MarketConfig()


@pytest.fixture
def default_sim_config() -> SimulationConfig:
    """Provides default simulation parameter configuration."""
    return SimulationConfig(m_simulations=1000, seed=42)


class TestConfig:
    """Tests configuration dataclasses and parameter calibration."""

    def test_market_config_defaults(
        self, default_market_config: MarketConfig
    ) -> None:
        cfg = default_market_config
        assert cfg.s0 == 311.21
        assert cfg.k == 311.21
        assert cfg.r == 0.03903
        assert cfg.sigma == 0.466810
        assert cfg.q == 0.0
        assert cfg.t == 0.5

    def test_simulation_config_defaults(
        self, default_sim_config: SimulationConfig
    ) -> None:
        cfg = default_sim_config
        assert cfg.n_steps == 20
        assert cfg.m_simulations == 1000
        assert cfg.seed == 42


class TestBlackScholesEngine:
    """Tests Black-Scholes analytical pricing and put-call parity."""

    def test_bs_prices_and_parity(
        self, default_market_config: MarketConfig
    ) -> None:
        bs = BlackScholesEngine(default_market_config)
        call_price = bs.price_call()
        put_price = bs.price_put()

        assert call_price > 0.0
        assert put_price > 0.0

        # Put-Call Parity identity: C - P = S0 * e^(-qT) - K * e^(-rT)
        lhs = call_price - put_price
        rhs = default_market_config.s0 * np.exp(
            -default_market_config.q * default_market_config.t
        ) - default_market_config.k * np.exp(
            -default_market_config.r * default_market_config.t
        )
        assert np.isclose(lhs, rhs, atol=1e-5)

    def test_bs_greeks_structure(
        self, default_market_config: MarketConfig
    ) -> None:
        bs = BlackScholesEngine(default_market_config)
        greeks = bs.greeks()

        for opt in ["Call", "Put"]:
            assert opt in greeks
            for g_name in ["Delta", "Gamma", "Vega", "Theta", "Rho"]:
                assert g_name in greeks[opt]

        # Analytical Delta bounds
        assert 0.0 < greeks["Call"]["Delta"] < 1.0
        assert -1.0 < greeks["Put"]["Delta"] < 0.0
        assert np.isclose(
            greeks["Call"]["Delta"] - greeks["Put"]["Delta"], 1.0, atol=1e-4
        )


class TestBinomialTreeEngine:
    """Tests 20-step CRR Binomial Tree European and American option valuation."""

    def test_crr_european_convergence(
        self, default_market_config: MarketConfig
    ) -> None:
        bs = BlackScholesEngine(default_market_config)
        bs_call = bs.price_call()
        bs_put = bs.price_put()

        tree = BinomialTreeEngine(default_market_config, n_steps=20)
        crr_call = tree.price_standard_option(is_call=True, is_american=False)
        crr_put = tree.price_standard_option(is_call=False, is_american=False)

        # 20-step CRR converges within $0.60 of analytical Black-Scholes
        assert np.isclose(crr_call, bs_call, atol=0.6)
        assert np.isclose(crr_put, bs_put, atol=0.6)

    def test_american_early_exercise_properties(
        self, default_market_config: MarketConfig
    ) -> None:
        tree = BinomialTreeEngine(default_market_config, n_steps=20)
        crr_euro_call = tree.price_standard_option(
            is_call=True, is_american=False
        )
        crr_amer_call = tree.price_standard_option(
            is_call=True, is_american=True
        )
        crr_euro_put = tree.price_standard_option(
            is_call=False, is_american=False
        )
        crr_amer_put = tree.price_standard_option(
            is_call=False, is_american=True
        )

        # Zero dividend yield (q = 0) implies American Call == European Call
        assert np.isclose(crr_amer_call, crr_euro_call, atol=1e-6)

        # American Put early exercise premium is strictly positive
        assert crr_amer_put >= crr_euro_put
        assert (crr_amer_put - crr_euro_put) > 0.1

    def test_compound_option_pricing(
        self, default_market_config: MarketConfig
    ) -> None:
        tree = BinomialTreeEngine(default_market_config, n_steps=20)
        compound_val = tree.price_compound_option()
        assert compound_val > 0.0
        assert compound_val < default_market_config.s0


class TestMonteCarloEngine:
    """Tests path generation and European option simulation accuracy."""

    def test_path_generation_shape(
        self,
        default_market_config: MarketConfig,
        default_sim_config: SimulationConfig,
    ) -> None:
        mc = MonteCarloEngine(default_market_config, default_sim_config)

        paths_plain = mc.generate_paths_numpy(1000)
        assert paths_plain.shape == (1000, 21)
        assert np.allclose(paths_plain[:, 0], default_market_config.s0)

        paths_anti = mc.generate_paths_numpy(1000, use_antithetic=True)
        assert paths_anti.shape == (1000, 21)

        paths_sobol = mc.generate_paths_numpy(1000, use_sobol=True)
        assert paths_sobol.shape == (1000, 21)

    def test_torch_path_generation(
        self,
        default_market_config: MarketConfig,
        default_sim_config: SimulationConfig,
    ) -> None:
        mc = MonteCarloEngine(default_market_config, default_sim_config)
        torch_paths = mc.generate_paths_torch_mps(500)
        assert torch_paths.shape == (500, 21)
        assert isinstance(torch_paths, torch.Tensor)

    def test_european_mc_pricing(
        self,
        default_market_config: MarketConfig,
        default_sim_config: SimulationConfig,
    ) -> None:
        bs = BlackScholesEngine(default_market_config)
        bs_call = bs.price_call()
        bs_put = bs.price_put()

        mc = MonteCarloEngine(default_market_config, default_sim_config)
        res = mc.evaluate_european(1000)

        # 95% Confidence interval must bound analytical Black-Scholes benchmark
        assert res["Call_CI"][0] <= bs_call <= res["Call_CI"][1]
        assert res["Put_CI"][0] <= bs_put <= res["Put_CI"][1]


class TestLongstaffSchwartzEngine:
    """Tests Least-Squares Monte Carlo (LSM) American option valuation."""

    def test_laguerre_basis_shape(self) -> None:
        x = np.array([1.0, 1.1, 0.9])
        basis = LongstaffSchwartzEngine.laguerre_basis(x)
        assert basis.shape == (3, 4)

    def test_lsm_american_option_pricing(
        self,
        default_market_config: MarketConfig,
        default_sim_config: SimulationConfig,
    ) -> None:
        mc = MonteCarloEngine(default_market_config, default_sim_config)
        paths = mc.generate_paths_numpy(2000)

        lsm = LongstaffSchwartzEngine(default_market_config, default_sim_config)
        call_res = lsm.evaluate_american(paths, is_call=True)
        put_res = lsm.evaluate_american(paths, is_call=False)

        assert call_res["Price"] > 0.0
        assert put_res["Price"] > 0.0

        euro_put = mc.evaluate_european(2000)["Put_Price"]
        assert put_res["Price"] >= euro_put - 0.5


class TestGreeksEngine:
    """Tests finite difference sensitivity calculations."""

    def test_finite_difference_greeks(
        self,
        default_market_config: MarketConfig,
        default_sim_config: SimulationConfig,
    ) -> None:
        bs = BlackScholesEngine(default_market_config)
        bs_greeks = bs.greeks()

        greeks_eng = GreeksEngine(default_market_config, default_sim_config)
        mc_call_greeks = greeks_eng.compute_mc_greeks(is_call=True)

        # Monte Carlo finite difference Delta matches analytical BS Delta within tolerance
        assert np.isclose(
            mc_call_greeks["Delta"], bs_greeks["Call"]["Delta"], atol=0.15
        )