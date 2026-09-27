"""Master Execution Pipeline orchestrating calibration, benchmarking, and reporting."""

import logging
import time
import pandas as pd

from src.analytics import BinomialTreeEngine, BlackScholesEngine
from src.config import MarketConfig, SimulationConfig
from src.greeks import GreeksEngine
from src.longstaff_schwartz import LongstaffSchwartzEngine
from src.monte_carlo import MonteCarloEngine
from src.reporting import ReportingEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_pipeline() -> None:
  logging.info(
      "Initialising TSLA Option Valuation Pipeline (Apple Silicon Native)..."
  )

  # 1. Config Calibration
  mkt_cfg = MarketConfig()
  sim_cfg = SimulationConfig()

  logging.info(
      f"Calibrated Spot S0={mkt_cfg.s0}, Risk-Free r={mkt_cfg.r}, Volatility"
      f" sigma={mkt_cfg.sigma}"
  )

  # 2. Analytic Benchmarks (Black-Scholes)
  bs_engine = BlackScholesEngine(mkt_cfg)
  bs_call = bs_engine.price_call()
  bs_put = bs_engine.price_put()
  bs_greeks = bs_engine.greeks()

  logging.info(
      f"Black-Scholes Benchmarks -> Call: ${bs_call:.4f}, Put: ${bs_put:.4f}"
  )

  # 3. Binomial Tree Benchmarks (Q1 & Q2)
  tree_engine = BinomialTreeEngine(mkt_cfg, n_steps=sim_cfg.n_steps)
  crr_euro_call = tree_engine.price_standard_option(
      is_call=True, is_american=False
  )
  crr_euro_put = tree_engine.price_standard_option(
      is_call=False, is_american=False
  )
  crr_amer_call = tree_engine.price_standard_option(
      is_call=True, is_american=True
  )
  crr_amer_put = tree_engine.price_standard_option(
      is_call=False, is_american=True
  )
  crr_compound_call = tree_engine.price_compound_option()

  logging.info(
      "Q1 CRR Binomial 20-Step -> Euro Call: $"
      f" {crr_euro_call:.4f}, Euro Put: ${crr_euro_put:.4f}"
  )
  logging.info(
      "Q1 CRR Binomial 20-Step -> Amer Call: $"
      f" {crr_amer_call:.4f}, Amer Put: ${crr_amer_put:.4f}"
  )
  logging.info(f"Q2 Compound Call-on-Call Price -> ${crr_compound_call:.4f}")

  # 4. Monte Carlo Engine Simulations (Q3a)
  mc_engine = MonteCarloEngine(mkt_cfg, sim_cfg)

  t0 = time.time()
  res_plain = mc_engine.evaluate_european(sim_cfg.m_simulations)
  res_anti = mc_engine.evaluate_european(
      sim_cfg.m_simulations, use_antithetic=True
  )
  res_sobol = mc_engine.evaluate_european(sim_cfg.m_simulations, use_sobol=True)
  res_cv = mc_engine.evaluate_european(
      sim_cfg.m_simulations,
      use_control_variate=True,
      bs_call_exact=bs_call,
      bs_put_exact=bs_put,
  )
  t_mc = time.time() - t0

  logging.info(
      f"Monte Carlo European evaluation completed in {t_mc:.4f} seconds."
  )

  # 5. Longstaff-Schwartz American Engine (Q3b)
  lsm_engine = LongstaffSchwartzEngine(mkt_cfg, sim_cfg)

  lsm_paths = mc_engine.generate_paths_numpy(sim_cfg.m_simulations)
  res_lsm_call = lsm_engine.evaluate_american(lsm_paths, is_call=True)
  res_lsm_put = lsm_engine.evaluate_american(lsm_paths, is_call=False)

  logging.info(
      f"Longstaff-Schwartz American Call Price -> ${res_lsm_call['Price']:.4f}"
  )
  logging.info(
      f"Longstaff-Schwartz American Put Price  -> ${res_lsm_put['Price']:.4f}"
  )

  put_reg_df = res_lsm_put.get("Regression_Table")
  call_reg_df = res_lsm_call.get("Regression_Table")
  if put_reg_df is not None:
    logging.info(
        "\n--- LSM American Put Backward Regression Details (First 5 steps)"
        " ---\n"
        + put_reg_df.head(5).to_string(index=False)
    )

  # 6. High-Precision Path Simulation for Graphics & Convergence
  large_paths = mc_engine.generate_paths_numpy(sim_cfg.m_large)
  res_lsm_put_large = lsm_engine.evaluate_american(large_paths, is_call=False)

  # 7. Construct Summary DataFrame
  summary_data = [
      {
          "Model / Method": "Black-Scholes Analytical",
          "Option Type": "European Call",
          "Price ($)": bs_call,
          "Std Error": 0.0,
          "95% CI Lower": bs_call,
          "95% CI Upper": bs_call,
      },
      {
          "Model / Method": "Black-Scholes Analytical",
          "Option Type": "European Put",
          "Price ($)": bs_put,
          "Std Error": 0.0,
          "95% CI Lower": bs_put,
          "95% CI Upper": bs_put,
      },
      {
          "Model / Method": "CRR Binomial Tree (20-Step)",
          "Option Type": "European Call",
          "Price ($)": crr_euro_call,
          "Std Error": 0.0,
          "95% CI Lower": crr_euro_call,
          "95% CI Upper": crr_euro_call,
      },
      {
          "Model / Method": "CRR Binomial Tree (20-Step)",
          "Option Type": "European Put",
          "Price ($)": crr_euro_put,
          "Std Error": 0.0,
          "95% CI Lower": crr_euro_put,
          "95% CI Upper": crr_euro_put,
      },
      {
          "Model / Method": "CRR Binomial Tree (20-Step)",
          "Option Type": "American Call",
          "Price ($)": crr_amer_call,
          "Std Error": 0.0,
          "95% CI Lower": crr_amer_call,
          "95% CI Upper": crr_amer_call,
      },
      {
          "Model / Method": "CRR Binomial Tree (20-Step)",
          "Option Type": "American Put",
          "Price ($)": crr_amer_put,
          "Std Error": 0.0,
          "95% CI Lower": crr_amer_put,
          "95% CI Upper": crr_amer_put,
      },
      {
          "Model / Method": "Q2 Compound Call-on-Call",
          "Option Type": "European Compound",
          "Price ($)": crr_compound_call,
          "Std Error": 0.0,
          "95% CI Lower": crr_compound_call,
          "95% CI Upper": crr_compound_call,
      },
      {
          "Model / Method": "Plain Monte Carlo (M=1,000)",
          "Option Type": "European Call",
          "Price ($)": res_plain["Call_Price"],
          "Std Error": res_plain["Call_SE"],
          "95% CI Lower": res_plain["Call_CI"][0],
          "95% CI Upper": res_plain["Call_CI"][1],
      },
      {
          "Model / Method": "Plain Monte Carlo (M=1,000)",
          "Option Type": "European Put",
          "Price ($)": res_plain["Put_Price"],
          "Std Error": res_plain["Put_SE"],
          "95% CI Lower": res_plain["Put_CI"][0],
          "95% CI Upper": res_plain["Put_CI"][1],
      },
      {
          "Model / Method": "Antithetic Variate MC (M=1,000)",
          "Option Type": "European Call",
          "Price ($)": res_anti["Call_Price"],
          "Std Error": res_anti["Call_SE"],
          "95% CI Lower": res_anti["Call_CI"][0],
          "95% CI Upper": res_anti["Call_CI"][1],
      },
      {
          "Model / Method": "Antithetic Variate MC (M=1,000)",
          "Option Type": "European Put",
          "Price ($)": res_anti["Put_Price"],
          "Std Error": res_anti["Put_SE"],
          "95% CI Lower": res_anti["Put_CI"][0],
          "95% CI Upper": res_anti["Put_CI"][1],
      },
      {
          "Model / Method": "Sobol QMC (M=1,000)",
          "Option Type": "European Call",
          "Price ($)": res_sobol["Call_Price"],
          "Std Error": res_sobol["Call_SE"],
          "95% CI Lower": res_sobol["Call_CI"][0],
          "95% CI Upper": res_sobol["Call_CI"][1],
      },
      {
          "Model / Method": "Sobol QMC (M=1,000)",
          "Option Type": "European Put",
          "Price ($)": res_sobol["Put_Price"],
          "Std Error": res_sobol["Put_SE"],
          "95% CI Lower": res_sobol["Put_CI"][0],
          "95% CI Upper": res_sobol["Put_CI"][1],
      },
      {
          "Model / Method": "Control Variate MC (M=1,000)",
          "Option Type": "European Call",
          "Price ($)": res_cv["Call_Price"],
          "Std Error": res_cv["Call_SE"],
          "95% CI Lower": res_cv["Call_CI"][0],
          "95% CI Upper": res_cv["Call_CI"][1],
      },
      {
          "Model / Method": "Control Variate MC (M=1,000)",
          "Option Type": "European Put",
          "Price ($)": res_cv["Put_Price"],
          "Std Error": res_cv["Put_SE"],
          "95% CI Lower": res_cv["Put_CI"][0],
          "95% CI Upper": res_cv["Put_CI"][1],
      },
      {
          "Model / Method": "Longstaff-Schwartz LSM (M=1,000)",
          "Option Type": "American Call",
          "Price ($)": res_lsm_call["Price"],
          "Std Error": res_lsm_call["SE"],
          "95% CI Lower": res_lsm_call["CI"][0],
          "95% CI Upper": res_lsm_call["CI"][1],
      },
      {
          "Model / Method": "Longstaff-Schwartz LSM (M=1,000)",
          "Option Type": "American Put",
          "Price ($)": res_lsm_put["Price"],
          "Std Error": res_lsm_put["SE"],
          "95% CI Lower": res_lsm_put["CI"][0],
          "95% CI Upper": res_lsm_put["CI"][1],
      },
  ]

  summary_df = pd.DataFrame(summary_data)

  # 8. Greeks Analysis Table
  greeks_engine = GreeksEngine(mkt_cfg, sim_cfg)
  mc_call_greeks = greeks_engine.compute_mc_greeks(is_call=True)
  mc_put_greeks = greeks_engine.compute_mc_greeks(is_call=False)

  greeks_df = pd.DataFrame({
      "BS Analytical Call": bs_greeks["Call"],
      "BS Analytical Put": bs_greeks["Put"],
      "MC Finite Diff Call": mc_call_greeks,
      "MC Finite Diff Put": mc_put_greeks,
  })

  # 9. Reporting and Plotting
  reporter = ReportingEngine()
  excel_path = reporter.export_excel_results(
      summary_df, greeks_df, put_reg_df=put_reg_df, call_reg_df=call_reg_df
  )
  logging.info(f"Excel report successfully generated: {excel_path}")

  reporter.plot_sample_paths(lsm_paths, n_display=50)

  sim_counts = [100, 500, 1000, 5000, 10000, 50000]
  c_list, p_list = [], []
  for m in sim_counts:
    eval_res = mc_engine.evaluate_european(m, use_sobol=True)
    c_list.append(eval_res["Call_Price"])
    p_list.append(eval_res["Put_Price"])

  reporter.plot_convergence(sim_counts, c_list, p_list, bs_call, bs_put)
  reporter.plot_exercise_boundary(
      res_lsm_put_large["Exercise_Boundary"], mkt_cfg.k
  )

  logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
  run_pipeline()