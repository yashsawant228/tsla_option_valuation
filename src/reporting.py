"""Automated Excel report writer and publication-quality plot generator."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ReportingEngine:
  """Handles structured Excel exports and visual figure generation."""

  def __init__(
      self, output_dir: str = "output", fig_dir: str = "output/figures"
  ) -> None:
    self.output_dir = output_dir
    self.fig_dir = fig_dir
    os.makedirs(self.output_dir, exist_ok=True)
    os.makedirs(self.fig_dir, exist_ok=True)

  def export_excel_results(
      self,
      summary_df: pd.DataFrame,
      greeks_df: pd.DataFrame,
      put_reg_df: pd.DataFrame = None,
      call_reg_df: pd.DataFrame = None,
  ) -> str:
    """Generates formatted Excel spreadsheet with valuation, greeks, and LSM regressions."""
    filepath = os.path.join(
        self.output_dir, "TSLA_Option_Valuation_Results.xlsx"
    )
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
      summary_df.to_excel(writer, sheet_name="Valuation Summary", index=False)
      greeks_df.to_excel(writer, sheet_name="Sensitivity Analysis", index=True)
      if put_reg_df is not None:
        put_reg_df.to_excel(
            writer, sheet_name="LSM Put Regressions", index=False
        )
      if call_reg_df is not None:
        call_reg_df.to_excel(
            writer, sheet_name="LSM Call Regressions", index=False
        )
    return filepath

  def plot_sample_paths(self, paths: np.ndarray, n_display: int = 50) -> str:
    """Plots sample Monte Carlo price paths."""
    plt.figure(figsize=(10, 6))
    time_grid = np.linspace(0, 0.5, paths.shape[1])
    for i in range(min(n_display, paths.shape[0])):
      plt.plot(time_grid, paths[i], lw=0.8, alpha=0.7)
    plt.axhline(
        y=311.21,
        color="black",
        linestyle="--",
        lw=1.5,
        label="Strike (K = $311.21)",
    )
    plt.title("TSLA Monte Carlo Sample Price Paths (M2 Accelerated)")
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Stock Price ($)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")

    filepath = os.path.join(self.fig_dir, "tsla_mc_sample_paths.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath

  def plot_convergence(
      self,
      sim_counts: list,
      call_prices: list,
      put_prices: list,
      bs_c: float,
      bs_p: float,
  ) -> str:
    """Plots QMC vs Plain MC option price convergence across simulation counts."""
    plt.figure(figsize=(10, 6))
    plt.semilogx(
        sim_counts, call_prices, "o-", label="Monte Carlo Call", color="navy"
    )
    plt.semilogx(
        sim_counts, put_prices, "s-", label="Monte Carlo Put", color="darkred"
    )
    plt.axhline(
        y=bs_c,
        color="navy",
        linestyle="--",
        label=f"Black-Scholes Call (${bs_c:.2f})",
    )
    plt.axhline(
        y=bs_p,
        color="darkred",
        linestyle="--",
        label=f"Black-Scholes Put (${bs_p:.2f})",
    )
    plt.title("Option Price Convergence vs Simulation Count (Log Scale)")
    plt.xlabel("Number of Path Simulations (M)")
    plt.ylabel("Option Price ($)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best")

    filepath = os.path.join(self.fig_dir, "tsla_convergence_plot.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath

  def plot_exercise_boundary(self, boundary_dict: dict, k: float) -> str:
    """Plots early exercise boundary S*(t) for American Put."""
    plt.figure(figsize=(10, 6))
    times = sorted(boundary_dict.keys())
    boundary_prices = [boundary_dict[t] for t in times]

    plt.plot(
        times,
        boundary_prices,
        "ro-",
        lw=2.0,
        label="Optimal Exercise Boundary S*(t)",
    )
    plt.axhline(
        y=k, color="black", linestyle="--", label=f"Strike Price K (${k:.2f})"
    )
    plt.fill_between(
        times, 0, boundary_prices, color="red", alpha=0.15, label="Exercise Region"
    )
    plt.title("Longstaff-Schwartz American Put Early Exercise Boundary")
    plt.xlabel("Time Step (Years)")
    plt.ylabel("Critical Stock Price S* ($)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left")

    filepath = os.path.join(self.fig_dir, "tsla_exercise_boundary.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    return filepath