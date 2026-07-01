"""Graficos del pipeline. matplotlib con backend no interactivo (Agg) para correr
en batch/servidores sin display."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402


def plot_forecast(
    history: pd.Series,
    backtest: pd.DataFrame,
    model: str,
    future: pd.DataFrame | None,
    ci_level: str,
    title: str,
    out_path: Path,
) -> None:
    """Historia + backtest (real vs predicho con banda de IC) + pronostico del proximo mes.

    `backtest` debe traer columnas y_true, pred_<model>, lo_<model>, hi_<model>.
    `future` (opcional) trae prediccion, ic_inferior, ic_superior.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    hist_tail = history.iloc[-min(len(history), 156):]
    ax.plot(hist_tail.index, hist_tail.to_numpy(), color="0.6", lw=1, label="Historia")

    bt = backtest
    ax.plot(bt.index, bt["y_true"], color="black", lw=1.6, label="Real (backtest)")
    ax.plot(bt.index, bt[f"pred_{model}"], color="tab:red", lw=1.5, ls="--",
            label=f"Prediccion backtest ({model})")
    if f"lo_{model}" in bt.columns:
        ax.fill_between(bt.index, bt[f"lo_{model}"], bt[f"hi_{model}"],
                        color="tab:red", alpha=0.15, label=f"IC {ci_level}")

    if future is not None and len(future):
        ax.axvline(history.index[-1], color="0.4", lw=1, ls=":")
        ax.plot(future.index, future["prediccion"], color="tab:blue", lw=2,
                marker="o", ms=4, label="Pronostico proximo mes")
        ax.fill_between(future.index, future["ic_inferior"], future["ic_superior"],
                        color="tab:blue", alpha=0.20, label=f"IC {ci_level} (futuro)")

    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("NumTotal")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_residual_diagnostics(residuals: np.ndarray, title: str, out_path: Path) -> None:
    """Panel 2x2: residuos en el tiempo, histograma, ACF y PACF."""
    resid = np.asarray(residuals, dtype="float64")
    resid = resid[np.isfinite(resid)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(resid, color="tab:blue", lw=0.8)
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set_title("Residuos en el tiempo")

    axes[0, 1].hist(resid, bins=30, color="tab:blue", alpha=0.8)
    axes[0, 1].set_title("Histograma de residuos")

    max_lags = min(40, len(resid) // 2 - 1)
    if max_lags >= 1:
        plot_acf(resid, lags=max_lags, ax=axes[1, 0])
        plot_pacf(resid, lags=max_lags, ax=axes[1, 1], method="ywm")
    axes[1, 0].set_title("ACF de residuos")
    axes[1, 1].set_title("PACF de residuos")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
