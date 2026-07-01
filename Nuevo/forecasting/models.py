"""Modelos: baselines naive, seleccion de orden ARIMA y ajuste/prediccion SARIMAX.

La estacionalidad anual la aportan los terminos de Fourier (exogenos), por eso el
componente estacional de SARIMAX se deja en (0,0,0,0): asi evitamos el costoso e
inestable s=52. `d` se fija por diferenciacion estacionaria (KPSS) y (p,q) por
mini-grid AICc.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

from .config import PipelineConfig


@dataclass(frozen=True)
class Order:
    p: int
    d: int
    q: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.p, self.d, self.q)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def seasonal_naive_forecast(train: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Repite el valor de hace `season` semanas (fallback: ultimo valor)."""
    train = np.asarray(train, dtype="float64")
    if len(train) >= season:
        base = train[-season:]
        return np.array([base[i % season] for i in range(horizon)])
    last = train[-1] if len(train) else 0.0
    return np.full(horizon, last)


def seasonal_naive_insample(y: np.ndarray, season: int) -> np.ndarray:
    """Ajuste in-sample del naive estacional: y_hat[t] = y[t-season].

    Los primeros `season` puntos se igualan al valor observado (residuo 0) para no
    contaminar el diagnostico con el arranque.
    """
    y = np.asarray(y, dtype="float64")
    fitted = y.copy()
    if len(y) > season:
        fitted[season:] = y[:-season]
    return fitted


# --------------------------------------------------------------------------- #
# Seleccion de orden
# --------------------------------------------------------------------------- #
def choose_d(y: np.ndarray, d_max: int) -> int:
    """Numero de diferencias por KPSS (rechazo de estacionariedad -> diferenciar)."""
    series = np.asarray(y, dtype="float64")
    for d in range(d_max + 1):
        if len(series) < 10:
            return d
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pvalue, *_ = kpss(series, regression="c", nlags="auto")
        except (ValueError, OverflowError):
            return d
        if pvalue > 0.05:      # no se rechaza estacionariedad -> suficiente
            return d
        series = np.diff(series)
    return d_max


def _aicc(aic: float, nobs: float, k: float) -> float:
    """AICc: AIC corregido para muestras finitas."""
    denom = nobs - k - 1
    if denom <= 0:
        return float("inf")
    return aic + (2.0 * k * (k + 1.0)) / denom


def select_order(
    endog: pd.Series, exog: pd.DataFrame | None, config: PipelineConfig
) -> Order:
    """Elige (p,d,q) minimizando AICc sobre el mini-grid configurado."""
    d = choose_d(endog.to_numpy(), config.d_max)
    best_order = Order(1, d, 1)
    best_aicc = float("inf")

    for p in config.p_range:
        for q in config.q_range:
            if p == 0 and q == 0:
                continue
            try:
                res = _fit(endog, exog, Order(p, d, q), config)
            except (np.linalg.LinAlgError, ValueError):
                continue
            aicc = _aicc(res.aic, nobs=res.nobs, k=res.df_model)
            if np.isfinite(aicc) and aicc < best_aicc:
                best_aicc = aicc
                best_order = Order(p, d, q)
    return best_order


# --------------------------------------------------------------------------- #
# Ajuste / prediccion
# --------------------------------------------------------------------------- #
def _fit(endog: pd.Series, exog: pd.DataFrame | None, order: Order, config: PipelineConfig):
    trend = "c" if order.d == 0 else "n"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model = SARIMAX(
            endog,
            exog=exog,
            order=order.as_tuple(),
            seasonal_order=config.seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=200)


def fit_sarimax(
    endog: pd.Series, exog: pd.DataFrame | None, order: Order, config: PipelineConfig
):
    """Ajusta SARIMAX con el orden dado (re-estima coeficientes)."""
    return _fit(endog, exog, order, config)


def forecast(result, horizon: int, exog_future: pd.DataFrame | None) -> np.ndarray:
    """Prediccion puntual a `horizon` pasos (en el mismo espacio del endog)."""
    fc = result.get_forecast(steps=horizon, exog=exog_future)
    return np.asarray(fc.predicted_mean, dtype="float64")


def forecast_with_ci(
    result, horizon: int, exog_future: pd.DataFrame | None, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prediccion puntual + intervalo de confianza (1-alpha) a `horizon` pasos.

    Devuelve (media, inferior, superior) en el mismo espacio del endog. Como la
    transformacion log1p/expm1 es monotona, el orden inferior<superior se conserva
    al revertir.
    """
    fc = result.get_forecast(steps=horizon, exog=exog_future)
    mean = np.asarray(fc.predicted_mean, dtype="float64")
    ci = np.asarray(fc.conf_int(alpha=alpha), dtype="float64")
    return mean, ci[:, 0], ci[:, 1]
