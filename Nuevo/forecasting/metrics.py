"""Metricas de error para forecasting, con guia de cuando usar cada una.

- MAE   : error medio absoluto, en las unidades del target. Robusto e interpretable.
- RMSE  : penaliza mas los errores grandes; util si los picos importan mucho.
- MAPE  : error porcentual; intuitivo pero se dispara/indefine con valores ~0.
- sMAPE : version simetrica y acotada de MAPE; mejor con ceros que MAPE.
- WAPE  : |error| total / demanda total; la mas estable para series intermitentes.
- MASE  : error escalado por el naive estacional del TRAIN. <1 = mejor que el
          baseline; es la unica comparable entre series de escalas distintas.

Recomendacion: elegir el mejor modelo por MASE (comparabilidad) y reportar
ademas WAPE y MAE para lectura operativa.
"""

from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype="float64")


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_arr(y_true) - _arr(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((_arr(y_true) - _arr(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def wape(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100.0)


def mase(y_true, y_pred, train_target, season: int) -> float:
    """MASE escalado por el naive estacional ajustado en el train de la serie."""
    train = _arr(train_target)
    if len(train) > season:
        scale = np.mean(np.abs(train[season:] - train[:-season]))
    else:  # fallback a naive de lag 1 si no hay un ciclo estacional completo
        scale = np.mean(np.abs(np.diff(train))) if len(train) > 1 else np.nan
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return float(mae(y_true, y_pred) / scale)


def compute_all(y_true, y_pred, train_target, season: int) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, train_target, season),
    }
