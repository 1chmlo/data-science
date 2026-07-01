"""Construccion de regresores exogenos: terminos de Fourier + meteo estandarizada.

Los terminos de Fourier son funciones deterministas del tiempo -> no producen
fuga de informacion y se pueden precalcular sobre toda la serie. La meteo, en
cambio, se estandariza con parametros ajustados SOLO en el tramo de entrenamiento
de cada fold (ver `scale_meteo`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def fourier_terms(
    index: pd.DatetimeIndex, week_of_year: np.ndarray, config: PipelineConfig
) -> pd.DataFrame:
    """Genera sin/cos anuales cuya fase es la semana-del-anio (repite cada anio,
    tolerando anios de 52 o 53 semanas sin acumular desfase)."""
    phase = (np.asarray(week_of_year, dtype="float64") - 1.0)
    cols: dict[str, np.ndarray] = {}
    for k in range(1, config.fourier_terms + 1):
        ang = 2.0 * np.pi * k * phase / config.fourier_period
        cols[f"fourier_sin_{k}"] = np.sin(ang)
        cols[f"fourier_cos_{k}"] = np.cos(ang)
    return pd.DataFrame(cols, index=index)


def build_exog(
    frame: pd.DataFrame, meteo: list[str], week_of_year: np.ndarray, config: PipelineConfig
) -> pd.DataFrame:
    """Matriz exogena completa (Fourier + meteo cruda) alineada al indice de la serie.

    La estandarizacion de la meteo se aplica despues, por fold, en `scale_meteo`.
    """
    exog = fourier_terms(frame.index, week_of_year, config)
    if meteo:
        exog = pd.concat([exog, frame[meteo]], axis=1)
    return exog


def meteo_stats(
    exog_train: pd.DataFrame, meteo: list[str], config: PipelineConfig
) -> tuple[tuple[pd.Series, pd.Series] | None, list[str]]:
    """Media/desv de la meteo calculadas SOLO con el train (sin data leakage)."""
    present = [c for c in meteo if c in exog_train.columns]
    if not config.standardize_exog or not present:
        return None, present
    mean = exog_train[present].mean()
    std = exog_train[present].std(ddof=0).replace(0.0, 1.0)
    return (mean, std), present


def apply_meteo_scale(
    exog: pd.DataFrame, stats: tuple[pd.Series, pd.Series] | None, present: list[str]
) -> pd.DataFrame:
    """Aplica un scaler ya ajustado. Los Fourier no se tocan (ya en [-1,1])."""
    if stats is None or not present:
        return exog.copy()
    mean, std = stats
    out = exog.copy()
    out[present] = (out[present] - mean) / std
    return out


def scale_meteo(
    exog_train: pd.DataFrame,
    exog_full: pd.DataFrame,
    meteo: list[str],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estandariza (train, full) con parametros del TRAIN. Conveniencia para los folds."""
    stats, present = meteo_stats(exog_train, meteo, config)
    return apply_meteo_scale(exog_train, stats, present), apply_meteo_scale(exog_full, stats, present)
