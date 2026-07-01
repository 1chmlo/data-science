"""Walk-Forward Validation expansivo (rolling window sin data leakage).

Se reservan las ultimas `test_weeks` semanas como region de backtest. En cada
fold el conjunto de entrenamiento son TODAS las observaciones anteriores al fold
(ventana expansiva) y se predice el bloque siguiente de `horizon` semanas. Los
folds avanzan de a `step` semanas; con step == horizon quedan no solapados.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import PipelineConfig


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: int          # posiciones [0, train_end) son entrenamiento
    test_start: int
    test_end: int           # posiciones [test_start, test_end) son prueba


def expanding_walk_forward(n_obs: int, config: PipelineConfig) -> list[Fold]:
    """Devuelve la lista de folds para una serie de `n_obs` observaciones."""
    first_test = n_obs - config.test_weeks
    if first_test < config.min_train_weeks:
        # Serie corta: arrancar el backtest en cuanto haya train minimo.
        first_test = config.min_train_weeks

    folds: list[Fold] = []
    cur = first_test
    idx = 0
    while cur < n_obs:
        test_end = min(cur + config.horizon, n_obs)
        if test_end <= cur:
            break
        folds.append(Fold(index=idx, train_end=cur, test_start=cur, test_end=test_end))
        idx += 1
        cur += config.step
    return folds


def has_enough_data(n_obs: int, config: PipelineConfig) -> bool:
    """True si la serie permite al menos un fold con train minimo."""
    return n_obs > config.min_train_weeks + 1
