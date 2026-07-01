"""Orquestacion end-to-end de UNA serie temporal.

    load_and_prepare -> folds -> (por fold) fit SARIMAX + baseline -> predict
    -> metricas -> eleccion de modelo -> diagnostico -> pronostico futuro -> guardar.

Todo el manejo de exogenas respeta la separacion train/test de cada fold para no
introducir data leakage (el scaler de la meteo se ajusta solo con el train).

Se comparan un baseline (seasonal_naive) y dos SARIMAX (con y sin meteo). El
CAMPEON reportado puede ser el baseline (transparencia), pero el modelo que se
GUARDA (.pkl), del que salen los intervalos de confianza y el pronostico del
proximo mes, es siempre el mejor SARIMAX: el entregable del proyecto es un SARIMAX.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import diagnostics, features, metrics, models, plots
from .config import PipelineConfig
from .data import SeriesData, load_and_prepare
from .validation import Fold, expanding_walk_forward, has_enough_data

# Modelos SARIMAX que se comparan y que exogenas usa cada uno.
_MODEL_EXOG = {
    "arima_fourier": "fourier",   # solo estacionalidad de Fourier
    "sarimax": "all",             # Fourier + meteo
}


@dataclass
class SeriesResult:
    series_id: str
    level: str
    status: str                       # "ok" | "skipped" | "error"
    n_weeks: int
    n_folds: int
    best_model: str | None = None     # campeon del backtest (incluye baseline)
    saved_model: str | None = None    # SARIMAX guardado en .pkl
    metrics: dict | None = None       # metricas del campeon
    order: list[int] | None = None
    message: str = ""


def run_series(path: str | Path, config: PipelineConfig) -> SeriesResult:
    """Ejecuta el pipeline completo sobre un archivo y persiste los resultados."""
    try:
        data = load_and_prepare(path, config)
    except Exception as exc:  # noqa: BLE001 - un archivo malo no debe frenar el lote
        return SeriesResult(Path(path).stem, config.level, "error", 0, 0, message=str(exc))

    if not has_enough_data(data.n_weeks, config):
        return SeriesResult(
            data.series_id, config.level, "skipped", data.n_weeks, 0,
            message=f"Serie demasiado corta (<{config.min_train_weeks + 1} semanas).",
        )

    folds = expanding_walk_forward(data.n_weeks, config)
    backtest, per_model_metrics, orders = _run_backtest(data, folds, config)

    best_model = _choose_best(per_model_metrics)             # campeon (incl. baseline)
    saved_model = _choose_best_sarimax(per_model_metrics)    # SARIMAX a guardar
    final = _fit_final_and_diagnose(data, saved_model, orders, config)

    _persist(data, config, folds, backtest, per_model_metrics, orders,
             best_model, saved_model, final)

    return SeriesResult(
        series_id=data.series_id,
        level=config.level,
        status="ok",
        n_weeks=data.n_weeks,
        n_folds=len(folds),
        best_model=best_model,
        saved_model=saved_model,
        metrics=per_model_metrics[best_model],
        order=list(orders[saved_model].as_tuple()),
        message="; ".join(data.warnings),
    )


# --------------------------------------------------------------------------- #
# Backtest walk-forward (con intervalos de confianza para los SARIMAX)
# --------------------------------------------------------------------------- #
def _run_backtest(
    data: SeriesData, folds: list[Fold], config: PipelineConfig
) -> tuple[pd.DataFrame, dict[str, dict], dict[str, models.Order]]:
    frame = data.frame
    raw_target = frame[data.target]
    endog_full = np.log1p(raw_target) if config.log_transform else raw_target.astype("float64")

    exog_all = features.build_exog(frame, data.exog_meteo, data.week_of_year, config)
    exog_by_kind = {
        "fourier": exog_all[config.fourier_regressor_names()],
        "all": exog_all,
    }
    orders = _select_orders(endog_full, exog_by_kind, folds[0], data, config)

    rows: list[dict] = []
    for fold in folds:
        test_idx = frame.index[fold.test_start:fold.test_end]
        y_true = raw_target.iloc[fold.test_start:fold.test_end].to_numpy()
        train_raw = raw_target.iloc[:fold.train_end].to_numpy()

        naive = models.seasonal_naive_forecast(train_raw, len(test_idx), config.season)
        sarimax_pred = {
            name: _fit_predict_fold(
                endog_full, exog_by_kind[kind], data.exog_meteo, orders[name], fold, config
            )
            for name, kind in _MODEL_EXOG.items()
        }

        for offset, date in enumerate(test_idx):
            row = {"fecha": date, "fold": fold.index, "y_true": float(y_true[offset]),
                   "pred_seasonal_naive": float(naive[offset])}
            for name, (mean, lo, hi) in sarimax_pred.items():
                row[f"pred_{name}"] = float(mean[offset])
                row[f"lo_{name}"] = float(lo[offset])
                row[f"hi_{name}"] = float(hi[offset])
            rows.append(row)

    backtest = pd.DataFrame(rows).set_index("fecha")
    per_model = _score_models(backtest, raw_target, folds, config)
    return backtest, per_model, orders


def _fit_predict_fold(
    endog_full: pd.Series,
    exog_full: pd.DataFrame,
    meteo: list[str],
    order: models.Order,
    fold: Fold,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta un SARIMAX en el train del fold y predice el test con IC."""
    endog_train = endog_full.iloc[:fold.train_end]
    exog_train, exog_full_scaled = features.scale_meteo(
        exog_full.iloc[:fold.train_end], exog_full, meteo, config
    )
    exog_test = exog_full_scaled.iloc[fold.test_start:fold.test_end]

    try:
        res = models.fit_sarimax(endog_train, exog_train, order, config)
        mean, lo, hi = models.forecast_with_ci(res, len(exog_test), exog_test, config.ci_alpha)
    except Exception:  # noqa: BLE001 - fold no convergente -> se degrada a naive (sin IC)
        naive = models.seasonal_naive_forecast(
            _inverse(endog_train.to_numpy(), config), len(exog_test), config.season
        )
        nan = np.full(len(exog_test), np.nan)
        return naive, nan, nan

    return (_clip_counts(_inverse(mean, config)),
            _clip_counts(_inverse(lo, config)),
            _clip_counts(_inverse(hi, config)))


def _select_orders(
    endog_full: pd.Series,
    exog_by_kind: dict[str, pd.DataFrame],
    first: Fold,
    data: SeriesData,
    config: PipelineConfig,
) -> dict[str, models.Order]:
    endog_train = endog_full.iloc[:first.train_end]
    orders: dict[str, models.Order] = {}
    for name, kind in _MODEL_EXOG.items():
        exog_train, _ = features.scale_meteo(
            exog_by_kind[kind].iloc[:first.train_end], exog_by_kind[kind],
            data.exog_meteo, config,
        )
        orders[name] = models.select_order(endog_train, exog_train, config)
    return orders


# --------------------------------------------------------------------------- #
# Scoring y eleccion de modelo
# --------------------------------------------------------------------------- #
def _score_models(
    backtest: pd.DataFrame, raw_target: pd.Series, folds: list[Fold], config: PipelineConfig
) -> dict[str, dict]:
    train_ref = raw_target.iloc[: folds[0].train_end].to_numpy()
    y_true = backtest["y_true"].to_numpy()
    scores: dict[str, dict] = {}
    for col in backtest.columns:
        if not col.startswith("pred_"):
            continue
        name = col[len("pred_"):]
        scores[name] = metrics.compute_all(y_true, backtest[col].to_numpy(), train_ref, config.season)
    return scores


def _mase_wape(m: dict) -> tuple[float, float]:
    mase = m["MASE"] if np.isfinite(m["MASE"]) else float("inf")
    wape = m["WAPE"] if np.isfinite(m["WAPE"]) else float("inf")
    return (mase, wape)


def _choose_best(per_model: dict[str, dict]) -> str:
    """Campeon del backtest por MASE (empate -> WAPE); incluye el baseline."""
    return min(per_model, key=lambda n: _mase_wape(per_model[n]))


def _choose_best_sarimax(per_model: dict[str, dict]) -> str:
    """Mejor de los dos SARIMAX (con/sin meteo). Es el modelo que se guarda."""
    cands = {n: per_model[n] for n in _MODEL_EXOG if n in per_model}
    return min(cands, key=lambda n: _mase_wape(cands[n]))


# --------------------------------------------------------------------------- #
# Modelo final (SARIMAX) + diagnostico + pronostico futuro con IC
# --------------------------------------------------------------------------- #
def _fit_final_and_diagnose(
    data: SeriesData, saved_model: str, orders: dict[str, models.Order], config: PipelineConfig
) -> dict:
    """Reajusta el mejor SARIMAX sobre toda la serie: residuos, .pkl y futuro con IC."""
    frame = data.frame
    raw_target = frame[data.target]
    endog_full = np.log1p(raw_target) if config.log_transform else raw_target.astype("float64")

    exog_all = features.build_exog(frame, data.exog_meteo, data.week_of_year, config)
    kind = _MODEL_EXOG[saved_model]
    exog = exog_all if kind == "all" else exog_all[config.fourier_regressor_names()]

    stats, present = features.meteo_stats(exog, data.exog_meteo, config)
    exog_scaled = features.apply_meteo_scale(exog, stats, present)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = models.fit_sarimax(endog_full, exog_scaled, orders[saved_model], config)
        resid = np.asarray(result.resid, dtype="float64")
        future, note = _future_forecast(data, result, kind, stats, present, config)
    except Exception as exc:  # noqa: BLE001 - si el fit final falla, seguimos sin .pkl/futuro
        return {"result": None, "residuals": np.array([]), "order": None,
                "future": None, "future_note": f"fit final fallo: {exc}"}

    return {"result": result, "residuals": resid,
            "order": orders[saved_model].as_tuple(), "future": future, "future_note": note}


def _future_forecast(
    data: SeriesData, result, kind: str,
    stats, present: list[str], config: PipelineConfig,
) -> tuple[pd.DataFrame, str | None]:
    """Pronostica las proximas `horizon` semanas (proximo mes) con IC."""
    h = config.horizon
    last_date = data.frame.index[-1]
    future_index = pd.date_range(start=last_date, periods=h + 1, freq="W-MON")[1:]
    future_woy = _next_weeks(int(data.week_of_year[-1]), h)

    exog_future = features.fourier_terms(future_index, future_woy, config)
    note = None
    if kind == "all":
        clim = _meteo_climatology(data, future_woy, future_index)
        exog_future = pd.concat([exog_future, features.apply_meteo_scale(clim, stats, present)], axis=1)
        note = "meteo futura estimada con climatologia (promedio por semana-del-anio)."

    mean, lo, hi = models.forecast_with_ci(result, h, exog_future, config.ci_alpha)
    future = pd.DataFrame(
        {
            "semana": future_woy,
            "prediccion": _clip_counts(_inverse(mean, config)),
            "ic_inferior": _clip_counts(_inverse(lo, config)),
            "ic_superior": _clip_counts(_inverse(hi, config)),
        },
        index=future_index,
    )
    future.index.name = "fecha"
    return future, note


def _next_weeks(last_woy: int, horizon: int, max_week: int = 52) -> np.ndarray:
    out, w = [], last_woy
    for _ in range(horizon):
        w = 1 if w >= max_week else w + 1
        out.append(w)
    return np.array(out)


def _meteo_climatology(data: SeriesData, future_woy: np.ndarray, future_index) -> pd.DataFrame:
    """Promedio historico de cada meteo por semana-del-anio para las semanas futuras."""
    tmp = data.frame[data.exog_meteo].copy()
    tmp["_woy"] = data.week_of_year
    clim = tmp.groupby("_woy")[data.exog_meteo].mean()
    rows = clim.reindex(future_woy).ffill().bfill()
    rows.index = future_index
    return rows


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def _persist(
    data: SeriesData,
    config: PipelineConfig,
    folds: list[Fold],
    backtest: pd.DataFrame,
    per_model: dict[str, dict],
    orders: dict[str, models.Order],
    best_model: str,
    saved_model: str,
    final: dict,
) -> None:
    out = config.series_out_dir(data.series_id)
    (out / "graficos").mkdir(parents=True, exist_ok=True)

    backtest.to_csv(out / "predicciones.csv")
    if final["future"] is not None:
        final["future"].to_csv(out / "prediccion_futura.csv")

    if len(final["residuals"]):
        pd.Series(final["residuals"], index=data.frame.index, name="residuo").to_csv(
            out / "residuos.csv"
        )

    diag = diagnostics.residual_report(final["residuals"]) if len(final["residuals"]) else {}
    metricas = {
        "series_id": data.series_id,
        "level": config.level,
        "best_model": best_model,                 # campeon del backtest (incl. baseline)
        "modelo_guardado": saved_model,           # SARIMAX en modelo.pkl
        "orders": {k: list(v.as_tuple()) for k, v in orders.items()},
        "metrics_por_modelo": per_model,
        "metrics_campeon": per_model[best_model],
        "metrics_modelo_guardado": per_model[saved_model],
        "prediccion_futura": _future_dict(final["future"]),
        "nota_futuro": final["future_note"],
        "ic_nivel": f"{int((1 - config.ci_alpha) * 100)}%",
        "diagnostico_residuos": diag,
        "n_weeks": data.n_weeks,
        "n_folds": len(folds),
        "exogenas_meteo": data.exog_meteo,
        "target_gaps_imputados": data.n_target_gaps_filled,
        "exog_gaps_imputados": data.exog_gaps_filled,
        "warnings": data.warnings,
        "config": _config_snapshot(config),
    }
    (out / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    if final["result"] is not None:
        joblib.dump(final["result"], out / "modelo.pkl")

    if config.make_plots and len(final["residuals"]):
        title = f"[{config.level}] {data.series_id} - guardado: {saved_model} (campeon: {best_model})"
        plots.plot_forecast(
            history=data.frame[data.target],
            backtest=backtest,
            model=saved_model,
            future=final["future"],
            ci_level=metricas["ic_nivel"],
            title=title,
            out_path=out / "graficos" / "forecast_vs_real.png",
        )
        plots.plot_residual_diagnostics(
            final["residuals"], f"Diagnostico residuos - {title}",
            out / "graficos" / "residuos_diagnostico.png",
        )


def _future_dict(future: pd.DataFrame | None) -> list[dict] | None:
    if future is None:
        return None
    return [
        {"fecha": str(idx.date()), "semana": int(r["semana"]),
         "prediccion": round(float(r["prediccion"]), 2),
         "ic_inferior": round(float(r["ic_inferior"]), 2),
         "ic_superior": round(float(r["ic_superior"]), 2)}
        for idx, r in future.iterrows()
    ]


def _config_snapshot(config: PipelineConfig) -> dict:
    d = asdict(config)
    d["out_dir"] = str(config.out_dir)
    return d


# --------------------------------------------------------------------------- #
# Utilidades de transformacion
# --------------------------------------------------------------------------- #
def _inverse(values: np.ndarray, config: PipelineConfig) -> np.ndarray:
    return np.expm1(values) if config.log_transform else np.asarray(values, dtype="float64")


def _clip_counts(values: np.ndarray) -> np.ndarray:
    return np.clip(values, a_min=0.0, a_max=None)
