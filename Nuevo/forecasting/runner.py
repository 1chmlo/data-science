"""Runner / CLI: aplica el pipeline a todos los archivos de un directorio.

Uso:
    python -m forecasting.runner --input-dir datasets_regionales --level region
    python -m forecasting.runner --input-dir datasets_hospitales --level hospital \
        --out-dir resultados --jobs 4

Descubre .parquet y .csv, ejecuta `run_series` en paralelo (joblib) y escribe un
resumen global ordenado por MASE en `<out-dir>/_resumen/`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from .config import PipelineConfig
from .pipeline import SeriesResult, run_series

_INPUT_GLOBS = ("*.parquet", "*.csv")


def discover_inputs(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _INPUT_GLOBS:
        files.extend(sorted(input_dir.glob(pattern)))
    return files


def run_directory(config: PipelineConfig, input_dir: Path) -> pd.DataFrame:
    files = discover_inputs(input_dir)
    if not files:
        raise FileNotFoundError(f"No hay .parquet/.csv en {input_dir}")

    results: list[SeriesResult] = Parallel(n_jobs=config.n_jobs, verbose=10)(
        delayed(run_series)(f, config) for f in files
    )
    summary = _summary_frame(results)
    _write_summary(summary, config)
    return summary


def _summary_frame(results: list[SeriesResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "series_id": r.series_id, "level": r.level, "status": r.status,
            "best_model": r.best_model, "modelo_guardado": r.saved_model,
            "n_weeks": r.n_weeks, "n_folds": r.n_folds,
            "order": r.order, "message": r.message,
        }
        if r.metrics:
            row.update({k: r.metrics[k] for k in ("MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "MASE")})
        rows.append(row)
    df = pd.DataFrame(rows)
    if "MASE" in df.columns:
        df = df.sort_values("MASE", na_position="last").reset_index(drop=True)
    return df


def _write_summary(summary: pd.DataFrame, config: PipelineConfig) -> None:
    out = config.out_dir / "_resumen"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / f"metricas_global_{config.level}.csv", index=False)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        level=args.level,
        out_dir=Path(args.out_dir),
        n_jobs=args.jobs,
        horizon=args.horizon,
        test_weeks=args.test_weeks,
        make_plots=not args.no_plots,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline SARIMAX walk-forward por serie.")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--level", required=True, choices=["region", "hospital"])
    p.add_argument("--out-dir", default="resultados")
    p.add_argument("--jobs", type=int, default=-1, help="Procesos paralelos (joblib).")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--test-weeks", type=int, default=52)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    summary = run_directory(config, args.input_dir)
    ok = (summary["status"] == "ok").sum()
    print(f"\nSeries procesadas: {len(summary)} | ok: {ok}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
