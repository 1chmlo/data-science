"""Carga, agregacion a serie temporal y limpieza de faltantes.

Un archivo de entrada contiene muchas filas (establecimiento x causa x semana).
Aqui se colapsa a UNA serie semanal por SUMA de `NumTotal`, con un indice de
fechas semanal uniforme (lunes, freq 'W-MON'). Las meteo se toman por media (son
constantes dentro de una region-semana, asi que la media == el valor real).

Nota sobre el tiempo: `SemanaEstadistica` es la semana EPIDEMIOLOGICA de MINSAL
(algunos anios tienen 53 semanas), no la semana ISO. Por eso el indice de fechas
se construye por POSICION (lunes consecutivos) en vez de parsear un calendario
fragil, y el ciclo estacional se modela con la semana-del-anio (ver features).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig, WEEK_COL, YEAR_COL


@dataclass
class SeriesData:
    """Serie lista para modelar mas metadatos de trazabilidad."""

    series_id: str
    frame: pd.DataFrame          # indice DatetimeIndex 'W-MON'; col target + meteo
    week_of_year: np.ndarray     # SemanaEstadistica por fila (fase estacional)
    target: str
    exog_meteo: list[str]
    n_weeks: int
    n_target_gaps_filled: int
    exog_gaps_filled: dict[str, int]
    warnings: list[str]


def _read_any(path: Path, columns: list[str]) -> pd.DataFrame:
    """Lee solo las columnas necesarias (column pushdown) para acotar memoria/tiempo.

    Los archivos regionales tienen ~850k filas x 32 columnas, pero el pipeline solo
    usa 7; leer el resto dispara el uso de RAM al paralelizar varias series.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"Formato no soportado: {path.suffix} ({path})")


def _coerce_numeric(s: pd.Series) -> pd.Series:
    """Meteo con placeholders de texto ('Vacio: ...') -> NaN numerico."""
    return pd.to_numeric(s, errors="coerce")


def load_and_prepare(path: str | Path, config: PipelineConfig) -> SeriesData:
    """Lee un archivo y devuelve la serie semanal lista para el pipeline."""
    path = Path(path)
    keep_meteo = [c for c in config.meteo_exog if c not in set(config.drop_exog)]
    needed = [YEAR_COL, WEEK_COL, config.target, *keep_meteo]
    raw = _read_any(path, columns=needed)

    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise KeyError(f"Faltan columnas {missing} en {path.name}")

    work = pd.DataFrame({
        "anio": raw[YEAR_COL].astype("int64"),
        "semana": raw[WEEK_COL].astype("int64"),
        config.target: pd.to_numeric(raw[config.target], errors="coerce"),
    })
    for col in keep_meteo:
        work[col] = _coerce_numeric(raw[col])

    # Agregacion a serie: SUMA del target, MEDIA de meteo (constante por semana).
    agg = {config.target: "sum", **{c: "mean" for c in keep_meteo}}
    series = (
        work.groupby(["anio", "semana"], as_index=False)
        .agg(agg)
        .sort_values(["anio", "semana"])
        .reset_index(drop=True)
    )

    tail_warns: list[str] = []
    series = _trim_incomplete_tail(series, config, tail_warns, path.name)

    week_of_year = series["semana"].to_numpy()
    index = _weekly_index(int(series["anio"].iloc[0]), len(series))
    frame = series[[config.target, *keep_meteo]].set_axis(index, axis=0)
    frame.index.name = "fecha"

    frame, target_gaps, exog_gaps, warns = _fill_gaps(frame, config, keep_meteo, path.name)
    warns = tail_warns + warns

    return SeriesData(
        series_id=_series_id_from_path(path),
        frame=frame,
        week_of_year=week_of_year,
        target=config.target,
        exog_meteo=keep_meteo,
        n_weeks=len(frame),
        n_target_gaps_filled=target_gaps,
        exog_gaps_filled=exog_gaps,
        warnings=warns,
    )


def _trim_incomplete_tail(
    series: pd.DataFrame, config: PipelineConfig, warns: list[str], fname: str
) -> pd.DataFrame:
    """Descarta el final espurio de la serie (semana fuera de secuencia + semanas parciales).

    (1) En el ultimo anio, corta todo lo que venga tras el primer salto de semana
        (p. ej. una semana 53 que aparece junto a semanas 1..18). Los anios de 53
        semanas completos son contiguos, asi que no se tocan.
    (2) Quita semanas finales cuyo total cae por debajo de `partial_week_frac` de la
        mediana de las 8 previas: son cortes de datos incompletos que sesgarian el
        pronostico del proximo mes.
    """
    s = series.reset_index(drop=True)
    target = config.target

    last_year = int(s["anio"].max())
    ly_pos = s.index[s["anio"] == last_year].to_numpy()
    weeks = s.loc[ly_pos, "semana"].to_numpy()
    cut = len(weeks)
    for i in range(1, len(weeks)):
        if weeks[i] != weeks[i - 1] + 1:
            cut = i
            break
    if cut < len(weeks):
        drop_pos = ly_pos[cut:]
        dropped = s.loc[drop_pos, "semana"].tolist()
        warns.append(
            f"{fname}: en {last_year} se descartan semanas fuera de secuencia {dropped} "
            f"(artefacto, p. ej. semana 53 espuria)."
        )
        s = s.drop(index=drop_pos).reset_index(drop=True)

    while len(s) > 9:
        recent = float(s[target].iloc[-9:-1].median())
        last_val = float(s[target].iloc[-1])
        if recent > 0 and last_val < config.partial_week_frac * recent:
            wk = int(s["semana"].iloc[-1])
            warns.append(
                f"{fname}: descartada semana final {wk} por incompleta "
                f"(total {last_val:.0f} < {config.partial_week_frac:.0%} de la mediana reciente {recent:.0f})."
            )
            s = s.iloc[:-1].reset_index(drop=True)
        else:
            break
    return s


def _weekly_index(first_year: int, n: int) -> pd.DatetimeIndex:
    """Indice uniforme de `n` lunes consecutivos desde el primer lunes del anio.

    Es un calendario semanal regular (freq 'W-MON') que preserva el orden y el
    espaciado de 7 dias. No pretende clavar la fecha calendaria exacta de cada
    semana epidemiologica (irrelevante para el modelo: la estacionalidad va por
    semana-del-anio), solo dar un eje temporal valido y monotono.
    """
    return pd.date_range(start=f"{first_year}-01-01", periods=n, freq="W-MON")


def _fill_gaps(
    series: pd.DataFrame,
    config: PipelineConfig,
    meteo: list[str],
    fname: str,
) -> tuple[pd.DataFrame, int, dict[str, int], list[str]]:
    """Interpola huecos internos y rellena bordes; registra cuanto se imputo."""
    warns: list[str] = []
    out = series.copy()

    target_gaps = int(out[config.target].isna().sum())
    _warn_long_gaps(out[config.target], config, "target", fname, warns)
    out[config.target] = (
        out[config.target].interpolate(method="time", limit_direction="both").ffill().bfill()
    )

    exog_gaps: dict[str, int] = {}
    for col in meteo:
        exog_gaps[col] = int(out[col].isna().sum())
        _warn_long_gaps(out[col], config, col, fname, warns)
        out[col] = out[col].interpolate(method="time", limit_direction="both").ffill().bfill()

    return out, target_gaps, exog_gaps, warns


def _warn_long_gaps(
    col: pd.Series, config: PipelineConfig, name: str, fname: str, warns: list[str]
) -> None:
    longest = _longest_run(col.isna().to_numpy())
    if longest > config.max_gap_weeks:
        warns.append(
            f"{fname}: hueco de {longest} semanas en '{name}' "
            f"(> {config.max_gap_weeks}); imputacion poco fiable en ese tramo."
        )


def _longest_run(mask) -> int:
    longest = run = 0
    for v in mask:
        run = run + 1 if v else 0
        longest = max(longest, run)
    return longest


def _series_id_from_path(path: Path) -> str:
    stem = path.stem
    for prefix in ("dataset_hospital_", "dataset_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem
