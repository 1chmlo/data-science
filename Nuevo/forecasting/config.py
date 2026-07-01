"""Configuracion central del pipeline y registro de las decisiones de diseno.

Cada campo del `PipelineConfig` documenta *por que* se eligio ese valor. Cambiar
el comportamiento del pipeline no deberia requerir tocar la logica: basta con
construir un `PipelineConfig` distinto.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Columnas del dataset (esquema fijo, ya validado sobre los parquet reales).
# ---------------------------------------------------------------------------
YEAR_COL = "Anio"
WEEK_COL = "SemanaEstadistica"
TARGET_COL = "NumTotal"

#: Variables meteorologicas candidatas a exogenas.
ALL_METEO = (
    "temp_promedio_semanal",
    "humedad_relativa_promedio_porcentaje",
    "precipitacion_promedio_diario_mm",
    "radiacion_solar_media_MJ_m2",
    "pm2_5_ug_m3",
    "pm10_ug_m3",
)

#: PM2.5 / PM10 se descartan por defecto: en los datos vienen como texto
#: ("Vacio: Registro historico no disponible en modelo CAMS...") para todo el
#: periodo previo a ~2020. Un hueco estructural de ~6 anios rompe SARIMAX; no es
#: un faltante imputable. Se pueden re-activar para un experimento acotado a
#: 2020+ moviendolas a `meteo_exog`.
DROP_EXOG_DEFAULT = ("pm2_5_ug_m3", "pm10_ug_m3")


@dataclass(frozen=True)
class PipelineConfig:
    """Parametros del pipeline. Inmutable para evitar estado compartido entre series."""

    # -- Nivel de la serie -------------------------------------------------
    #: "hospital" o "region". Solo afecta a las rutas de salida y a la etiqueta;
    #: la agregacion (suma de NumTotal por semana) es identica en ambos.
    level: str = "region"

    # -- Variable objetivo -------------------------------------------------
    #: Endogena. NumTotal = total de atenciones de urgencia por semana. Se agrega
    #: por SUMA colapsando causas (y establecimientos, en el nivel regional).
    target: str = TARGET_COL

    #: Estabiliza la varianza de un conteo no negativo. Se modela en log1p y se
    #: revierte con expm1 antes de calcular metricas (siempre en unidades reales).
    log_transform: bool = True

    # -- Variables exogenas ------------------------------------------------
    #: Meteo usada como exogena (sin PM por el hueco CAMS). Se estandarizan
    #: (media/desv) con un scaler ajustado SOLO en train de cada fold (sin fuga).
    meteo_exog: tuple[str, ...] = (
        "temp_promedio_semanal",
        "humedad_relativa_promedio_porcentaje",
        "precipitacion_promedio_diario_mm",
        "radiacion_solar_media_MJ_m2",
    )
    drop_exog: tuple[str, ...] = DROP_EXOG_DEFAULT
    standardize_exog: bool = True

    #: Estacionalidad anual mediante terminos de Fourier en lugar de SARIMA con
    #: s=52 (que es lento e inestable con datos semanales). El periodo es el
    #: promedio real de semanas por anio (365.25/7), lo que absorbe los anios ISO
    #: de 52 y 53 semanas sin discontinuidades. K=2 armonicos = 4 regresores.
    fourier_period: float = 365.25 / 7.0
    fourier_terms: int = 2

    # -- Rolling Window (Walk-Forward expansivo) ---------------------------
    #: Se reservan las ultimas `test_weeks` semanas como backtest. El train
    #: EXPANDE desde el inicio en cada fold (nunca ve el futuro -> sin leakage).
    horizon: int = 4          # h: se predice a 4 semanas (mensual).
    step: int = 4             # folds NO solapados (avanza h semanas por fold).
    test_weeks: int = 52      # ultimo anio como region de evaluacion.
    min_train_weeks: int = 156  # >=3 anios para ver >=3 ciclos estacionales.
    ci_alpha: float = 0.05    # nivel del intervalo de confianza (0.05 -> IC 95%).

    # -- Seleccion de orden ARIMA (p,d,q) ----------------------------------
    #: Estrategia: d se fija por diferenciacion estacionaria (KPSS); (p,q) por
    #: mini-grid minimizando AICc. El orden se elige UNA vez sobre el train
    #: inicial y se RE-ESTIMAN los coeficientes en cada fold (rapido y estable).
    #: No se usa SARIMA estacional: la estacionalidad la aportan los Fourier.
    p_range: tuple[int, ...] = (0, 1, 2, 3)
    q_range: tuple[int, ...] = (0, 1, 2, 3)
    d_max: int = 2
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)

    # -- Modelos a comparar ------------------------------------------------
    #: Se entrenan y comparan tres modelos; se elige el mejor por MASE en backtest:
    #:   - "seasonal_naive": baseline obligatorio (valor de hace `season` semanas).
    #:   - "arima_fourier":  ARIMA + Fourier (sin meteo).
    #:   - "sarimax":        ARIMA + Fourier + meteo (mide el aporte real de la meteo).
    season: int = 52          # baseline estacional: misma semana del anio anterior.

    # -- Limpieza de faltantes --------------------------------------------
    #: Huecos internos cortos se interpolan en el tiempo; los bordes con ffill/bfill.
    #: Si un hueco supera `max_gap_weeks` se registra una advertencia en meta.json.
    max_gap_weeks: int = 6

    #: Limpieza del tramo final: se descartan semanas fuera de secuencia (p. ej. una
    #: semana 53 espuria en un anio incompleto) y semanas finales parciales cuyo total
    #: cae por debajo de esta fraccion de la mediana de las 8 semanas previas (dato de
    #: corte incompleto). Evita que una semana truncada sesgue el pronostico futuro.
    partial_week_frac: float = 0.5

    # -- Ejecucion ---------------------------------------------------------
    n_jobs: int = -1          # joblib: usar todos los nucleos al iterar series.
    random_seed: int = 42
    make_plots: bool = True

    # -- Salidas -----------------------------------------------------------
    out_dir: Path = Path("resultados")

    def fourier_regressor_names(self) -> list[str]:
        names: list[str] = []
        for k in range(1, self.fourier_terms + 1):
            names += [f"fourier_sin_{k}", f"fourier_cos_{k}"]
        return names

    def series_out_dir(self, series_id: str) -> Path:
        return self.out_dir / self.level / _safe_name(series_id)

    def with_level(self, level: str) -> "PipelineConfig":
        return replace(self, level=level)


def _safe_name(name: str) -> str:
    """Nombre de carpeta seguro para cualquier SO (sin espacios ni separadores)."""
    bad = ' /\\:*?"<>|\''
    return "".join("_" if c in bad else c for c in str(name)).strip("_")
