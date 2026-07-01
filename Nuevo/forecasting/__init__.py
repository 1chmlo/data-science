"""Pipeline modular de forecasting SARIMAX con validacion temporal walk-forward.

Flujo (identico para el nivel `hospital` y el nivel `region`):

    CSV/Parquet listo
        -> Seleccion de variables      (data.load_and_prepare)
        -> Rolling Window expansivo     (validation.expanding_walk_forward)
        -> Entrenamiento SARIMAX        (models.fit_sarimax)
        -> Prediccion                   (models.forecast)
        -> Evaluacion                   (metrics.compute_all)
        -> Diagnostico de residuos      (diagnostics.residual_report)
        -> Guardar resultados           (pipeline.run_series)

Las decisiones de diseno estan documentadas en `config.py` y en README_forecasting.md.
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
