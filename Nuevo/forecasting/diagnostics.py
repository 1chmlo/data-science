"""Diagnostico de residuos del modelo final.

Un buen modelo deja residuos que parecen ruido blanco: sin autocorrelacion, media
cero y varianza estable. Se reportan pruebas formales y su interpretacion.

- Ljung-Box  : H0 = sin autocorrelacion hasta el lag L. p<0.05 -> queda estructura
               sin capturar (revisar p, q o estacionalidad).
- Jarque-Bera: H0 = residuos normales. p<0.05 -> colas/asimetria; los intervalos
               de prediccion pueden quedar mal calibrados (el punto sigue siendo util).
- Media ~ 0  : sesgo del modelo; una media lejos de 0 indica sub/sobreprediccion.
"""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


def residual_report(residuals) -> dict:
    """Bateria de pruebas sobre los residuos. Devuelve valores + interpretacion."""
    resid = np.asarray(residuals, dtype="float64")
    resid = resid[np.isfinite(resid)]
    n = len(resid)
    if n < 8:
        return {"n": n, "insuficiente": True}

    lags = min(10, n // 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
        lb_stat = float(lb["lb_stat"].iloc[0])
        lb_p = float(lb["lb_pvalue"].iloc[0])
        jb_stat, jb_p, skew, kurt = jarque_bera(resid)

    mean = float(np.mean(resid))
    std = float(np.std(resid, ddof=1))

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ljung_box": {"lags": lags, "stat": lb_stat, "pvalue": lb_p},
        "jarque_bera": {"stat": float(jb_stat), "pvalue": float(jb_p),
                        "skew": float(skew), "kurtosis": float(kurt)},
        "interpretacion": {
            "residuos_sin_autocorrelacion": bool(lb_p >= 0.05),
            "residuos_normales": bool(jb_p >= 0.05),
            "sesgo_bajo": bool(abs(mean) < 0.1 * std) if std > 0 else True,
        },
    }
