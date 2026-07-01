# Pipeline de Forecasting SARIMAX — Urgencias Hospitalarias

Pipeline modular y reutilizable para pronosticar la demanda semanal de atenciones
de urgencia (`NumTotal`) por **hospital** o por **región**, con validación temporal
walk-forward, variables meteorológicas exógenas y diagnóstico de residuos.

```
CSV/Parquet listo
   → Selección de variables      (data.load_and_prepare)
   → Rolling Window expansivo     (validation.expanding_walk_forward)
   → Entrenamiento SARIMAX        (models.fit_sarimax)
   → Predicción                   (models.forecast)
   → Evaluación                   (metrics.compute_all)
   → Diagnóstico de residuos      (diagnostics.residual_report)
   → Guardar resultados           (pipeline.run_series)
```

---

## 1. Instalación y ejecución

```bash
# 1. Entorno (una vez)
bash setup_env.sh
source .venv/bin/activate

# 2. (si aún no existen) generar los datasets por serie
jupyter nbconvert --to notebook --execute Division_dataset.ipynb   # o correrlo a mano

# 3. Ejecutar el pipeline sobre TODAS las series de un nivel
python -m forecasting.runner --input-dir datasets_regionales  --level region
python -m forecasting.runner --input-dir datasets_hospitales --level hospital --jobs 4
```

Parámetros del CLI: `--out-dir` (def. `resultados`), `--jobs` (procesos paralelos,
`-1` = todos los núcleos), `--horizon` (def. 4), `--test-weeks` (def. 52),
`--no-plots`. Iterar sobre decenas de archivos = poner los `.parquet`/`.csv` en el
directorio de entrada; el runner los descubre y procesa en paralelo.

---

## 2. Estructura del paquete

| Módulo            | Responsabilidad                                                        |
|-------------------|------------------------------------------------------------------------|
| `config.py`       | `PipelineConfig`: **todas** las decisiones como parámetros documentados |
| `data.py`         | Carga, agregación a serie semanal, índice temporal, imputación         |
| `features.py`     | Términos de Fourier (estacionalidad) + estandarización de meteo         |
| `validation.py`   | Walk-forward expansivo (folds sin data leakage)                        |
| `models.py`       | Baselines, selección de orden ARIMA, ajuste/predicción SARIMAX         |
| `metrics.py`      | MAE, RMSE, MAPE, sMAPE, WAPE, MASE                                      |
| `diagnostics.py`  | Ljung-Box, Jarque-Bera, sesgo — con interpretación                     |
| `plots.py`        | Forecast vs. real + panel de diagnóstico de residuos                   |
| `pipeline.py`     | Orquesta una serie end-to-end y persiste artefactos                    |
| `runner.py`       | CLI: itera sobre un directorio en paralelo + resumen global            |

Diseño: funciones pequeñas y tipadas, `PipelineConfig` inmutable (sin estado
compartido entre series → seguro para paralelizar), y un núcleo idéntico para los
dos niveles (solo cambian rutas de salida y etiqueta).

---

## 3. Decisiones de diseño (y su porqué)

### 3.1 Unidad de serie y variable objetivo
- **Dos pipelines, un núcleo:** `hospital-semana` y `region-semana`. Cada archivo de
  entrada trae muchas filas (`establecimiento × causa × semana`); se **colapsa a una
  serie por SUMA de `NumTotal`** (media para la meteo, que es constante dentro de una
  región-semana).
- **Objetivo (endógena):** `NumTotal`. Es el agregado más estable; las franjas etarias
  quedan disponibles para replicar el pipeline si se necesita desagregar por edad.

### 3.2 Transformación
- **`log1p`** sobre el conteo (no negativo) para estabilizar la varianza; se revierte
  con `expm1` y se recorta a ≥ 0 **antes** de calcular métricas (siempre en unidades
  reales de atenciones).

### 3.3 Variables exógenas
- **Se usan:** `temp_promedio_semanal`, `humedad_relativa_promedio_porcentaje`,
  `precipitacion_promedio_diario_mm`, `radiacion_solar_media_MJ_m2`.
- **Se descartan PM2.5 y PM10:** en los datos vienen como **texto**
  (`"Vacío: Registro histórico no disponible en modelo CAMS…"`) para todo el período
  previo a ~2020. Un hueco estructural de ~6 años no es imputable y rompería SARIMAX.
  Se pueden reactivar para un experimento acotado a 2020+ moviéndolas a `meteo_exog`.
- **Estandarización** (media/desv) de la meteo con parámetros ajustados **solo en el
  train de cada fold** → sin data leakage.

### 3.4 Estacionalidad — Fourier en vez de `s=52`
Con datos semanales, SARIMA estacional con `s=52` es **lento e inestable**. Se modela
la estacionalidad anual con **términos de Fourier** (K=2 → 4 regresores exógenos)
cuya **fase es la semana-del-año** (`SemanaEstadistica`). Ventajas:
- Absorbe los años de **52 y 53 semanas** (`SemanaEstadistica` es semana
  **epidemiológica MINSAL**, no ISO) sin acumular desfase.
- Rápido y estable; escala a decenas/cientos de series.
- Es **determinista y conocido a futuro** → el modelo sin meteo es 100% desplegable.

### 3.5 Selección de orden `(p,d,q)`
- **`d`** por diferenciación hasta estacionariedad (**KPSS**, hasta `d_max=2`).
- **`(p,q)`** por **mini-grid `p,q ∈ {0..3}` minimizando AICc** (AIC corregido para
  muestra finita).
- Componente estacional de SARIMAX en **`(0,0,0,0)`**: la estacionalidad ya la aportan
  los Fourier.
- El orden se elige **una vez** sobre el train inicial y se **re-estiman los
  coeficientes** en cada fold (equilibrio correcto/costo).

### 3.6 Modelos comparados
En cada serie se entrenan y comparan tres, y se elige el mejor por **MASE**:
1. `seasonal_naive` — baseline obligatorio (valor de hace 52 semanas).
2. `arima_fourier` — ARIMA + Fourier (**sin** meteo).
3. `sarimax` — ARIMA + Fourier + **meteo** (mide el aporte real de las exógenas).

### 3.7 Rolling Window (Walk-Forward Validation)
- **Ventana expansiva:** el train crece en cada fold (aprovecha ~12 años → ≥ 3 ciclos
  anuales). Se comparó contra la deslizante fija; la expansiva es preferible salvo que
  se sospeche un cambio de régimen fuerte (el shock COVID 2020–21 no lo justificó como
  default; puede probarse cambiando la config).
- **Backtest sobre el último año** (`test_weeks=52`); train inicial ≥ `min_train_weeks=156`.
- **Horizonte `h=4`** semanas, **paso = 4** (folds **no solapados**) → 13 folds que
  evalúan honestamente el pronóstico a 1–4 semanas.
- **Sin data leakage:** el train de cada fold son solo observaciones anteriores; el
  scaler de meteo y (la primera vez) el orden ARIMA se ajustan solo con datos pasados.

**Estrategias de Rolling Window (resumen):**
| Estrategia            | Cuándo usarla                                        |
|-----------------------|------------------------------------------------------|
| Expansiva (elegida)   | Mucho histórico, proceso estable → máximo uso de datos |
| Deslizante fija       | Sospecha de cambio de régimen; “olvidar” lo antiguo  |
| Bloqueada (gap)       | Si hubiera fuerte autocorrelación de corto plazo a aislar |

---

## 4. Métricas — cuándo usar cada una

| Métrica | Lectura                              | Cuándo conviene                                  |
|---------|--------------------------------------|--------------------------------------------------|
| MAE     | Error medio absoluto (unidades)      | Interpretable y robusto; reporte operativo       |
| RMSE    | Penaliza errores grandes             | Cuando los picos importan mucho                  |
| MAPE    | Error porcentual                     | Intuitivo, **pero se dispara/indefine con ~0**   |
| sMAPE   | MAPE simétrico y acotado             | Mejor que MAPE cuando hay ceros                  |
| WAPE    | \|error\| total / demanda total      | **Estable** en series intermitentes/con ceros    |
| MASE    | Error escalado por naive estacional  | **Comparable entre series**; <1 = mejor que baseline |

**Criterio de selección:** el mejor modelo se elige por **MASE** (comparabilidad entre
series de distinta escala); se reportan además **WAPE** y **MAE** para lectura operativa.

---

## 5. Diagnóstico de residuos
Un buen modelo deja residuos ≈ ruido blanco. Se reportan (en `metricas.json` y en
`graficos/residuos_diagnostico.png`):
- **Ljung-Box** (autocorrelación): `p<0.05` ⇒ queda estructura sin capturar.
- **Jarque-Bera** (normalidad): `p<0.05` ⇒ colas/asimetría; los **intervalos** de
  predicción pueden quedar mal calibrados (el pronóstico puntual sigue siendo útil).
- **Sesgo** (media de residuos ≈ 0): una media lejos de 0 indica sub/sobre-predicción.
- Panel gráfico: residuos en el tiempo, histograma, **ACF** y **PACF**.

---

## 6. Salidas

```
resultados/
  <nivel>/                      # region | hospital
    <id_serie>/
      predicciones.csv          # backtest: fecha, fold, y_true, pred_/lo_/hi_ por modelo
      prediccion_futura.csv     # próximo mes: fecha, semana, predicción, ic_inferior, ic_superior
      metricas.json             # métricas por modelo, campeón, modelo guardado, futuro, diagnóstico
      residuos.csv              # residuos del SARIMAX guardado
      modelo.pkl                # SARIMAXResults del mejor SARIMAX (joblib) — SIEMPRE presente
      graficos/
        forecast_vs_real.png    # historia + backtest con banda IC + pronóstico próximo mes con IC
        residuos_diagnostico.png
  _resumen/
    metricas_global_<nivel>.csv # una fila por serie, ordenada por MASE (ranking)
```

### 6.1 Intervalos de confianza, pronóstico futuro y modelo guardado
- **Campeón vs. modelo guardado.** El backtest compara 3 modelos y reporta el
  **campeón** por MASE (puede ser el `seasonal_naive`). Pero el modelo que se
  **guarda en `modelo.pkl`**, del que salen los IC y el pronóstico, es siempre el
  **mejor SARIMAX** (el entregable del proyecto es un SARIMAX). Ambos se registran en
  `metricas.json` (`best_model` y `modelo_guardado`).
- **Intervalos de confianza (95%).** Se toman de `get_forecast().conf_int()` del
  SARIMAX y se revierten con `expm1` (la transformación es monótona → el orden se
  conserva). Se muestran como banda en el backtest y en el pronóstico futuro.
- **Pronóstico del próximo mes (h=4).** El SARIMAX final (ajustado con toda la serie)
  proyecta las 4 semanas siguientes con IC. Si el modelo usa meteo, la **meteo futura
  se estima con climatología** (promedio por semana-del-año) — documentado en
  `nota_futuro`.
- **Limpieza del tramo final.** Antes de modelar se descartan semanas fuera de
  secuencia (p. ej. una **semana 53 espuria** en un año incompleto) y **semanas
  finales parciales** (total < 50% de la mediana reciente = corte de datos incompleto),
  para que una semana truncada no sesgue el pronóstico. Cada descarte queda en
  `warnings`.

---

## 7. Caveat importante (honestidad del backtest)
El modelo `sarimax` usa la **meteo del período a predecir** como exógena. En el
backtest esa meteo es conocida, por lo que sus métricas son **optimistas** respecto a
producción, donde habría que **pronosticar la meteo** primero (o usar climatología).
El modelo **`arima_fourier` no tiene esta dependencia** (Fourier es determinista y
conocido a futuro), por lo que suele ser la opción **operativamente más segura** si su
MASE es cercano al de `sarimax`. El pipeline reporta ambos para tomar esa decisión con
datos.

---

## 7bis. Sitio web de resultados
Página estática para explorar todos los resultados, con **filtro por región y por
hospital**, que muestra en cada caso las **métricas** y los **gráficos** (validación
con IC + diagnóstico de residuos) y el pronóstico del próximo mes.

```bash
# 1. Generar los datos del sitio (lee resultados/, copia gráficos, escribe web/data.js)
python -m forecasting.build_web --results resultados --out web

# 2a. Abrir directo: doble clic en web/index.html   (es autocontenido, no necesita servidor)
# 2b. O servir localmente:
python -m http.server 8123 --directory web   # y abrir http://localhost:8123
```

Se versiona `web/index.html` y `forecasting/build_web.py`; `web/data.js` y
`web/assets/` son generados (regenerar tras cada corrida del pipeline).

**Planificación de capacidad (para dirección).** En cada serie, el director ingresa
la **capacidad semanal** (regional o del hospital) y el sistema la contrasta con la
**semana pico** del próximo mes más el margen de error del modelo:
`umbral = predicción_semana_pico × (1 + MAPE/100)`.
- `capacidad ≥ umbral` → mensaje **verde**: la capacidad da abasto (muestra la holgura).
- `capacidad < umbral` → mensaje **rojo**: el próximo mes se superará la capacidad
  (muestra el déficit) y recomienda medidas: más insumos, más camillas, reforzar personal.

## 8. Cómo cambiar el comportamiento
Todo se controla desde `PipelineConfig` (ver `config.py`), sin tocar la lógica.
Ejemplos: `horizon`, `test_weeks`, `step`, `fourier_terms`, `p_range`/`q_range`,
`log_transform`, `standardize_exog`, `meteo_exog`/`drop_exog`, `n_jobs`.
