
**Uso de los extractores (módulo)**

Este directorio contiene un conjunto de scripts en `extractores/` que, cuando se ejecutan, descargan datos por semana y región (Chile) en el periodo 2014–2026 y generan archivos CSV consolidados dentro de la carpeta `csv/`.

Patrón de nombre de salida

- Cada extractor genera un archivo con el siguiente patrón:

  TEMA_semanal_regiones_2014_2026.csv

  Donde `TEMA` corresponde al tema extraído por el script (por ejemplo `temperatura`, `humedad`, `precipitacion`, `radiacion`, `calidad_aire`).

Mapeo actual de extractores → nombres de archivo

- `extractores/temperatura.py` → `csv/temperatura_semanal_regiones_2014_2026.csv`
- `extractores/humedad.py` → `csv/humedad_semanal_regiones_2014_2026.csv`
- `extractores/precipitacion.py` → `csv/precipitacion_semanal_regiones_2014_2026.csv`
- `extractores/radiacionsolar.py` → `csv/radiacion_semanal_regiones_2014_2026.csv`
- `extractores/calidadaire.py` → `csv/calidad_aire_semanal_regiones_2014_2026.csv`

Cómo usar

1. Sitúate en la raíz del módulo (es decir, en la carpeta que contiene `extractores/` y `csv/`).
2. Ejecuta el extractor que desees. Ejemplos (Windows PowerShell):

```powershell
python .\extractores\temperatura.py
python .\extractores\humedad.py
python .\extractores\precipitacion.py
python .\extractores\radiacionsolar.py
python .\extractores\calidadaire.py
```

3. El CSV resultante se guardará en la carpeta `csv/` con el nombre que sigue el patrón indicado.

Advertencia importante: sobrescritura

- Los extractores producen archivos con nombres fijos usando el patrón `TEMA_semanal_regiones_2014_2026.csv`.
- Al ejecutar un extractor varias veces, el archivo existente en `csv/` será sobrescrito sin confirmación.

Recomendaciones

- Si necesitas conservar versiones anteriores, mueve o renombra el CSV existente antes de ejecutar de nuevo el extractor.
- Para integración automatizada, añade una copia de seguridad automática (por ejemplo, renombrar con timestamp) antes de ejecutar.

Notas sobre integridad de datos

- Los scripts están configurados para extraer datos desde 2014 hasta 2026-06-17 y agrupar por semana ISO para las 16 regiones de Chile.
- La presencia o ausencia de datos depende de las APIs externas (Open-Meteo, Air Quality API). Algunos valores pueden aparecer como texto descriptivo indicando vacíos cuando no hay datos disponibles.
- El extractor de calidad de aire incluye diagnósticos específicos para huecos; los demás extractores devuelven valores o cadenas de vacío según la respuesta de la API.

Fin.
