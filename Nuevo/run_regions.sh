#!/usr/bin/env bash
# Ejecuta el pipeline sobre las 16 regiones, de forma controlada (una sola corrida).
set -e
cd /home/prettyperlix/dev/data-science/Nuevo

# Seguridad: no dejar corridas previas compitiendo por memoria.
pkill -9 -f 'forecasting.runner' 2>/dev/null || true
sleep 1
rm -rf resultados forecasting/__pycache__

echo "Procesos runner activos (debe ser 0):"
pgrep -af 'forecasting.runner' | grep -v pgrep || echo "  ninguno"
echo "Memoria disponible:"
free -h | sed -n '2p'
echo "=== inicio $(date +%H:%M:%S) ==="

.venv/bin/python -m forecasting.runner \
    --input-dir datasets_regionales --level region \
    --out-dir resultados --jobs 4

echo "=== fin $(date +%H:%M:%S) ==="
