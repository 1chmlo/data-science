import pandas as pd
import requests
import time
from pathlib import Path

REGIONES_CHILE = {
    "Arica y Parinacota": {"lat": -18.4746, "lon": -70.2979}, "Tarapacá": {"lat": -20.2133, "lon": -70.1457},
    "Antofagasta": {"lat": -23.6509, "lon": -70.3975}, "Atacama": {"lat": -27.3668, "lon": -70.3323},
    "Coquimbo": {"lat": -29.9533, "lon": -71.2436}, "Valparaíso": {"lat": -33.0472, "lon": -71.6127},
    "Metropolitana": {"lat": -33.4489, "lon": -70.6693}, "O'Higgins": {"lat": -34.1708, "lon": -70.7444},
    "Maule": {"lat": -35.4264, "lon": -71.6554}, "Ñuble": {"lat": -36.6066, "lon": -72.1034},
    "Bío Bío": {"lat": -36.8270, "lon": -73.0503}, "La Araucanía": {"lat": -38.7359, "lon": -72.5904},
    "Los Ríos": {"lat": -39.8142, "lon": -73.2459}, "Los Lagos": {"lat": -41.4693, "lon": -72.9424},
    "Aysén": {"lat": -45.5712, "lon": -72.0685}, "Magallanes": {"lat": -53.1638, "lon": -70.9171}
}

START_DATE, END_DATE = "2014-01-01", "2026-06-17"
lista_df = []

print("=== EXTRACCIÓN DE HUMEDAD RELATIVA ===")
for region, coords in REGIONES_CHILE.items():
    print(f"Descargando {region}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"], "longitude": coords["lon"],
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": "relative_humidity_2m_mean", "timezone": "America/Santiago"
    }
    try:
        res = requests.get(url, params=params).json()
        df_t = pd.DataFrame({
            "fecha": pd.to_datetime(res["daily"]["time"]),
            "humedad": res["daily"]["relative_humidity_2m_mean"],
            "region": region
        })
        lista_df.append(df_t)
        time.sleep(0.5)
    except Exception as e: print(f"Error en {region}: {e}")

df_completo = pd.concat(lista_df, ignore_index=True)
df_completo['anio'] = df_completo['fecha'].dt.isocalendar().year
df_completo['semana'] = df_completo['fecha'].dt.isocalendar().week

df_semanal = df_completo.groupby(['region', 'anio', 'semana'])['humedad'].mean().reset_index()

def detectar_nulo(row):
    return str(round(row['humedad'], 2)) if not pd.isna(row['humedad']) else "Vacío: Registro meteorológico ausente en estación o modelo"

df_semanal['humedad_relativa_promedio_porcentaje'] = df_semanal.apply(detectar_nulo, axis=1)
salida_csv = Path(__file__).resolve().parents[1] / "csv" / "humedad_semanal_regiones_2014_2026.csv"
salida_csv.parent.mkdir(parents=True, exist_ok=True)
df_semanal.drop(columns=['humedad']).to_csv(salida_csv, index=False, encoding='utf-8-sig')
print(f"¡Archivo '{salida_csv.name}' guardado en la carpeta csv!")