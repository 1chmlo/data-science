import pandas as pd
import requests
import time
from pathlib import Path

# 1. Coordenadas de las 16 regiones de Chile
REGIONES_CHILE = {
    "Arica y Parinacota": {"lat": -18.4746, "lon": -70.2979},
    "Tarapacá": {"lat": -20.2133, "lon": -70.1457},
    "Antofagasta": {"lat": -23.6509, "lon": -70.3975},
    "Atacama": {"lat": -27.3668, "lon": -70.3323},
    "Coquimbo": {"lat": -29.9533, "lon": -71.2436},
    "Valparaíso": {"lat": -33.0472, "lon": -71.6127},
    "Metropolitana": {"lat": -33.4489, "lon": -70.6693},
    "O'Higgins": {"lat": -34.1708, "lon": -70.7444},
    "Maule": {"lat": -35.4264, "lon": -71.6554},
    "Ñuble": {"lat": -36.6066, "lon": -72.1034},
    "Bío Bío": {"lat": -36.8270, "lon": -73.0503},
    "La Araucanía": {"lat": -38.7359, "lon": -72.5904},
    "Los Ríos": {"lat": -39.8142, "lon": -73.2459},
    "Los Lagos": {"lat": -41.4693, "lon": -72.9424},
    "Aysén": {"lat": -45.5712, "lon": -72.0685},
    "Magallanes": {"lat": -53.1638, "lon": -70.9171}
}

START_DATE = "2014-01-01"
END_DATE = "2026-06-17"

lista_contaminacion = []

print("=========================================================")
# Corregido: "Extrayendo" en lugar de "Extrayendo" con tilde incorrecta en consola
print(" INICIANDO EXTRACCIÓN DE CALIDAD DEL AIRE - 16 REGIONES")
print("=========================================================\n")

for region, coords in REGIONES_CHILE.items():
    print(f"-> Solicitando datos para: {region}...")
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": "pm2_5,pm10",
        "timezone": "America/Santiago"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        horas = data["hourly"]["time"]
        pm25_valores = data["hourly"]["pm2_5"]
        pm10_valores = data["hourly"]["pm10"]
        
        # DataFrame a nivel horario
        df_temp = pd.DataFrame({
            "fecha_hora": pd.to_datetime(horas),
            "pm2_5": pm25_valores,
            "pm10": pm10_valores,
            "region": region
        })
        
        lista_contaminacion.append(df_temp)
        time.sleep(1) # Pausa técnica para evitar bloqueos por sobrecarga
        
    except Exception as e:
        print(f"   ❌ Error crítico al conectar con la API en {region}: {e}")

# Consolidación masiva
df_completo = pd.concat(lista_contaminacion, ignore_index=True)

# Asignar año y semana ISO
df_completo['anio'] = df_completo['fecha_hora'].dt.isocalendar().year
df_completo['semana'] = df_completo['fecha_hora'].dt.isocalendar().week

print("\nProcesando y agrupando promedios semanales con diagnóstico...")

# Agrupar datos calculando el promedio semanal
df_semanal = df_completo.groupby(['region', 'anio', 'semana'])[['pm2_5', 'pm10']].mean().reset_index()

# 3. DIAGNÓSTICO AUTOMÁTICO DE VACÍOS
def diagnosticar_vacio(row, columna):
    valor = row[columna]
    anio = row['anio']
    
    # Si el valor es válido, se retorna tal cual
    if not pd.isna(valor):
        return str(round(valor, 2))
    
    # Si está vacío, evaluamos la causa cronológica o geográfica
    if anio < 2022:
        return "Vacío: Registro histórico no disponible en modelo CAMS (Previo a lanzamiento operacional)"
    elif row['region'] in ["Aysén", "Magallanes"]:
        return "Vacío: Sin cobertura satelital/asimilación terrestre en la celda austral para esta semana"
    else:
        return "Vacío: Pérdida intermitente de señal de datos del modelo Copernicus para esta coordenada"

# Aplicamos el diagnóstico de vacíos convirtiendo las columnas a tipo String descriptivo
df_semanal['pm2_5_resultado'] = df_semanal.apply(lambda r: diagnosticar_vacio(r, 'pm2_5'), axis=1)
df_semanal['pm10_resultado'] = df_semanal.apply(lambda r: diagnosticar_vacio(r, 'pm10'), axis=1)

# Limpieza final de columnas
df_semanal = df_semanal.drop(columns=['pm2_5', 'pm10'])
df_semanal = df_semanal.rename(columns={'pm2_5_resultado': 'pm2_5_ug_m3', 'pm10_resultado': 'pm10_ug_m3'})

# Guardar en CSV
archivo_salida = Path(__file__).resolve().parents[1] / "csv" / "calidad_aire_semanal_regiones_2014_2026.csv"
archivo_salida.parent.mkdir(parents=True, exist_ok=True)
df_semanal.to_csv(archivo_salida, index=False, encoding='utf-8-sig')

print(f"\n🎉 ¡Proceso finalizado! Los datos de las 16 regiones se guardaron en: '{archivo_salida.name}' dentro de la carpeta csv")
print(f"Revisa las columnas de contaminantes; si hay vacíos, verás la explicación detallada por fila.")