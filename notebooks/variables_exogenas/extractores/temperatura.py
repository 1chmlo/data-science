import pandas as pd
import requests
import time
from pathlib import Path

# 1. Diccionario con coordenadas aproximadas del centro/capital de cada región de Chile
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

# Configuración de fechas
START_DATE = "2014-01-01"
END_DATE = "2026-06-17"  # Ajustado a la fecha actual

# Lista para almacenar los DataFrames de cada región
lista_regiones = []

print("Iniciando la descarga de datos desde Open-Meteo...")

# 2. Iterar por cada región para consultar la API
for region, coords in REGIONES_CHILE.items():
    print(f"-> Descargando datos de la Región: {region}...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "temperature_2m_mean",
        "timezone": "America/Santiago"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Lanza error si la respuesta no es 200
        data = response.json()
        
        # Extraer fechas y temperaturas diarias
        fechas = data["daily"]["time"]
        temperaturas = data["daily"]["temperature_2m_mean"]
        
        # Crear un DataFrame temporal para esta región
        df_temp = pd.DataFrame({
            "fecha": pd.to_datetime(fechas),
            "temperatura_diaria": temperaturas,
            "region": region
        })
        
        lista_regiones.append(df_temp)
        
        # Respetar límites de la API gratuita (buena práctica)
        time.sleep(1)
        
    except Exception as e:
        print(f"Error al descargar los datos de {region}: {e}")

# 3. Consolidar todos los datos en un solo DataFrame masivo
df_completo = pd.concat(lista_regiones, ignore_index=True)

# 4. Agrupar por semanas de manera matemática estricta
print("\nCalculando promedios semanales...")

df_completo['anio'] = df_completo['fecha'].dt.isocalendar().year
df_completo['semana'] = df_completo['fecha'].dt.isocalendar().week

# Agrupar por Región, Año e ISO-Semana
df_semanal = df_completo.groupby(['region', 'anio', 'semana'])['temperatura_diaria'].mean().reset_index()

# Redondear el promedio a dos decimales y renombrar columna
df_semanal['temp_promedio_semanal'] = df_semanal['temperatura_diaria'].round(2)
df_semanal = df_semanal.drop(columns=['temperatura_diaria'])

# Ordenar los datos para que queden legibles (por región, luego cronológicamente)
df_semanal = df_semanal.sort_values(by=['region', 'anio', 'semana']).reset_index(drop=True)

# 5. Exportar el resultado final a CSV
salida_csv = Path(__file__).resolve().parents[1] / "csv" / "temperatura_semanal_regiones_2014_2026.csv"
salida_csv.parent.mkdir(parents=True, exist_ok=True)
df_semanal.to_csv(salida_csv, index=False, encoding='utf-8-sig')

print(f"\n¡Proceso finalizado con éxito! Archivo guardado como: '{salida_csv.name}' en la carpeta csv")
print(f"Total de registros generados: {len(df_semanal)}")