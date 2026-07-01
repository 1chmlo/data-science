import pandas as pd
import requests
import os

# 1. Definir la URL y dónde guardaremos el archivo localmente
url = "https://datos.gob.cl/dataset/606ef5bb-11d1-475b-b69f-b980da5757f4/resource/ae6c9887-106d-4e98-8875-40bf2b836041/download/at_urg_respiratorio_semanal.parquet"
archivo_local = "./dataset/at_urg_respiratorio_semanal.parquet"

# 2. Verificar si el archivo ya existe. Si no, lo descargamos de forma segura.
if not os.path.exists(archivo_local):
    print(f"Descargando datos desde el servidor... (Esto puede tardar un momento)")
    
    # Hacemos la petición con stream=True para no saturar la memoria y evitar cortes
    respuesta = requests.get(url, stream=True)
    respuesta.raise_for_status() # Verifica que la URL esté funcionando (Código 200)
    
    # Escribimos el archivo en el disco por "pedacitos" (chunks)
    with open(archivo_local, 'wb') as archivo:
        for chunk in respuesta.iter_content(chunk_size=8192):
            if chunk:
                archivo.write(chunk)
    print("¡Descarga completada con éxito!")
else:
    print(f"El archivo '{archivo_local}' ya existe localmente. Omitiendo descarga.")

# 3. Leer el archivo local con Pandas
# Nota: Si fastparquet te da problemas, también puedes intentar engine='pyarrow'
df_urgencias = pd.read_parquet(archivo_local, engine='fastparquet')

# 4. Ver los resultados
print(df_urgencias.head())