import pandas as pd

# 1. Cargar el dataset
ruta_dataset = 'dataset/dataset.parquet'
df = pd.read_parquet(ruta_dataset)

print(df.head())

# Exportar las primeras 5 filas (head) a un archivo CSV
df.head().to_csv('head_dataset.csv', index=False)
print("Las primeras 5 filas se han exportado a 'head_dataset.csv'")

'''


# 2. Explorar las columnas para saber cómo se llama tu columna de hospitales
print("Columnas en el dataset:", df.columns.tolist())

# 3. Contar las observaciones por hospital
# Reemplaza 'NOMBRE_DE_TU_COLUMNA' con el nombre real de la columna de hospital
# Ej: 'hospital', 'id_hospital', 'nombre_hospital', etc.
columna_hospital = 'EstablecimientoGlosa' 

if columna_hospital in df.columns:
    conteo_hospitales = df[columna_hospital].value_counts()
    print("\nCantidad de observaciones por hospital:")
    print(conteo_hospitales)
else:
    print(f"\nNo se encontró la columna '{columna_hospital}'. Por favor, actualiza la variable con un nombre de las columnas listadas arriba.")
    
    
'''