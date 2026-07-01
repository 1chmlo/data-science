import pandas as pd

# 1. Cargar el original
file_path = 'dataset/at_urg_respiratorio_2026-05-10.parquet'
cols_interes = ['Anio', 'SemanaEstadistica', 'OrdenCausa', 'Causa', 'NumTotal']
df = pd.read_parquet(file_path, columns=cols_interes)

# 2. Aplicar la estandarización (Lo que ya validamos)
df['Causa_Busqueda'] = df['Causa'].str.lower().str.replace(r'^- ', '', regex=True).str.strip()

mapeo_codigos = {
    'j00-j06': 'IRA alta (J00-J06)',
    'j09-j11': 'Influenza (J09-J11)',
    'j12-j18': 'Neumonía (J12-J18)',
    'j20-j21': 'Bronquitis/Bronquiolitis aguda (J20-J21)',
    'j40-j46': 'Crisis obstructiva bronquial (J40-J46)',
    'j22': 'Otra causa respiratoria (J22, J30-J39, J47, J60-J98)',
    'u07.1': 'COVID-19, virus identificado (U07.1)',
    'u07.2': 'COVID-19, virus no identificado (U07.2)'
}

def clasificar(texto):
    if 'total' in texto:
        return 'DESCARTAR'
    for codigo, nombre_oficial in mapeo_codigos.items():
        if codigo in texto:
            return nombre_oficial
    return 'DESCARTAR'

df['Causa_Final'] = df['Causa_Busqueda'].apply(clasificar)

# 3. Quedarnos solo con los datos limpios
df_final = df[df['Causa_Final'] != 'DESCARTAR'].copy()

# 4. Limpieza final de columnas (quitamos la auxiliar de búsqueda y renombramos)
df_final = df_final.drop(columns=['Causa', 'Causa_Busqueda'])
df_final = df_final.rename(columns={'Causa_Final': 'Causa'})

# =========================================================
# 5. GUARDAR COMO UN ARCHIVO NUEVO (ESTANDARIZADO)
# =========================================================
nuevo_nombre = 'dataset/at_urg_respiratorio_LIMPIO.parquet'
df_final.to_parquet(nuevo_nombre)

print(f"¡Listo! Se ha creado el archivo: {nuevo_nombre}")
print(f"Registros guardados: {len(df_final)}")