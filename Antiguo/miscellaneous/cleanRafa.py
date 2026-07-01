import pandas as pd

def limpiar_datos_urgencia(df):
    print("Limpieza de NivelAtencion y TipoUrgencia...")
    
    df_limpio = df.copy()
    
    columnas_objetivo = ['NivelAtencion', 'TipoUrgencia']
    
    print("Nulos antes de limpiar:")
    print(df_limpio[columnas_objetivo].isnull().sum())
    
    # Limpieza
    for col in columnas_objetivo:
        # Rellenar nulos
        df_limpio[col] = df_limpio[col].fillna('DESCONOCIDO')
        
        # Pasar a mayúsculas y quitar espacios extra
        df_limpio[col] = df_limpio[col].astype(str).str.strip().str.upper()

    # Nulos finales
    print("\nNulos después de limpiar:")
    print(df_limpio[columnas_objetivo].isnull().sum())
    print("-" * 50)
    
    return df_limpio