import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CARGA DIRECTA DEL ARCHIVO LIMPIO
df = pd.read_parquet('dataset/at_urg_respiratorio_LIMPIO.parquet')

print(f"--- ANÁLISIS DE INTEGRIDAD Y OUTLIERS ---\n")

# ==================================================
# 1. VALIDACIÓN DE RANGOS LÓGICOS (NUEVO)
# ==================================================
# Chequeo de Semanas (1-53)
semanas_invalidas = df[(df['SemanaEstadistica'] < 1) | (df['SemanaEstadistica'] > 53)]
min_sem = df['SemanaEstadistica'].min()
max_sem = df['SemanaEstadistica'].max()

print(f"1. Rango de Semanas: {min_sem} a {max_sem}")
print(f"   -> Registros fuera de rango (1-53): {len(semanas_invalidas)}")

# Chequeo de OrdenCausa (3-35 según tus datos)
min_ord = df['OrdenCausa'].min()
max_ord = df['OrdenCausa'].max()
print(f"2. Rango de OrdenCausa: {min_ord} a {max_ord}")
print(f"   -> Verificación: {'OK' if min_ord >= 3 and max_ord <= 35 else 'REVISAR RANGOS'}")

print("\n" + "-"*40 + "\n")

# ==================================================
# 2. ANÁLISIS DE OUTLIERS (IQR)
# ==================================================
def detect_outliers_iqr(group):
    Q1 = group['NumTotal'].quantile(0.25)
    Q3 = group['NumTotal'].quantile(0.75)
    IQR = Q3 - Q1
    limite_superior = Q3 + 1.5 * IQR
    outliers = group[group['NumTotal'] > limite_superior]
    return pd.Series([limite_superior, len(outliers)], index=['Limite_Sup', 'Cant_Outliers'])

outliers_por_causa = df.groupby('Causa').apply(detect_outliers_iqr)

print("3. Resumen de Límites y Cantidad de Outliers por Causa:")
print(outliers_por_causa)

# ==================================================
# 3. VISUALIZACIÓN Y REGISTROS EXTREMOS
# ==================================================
# Gráfico Boxplot
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='NumTotal', y='Causa', palette='viridis')
plt.title('Distribución de NumTotal y Detección de Outliers por Causa')
plt.xlabel('Número Total de Atenciones')
plt.ylabel('Causa Respiratoria')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n4. TOP 5 REGISTROS MÁS EXTREMOS (VALORES MÁS ALTOS):")
print(df.sort_values(by='NumTotal', ascending=False)[['Anio', 'SemanaEstadistica', 'Causa', 'NumTotal']].head())