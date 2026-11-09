import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Carga de datos
df = pd.read_csv('logistica_envios.csv')

# Variables predictoras (las seis cuantitativas de cantidad) y variable objetivo
cols = [
    'Cantidad_envios_mes',
    'Distancia_promedio_km',
    'Peso_promedio_kg',
    'Cantidad_vehiculos',
    'Dias_entrega_promedio',
    'Antiguedad_empresa_anios'
]

X = df[cols]
y = df['Ingresos_mensuales_usd']

# Factor de inflacion de la varianza (VIF)
datos = add_constant(X)
print(pd.DataFrame({
    'Variable': cols,
    'VIF': [round(variance_inflation_factor(datos.values, i + 1), 2)
            for i in range(len(cols))]
}).to_string(index=False))
