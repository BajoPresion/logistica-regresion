import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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

# Particion en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones sobre ambos conjuntos
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# Coeficientes y metricas

print()
print(pd.DataFrame({
    'Variable': cols,
    'Coeficiente': model.coef_.round(3)
}).to_string(index=False))

print('Intercepto:', round(model.intercept_, 2))

# Metricas

print()
print(f'RMSE test: {root_mean_squared_error(y_test, y_pred_test):.2f} USD')
print(f'MAE test:  {mean_absolute_error(y_test, y_pred_test):.2f} USD')
print(f'R2 train:  {r2_score(y_train, y_pred_train):.3f}')
print(f'R2 test:   {r2_score(y_test, y_pred_test):.3f}')

# Grafico de valores reales contra predichos

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_test, alpha=0.8, label='Predicciones')
plt.plot([y.min(), y.max()], [y.min(), y.max()],
         color='red', linewidth=2, label='Acierto perfecto')

plt.xlabel('Ingresos reales (USD)')
plt.ylabel('Ingresos predichos (USD)')
plt.title('Ingresos reales vs predichos - conjunto de prueba')

plt.legend()
plt.tight_layout()
plt.show()