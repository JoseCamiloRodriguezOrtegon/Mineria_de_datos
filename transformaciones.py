# Sesión 4 · Minería de Datos · UNIMINUTO Ibagué
# Transformación y reducción: escalado, codificación, discretización y PCA
# Ejecutar desde esta carpeta:  python3 transformaciones.py

from pathlib import Path
import warnings

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RUTA = Path(__file__).resolve().parents[2] / "datasets_base" / "telecom_clientes_limpio.csv"
NUMERICAS = ["edad", "facturacion_mensual", "antiguedad_meses",
             "minutos_llamada", "gb_datos", "num_quejas"]

df = pd.read_csv(RUTA)
X = df[NUMERICAS]
print("Dataset limpio:", df.shape, "| matriz numérica:", X.shape)


# 1 · POR QUÉ ESCALAR: la distancia la domina la variable con unidades grandes
def distancia(a, b):
    return ((a - b) ** 2).sum() ** 0.5


c1, c2 = X.iloc[0], X.iloc[1]
print("\n--- Distancia entre dos clientes SIN escalar ---")
for col in NUMERICAS:
    print(f"  {col:<20} aporta {abs(c1[col] - c2[col]):>10.1f}")
print("Distancia total sin escalar :", round(distancia(c1, c2), 1))

X_minmax = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=NUMERICAS)
c1e, c2e = X_minmax.iloc[0], X_minmax.iloc[1]
print("Distancia total con MinMax  :", round(distancia(c1e, c2e), 3))

# 2 · LOS TRES ESCALADORES
print("\n--- MinMaxScaler (todo a [0, 1]) ---")
print("Antes : min", X.min().round(1).tolist())
print("Después: min", X_minmax.min().round(3).tolist(), "max", X_minmax.max().round(3).tolist())

print("\n--- StandardScaler (media=0, desviación=1) ---")
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=NUMERICAS)
print("Medias tras escalar:", X_std.mean().round(3).tolist())
print("Desv. estándar     :", X_std.std().round(3).tolist())

print("\n--- RobustScaler (usa mediana e IQR, resiste outliers) ---")
X_rob = pd.DataFrame(RobustScaler().fit_transform(X), columns=NUMERICAS)
print("Mediana tras escalar:", X_rob.median().round(3).tolist())

# 3 · CODIFICACIÓN DE CATEGÓRICAS: one-hot
print("\n--- One-hot con get_dummies ---")
categoricas = ["tipo_plan", "pago_automatico"]
X_cat = pd.get_dummies(df[categoricas], prefix=categoricas, dtype=int)
print("Columnas nuevas:", X_cat.columns.tolist())
print(X_cat.head(3).to_string())

# 4 · DISCRETIZACIÓN de edad en 4 rangos por cuantiles
print("\n--- KBinsDiscretizer: edad -> 4 categorías ---")
discretizador = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
edad_bins = discretizador.fit_transform(df[["edad"]]).ravel()
bordes = discretizador.bin_edges_[0].round(0)
print("Bordes de los rangos:", bordes)
print(pd.Series(edad_bins).value_counts().sort_index().to_string())

# 5 · PCA sobre las numéricas estandarizadas
print("\n--- PCA: ¿cuántas dimensiones resumen los datos? ---")
pca = PCA()
componentes = pca.fit_transform(X_std)
razones = pca.explained_variance_ratio_
acumulada = razones.cumsum().round(3)
print("Varianza explicada por componente :", razones.round(3)[:5])
print("Varianza acumulada                :", acumulada[:5])
para_80 = int((acumulada < 0.80).sum()) + 1
para_90 = int((acumulada < 0.90).sum()) + 1
print(f"Componentes para explicar >=80%: {para_80} | para >=90%: {para_90} (de {len(NUMERICAS)})")

# 6 · MATRIZ DE FEATURES LISTA PARA MODELAR
X_final = pd.concat([X_std, X_cat], axis=1)
print("\n--- Matriz final para modelado ---")
print("Forma:", X_final.shape)
print("Columnas:", X_final.columns.tolist())
