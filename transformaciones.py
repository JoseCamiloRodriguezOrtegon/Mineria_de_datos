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

# Carga monolítica del dataset
RUTA = Path.cwd() / "Sensores_limpio.csv"
if not RUTA.exists():
    RUTA = Path.cwd().parent / "Sensores_limpio.csv"

NUMERICAS = [
    "TotalSteps",
    "VeryActiveMinutes",
    "FairlyActiveMinutes",
    "LightlyActiveMinutes",
    "SedentaryMinutes",
    "Calories",
]

CATEGORICAS = ["Nivel_Pasos", "Nivel_Sedentarismo"]

df = pd.read_csv(RUTA)
X = df[NUMERICAS]
print("Dataset limpio:", df.shape, "| Matriz numérica:", X.shape)


# 1 · POR QUÉ ESCALAR: la distancia la domina la variable con magnitudes grandes
def distancia(a, b):
    return ((a - b) ** 2).sum() ** 0.5


c1, c2 = X.iloc[0], X.iloc[1]
print("\n--- Distancia entre dos registros SIN escalar ---")
for col in NUMERICAS:
    print(f"  {col:<22} aporta {abs(c1[col] - c2[col]):>10.1f}")
print("Distancia total sin escalar :", round(distancia(c1, c2), 1))

X_minmax = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=NUMERICAS)
c1e, c2e = X_minmax.iloc[0], X_minmax.iloc[1]
print("Distancia total con MinMax  :", round(distancia(c1e, c2e), 3))

# 2 · ESCALADORES NUMÉRICOS
print("\n--- MinMaxScaler (Rango [0, 1]) ---")
print("Antes : min", X.min().round(1).tolist())
print("Después: min", X_minmax.min().round(3).tolist(), "max", X_minmax.max().round(3).tolist())

print("\n--- StandardScaler (media=0, desv=1) ---")
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=NUMERICAS)
print("Medias tras escalar:", X_std.mean().round(3).tolist())
print("Desv. estándar     :", X_std.std().round(3).tolist())

print("\n--- RobustScaler (resistente a outliers mediante mediana e IQR) ---")
X_rob = pd.DataFrame(RobustScaler().fit_transform(X), columns=NUMERICAS)
print("Mediana tras escalar:", X_rob.median().round(3).tolist())

# 3 · CATEGORIZACIÓN / CODIFICACIÓN (One-Hot Encoding)
print("\n--- Codificación de Categóricas con get_dummies ---")
X_cat = pd.get_dummies(df[CATEGORICAS], prefix=CATEGORICAS, dtype=int)
print("Columnas binarias generadas:", X_cat.columns.tolist())
print(X_cat.head(3).to_string())

# 4 · DISCRETIZACIÓN por cuantiles
print("\n--- KBinsDiscretizer: TotalSteps -> 4 rangos ordinales ---")
discretizador = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
pasos_bins = discretizador.fit_transform(df[["TotalSteps"]]).ravel()
bordes = discretizador.bin_edges_[0].round(0)
print("Bordes de los rangos:", bordes)
print(pd.Series(pasos_bins).value_counts().sort_index().to_string())

# 5 · REDUCCIÓN DE DIMENSIONALIDAD CON PCA
print("\n--- PCA sobre variables estandarizadas ---")
pca = PCA()
componentes = pca.fit_transform(X_std)
razones = pca.explained_variance_ratio_
acumulada = razones.cumsum().round(3)
print("Varianza explicada por componente :", razones.round(3))
print("Varianza acumulada                :", acumulada)
para_80 = int((acumulada < 0.80).sum()) + 1
para_90 = int((acumulada < 0.90).sum()) + 1
print(f"Componentes para explicar >=80%: {para_80} | para >=90%: {para_90} (de {len(NUMERICAS)})")

# 6 · MATRIZ MONOLÍTICA FINAL PARA MODELADO
X_final = pd.concat([X_std, X_cat], axis=1)
print("\n--- Matriz final consolidada ---")
print("Dimensiones finales:", X_final.shape)
print("Lista de columnas:", X_final.columns.tolist())