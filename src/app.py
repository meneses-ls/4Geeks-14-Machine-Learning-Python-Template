# - Importación de librerías:

import pandas as pd
import utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest

# - Importación del set de datos:

url = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'

# - Guardado:

rutg = '../data/raw/dataset.csv'

# - Lectura:

df = pd.read_csv(url)
df.to_csv(rutg, index=False)
print("El archivo fue guardado en la ruta:", rutg)

# - Búsqueda ignorando el índice:

duplicados = df.drop("host_id", axis=1).duplicated().sum()
print(f"Número de duplicados (ignorando 'host_id'): {duplicados}")

# - Eliminación ignorando el índice:

df = df.drop_duplicates(subset=df.columns.difference(['host_id']))
print("Forma del DataFrame después de eliminar duplicados (sin 'host_id'):", df.shape)

# - Búsqueda ignorando el índice:

duplicados = df.drop("host_id", axis=1).duplicated().sum()
print(f"Número de duplicados (ignorando 'host_id'): {duplicados}")

# - Eliminación ignorando el índice:

df = df.drop_duplicates(subset=df.columns.difference(['host_id']))
print("Forma del DataFrame después de eliminar duplicados (sin 'host_id'):", df.shape)

df.drop(["host_id", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)

utils.columnas_categoricas(df, lista_num_cat=[])

utils.analisis_categorico_categorico(df, lista_num_cat=[])

utils.columnas_numericas(df, columnas_excluidas=[])

utils.analisis_numerico_numerico(df, 'price', columnas_excluidas=['latitude', 'longitude', 'id'], limites=[])

utils.analisis_categorico_categorico_multivariante(df, 'neighbourhood_group', 'room_type')

# - Factorización de variables categóricas:

df['neighbourhood_group'] = pd.factorize(df['neighbourhood_group'])[0]
df['room_type'] = pd.factorize(df['room_type'])[0]
df['neighbourhood'] = pd.factorize(df['neighbourhood'])[0]  # ← Esta línea faltaba

# - Gráfico de correlación:

fig, axis = plt.subplots(figsize=(10, 6))
sns.heatmap(df[['neighbourhood_group', 'neighbourhood', 'room_type']].corr(), annot=True, fmt=".2f")

plt.tight_layout()
plt.show()

utils.analisis_numerico_numerico_multivariante(df, 'price', columnas_excluidas=['id'])

utils.analisis_numerico_categorico(df, columnas_excluidas=['id', 'neighbourhood', 'name'])

sns.pairplot(data = df)

# - Parámetros del gráfico:

fig, axis = plt.subplots(3, 3, figsize = (15, 10))

sns.boxplot(ax = axis[0, 0], data = df, y = "price")
sns.boxplot(ax = axis[0, 1], data = df, y = "minimum_nights")
sns.boxplot(ax = axis[0, 2], data = df, y = "number_of_reviews")
sns.boxplot(ax = axis[1, 0], data = df, y = "calculated_host_listings_count")
sns.boxplot(ax = axis[1, 1], data = df, y = "availability_365")

# - Visualización del gráfico:

plt.tight_layout()
plt.show()

# - Selección de variables numéricas:

num_variables = ['availability_365','calculated_host_listings_count','number_of_reviews','minimum_nights', 'latitude', 'longitude', 'neighbourhood_group', 'room_type', 'neighbourhood']

# - División:

X = df[num_variables]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# - Reconstrucción de DataFrames con la variable objetivo incluida:

df_train = X_train.copy()
df_train['price'] = y_train

df_test = X_test.copy()
df_test['price'] = y_test

# - Escalado:

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scal = scaler.transform(X_train)
X_train_scal = pd.DataFrame(X_train_scal, index = X_train.index, columns = num_variables)

X_test_scal = scaler.transform(X_test)
X_test_scal = pd.DataFrame(X_test_scal, index = X_test.index, columns = num_variables)

selection_model = SelectKBest(f_classif, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel['Price'] = list(y_train)
X_test_sel['Price'] = list(y_test)

X_train_sel.to_csv("../data/processed/train_clean.csv", index=False)
X_test_sel.to_csv("../data/processed/test_clean.csv", index=False)