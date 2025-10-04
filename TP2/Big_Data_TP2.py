
# ╔════════════════════════════════════════════════════════════════╗
# ║                       TRABAJO PRÁCTICO N°2                     ║
# ║                             E337 - DATA BIG               ║
# ╚════════════════════════════════════════════════════════════════╝
#
#   Integrantes:
#   • Rosario Namur
#   • Marcos Ohanian
#   • Juliana Belén
#
#
#   Fecha: 18 de Abril, 2025
#
# ═══════════════════════════════════════════════════════════════════════════════════════
# Hemos hecho una modificacion respecto de la base de datos del TP anterior.
#Ahora, los ingresos han sido "unificados" con el fin de poder ser comparables entre anos. 
#Como dice el antiguo y conocido refran: "Decir que en 2024 el ingreso es mayor que en 2004 es obvio, debemos ajustar por inflacion"

# La nueva base "EPH_Limpia" incluye estas modificaciones


###### Códigos iniciales    #########################################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import os


os.getcwd()

#os.chdir("/Users/Marcos/Desktop/Facu/Cuarto/Big Data/GitHub/E337-Grupo1/TP2/Bases")
os.chdir("/Users/rosarionamur/Desktop/Big Data/TP2")
#os.chdir("\\Users\\Usuario\\Desktop\\Big Data UdeSa\\GitHub\\E337-Grupo1\\TP1\\Bases de Datos")


#########   PARTE I: LIMPIEZA DE LA BASE DE DATOS   ################################

#########   EJERCICIO 1   #########################################################

#Abro la base de excel que ya limpiamos en el TP anterior
Base_Limpia= pd.read_excel("Base_EPH_Limpia.xlsx")

cant_columnas = len(Base_Limpia.columns)
cant_filas = len(Base_Limpia)

#########   EJERCICIO 2   #########################################################
# 1. Lista de todas las columnas
todas_las_columnas = Base_Limpia.columns.tolist()

# 2. Lista de variables categóricas (filtradas previamente)
vars_categ = ['ch03', 'ch07', 'ch08', 'ch09', 'ch10', 'ch12', 'ch14', 'ch15', 'ch16', 'nivel_ed', 'cat_ocup', 'cat_inac', 'pp02e', 'pp03h', 'intensi', 'pp04a', 'pp04c', 'pp04c99', 'pp04g', 'pp05c_1', 'pp05c_2', 'pp05c_3', 'pp05h', 'pp06e', 'pp07a', 'pp07d', 'pp07e', 'pp07j', 'pp07k', 'pp08j3', 'pp09a', 'pp10a', 'pp10e', 'pp11a', 'pp11c', 'pp11c99', 'pp11l', 'pp11m', 'pp11o', 'decocur', 'rdecocur', 'gdecocur', 'adecocur', 'decindr', 'rdecindr', 'gdecindr', 'adecindr', 'decifr', 'rdecifr', 'gdecifr', 'adecifr', 'deccfr', 'rdeccfr', 'gdeccfr', 'adeccfr']

# 3. Variables no categóricas
vars_no_categ = [col for col in todas_las_columnas if col not in vars_categ]

# 4. Total de columnas dummies (una por categoría)
total_dummies = sum(Base_Limpia[var].nunique() for var in vars_categ)

# 5. Total de columnas finales
total_columnas_final = total_dummies + len(vars_no_categ)

# 6. Total de filas (no cambia)
total_filas = Base_Limpia.shape[0]

# 7. Mostrar la dimensión final
print(f"La dimensión final del DataFrame sería: ({total_filas}, {total_columnas_final})")



#########   EJERCICIO 3   #########################################################

vars_interes = ["ch06","ch07", "ch08", "ch09", "nivel_ed", "pp02e", "intensi", "pp04a", "pp04g", "pp05h", "pp07k", "pp10a", "pp10e", "pp11l"]
#Observacion: a pesar de que ch04 si es una d enuestras vars de interes, como ya es una variable dummy, no la incluimos en esta lista pues no tiene sentido generar dos dummies adicionales
#Al inspeccionar cada variable, vamos a generar dummies por cada categoria 

#Generamos a mano las categorias de cada variable
for var in vars_interes:
    for value in Base_Limpia[var].dropna().unique():
        valor_str = int(value) if isinstance(value, float) and value.is_integer() else value
        nombre_columna = f"{var}_{valor_str}"
        Base_Limpia[nombre_columna] = (Base_Limpia[var] == value).astype(int)


Base_Limpia.to_excel("Base_Dummies.xlsx", index=False)


#########   PARTE II: CLUSTERING   #########################################################
Base_Dummies= pd.read_excel("Base_Dummies.xlsx")


#### Genramos la variable continua de anos de educacion #######################################################

import pandas as pd
import numpy as np

def calcular_anos_estudio(fila):
    ch12 = fila['ch12']
    ch13 = fila['ch13']
    ch14 = fila['ch14']

    # Diccionario de años "base" ya completados según el nivel alcanzado
    base_dict = {
        1: 0,  # Jardín
        2: 0,  # Primario
        3: 0,  # EGB
        4: 6,  # Secundario
        5: 6,  # Polimodal
        6: 12, # Terciario
        7: 12, # Universitario
        8: 17  # Posgrado universitario
    }
    
    base = base_dict.get(ch12, 0)
    
    # Si ch12 indica jardín/preescolar (nivel 1), el resultado es 0 sí o sí
    if ch12 == 1:
        return 0


    # Función para determinar años extra si finalizó el nivel
    def anios_completos(ch12):
        if ch12 == 2:   # Primario
            return 6
        elif ch12 == 3:      # EGB
            return 8
        elif ch12== 4: # Secundario 
            return 6
        elif ch12 == 5:      # Polimodal
            return 3
        elif ch12 == 6:      # Terciario
            return 3
        elif ch12 == 7:      # Universitario
            return 5
        elif ch12 == 8:      # Posgrado
            return 3
        else:
            return 0

    # Determinar años adicionales
    if ch13 == 1:
        extra = anios_completos(ch12)
    elif ch13 in [2, 3]:  # No finalizó o Ns/Nr
        try:
            ch14_int = int(str(ch14).zfill(2))
            if ch14_int in [98, 99]:
                extra = 0
            else:
                extra = ch14_int
        except:
            extra = 0
    else:
        extra = 0

    return base + extra

Base_Dummies['ano_ed'] = Base_Dummies.apply(calcular_anos_estudio, axis=1)

############################################################################################


#########   EJERCICIO 1   ######################################################### 

#Creamos la base de datos con los que si respondieron 

Base_Dummies_Respondieron = Base_Dummies[Base_Dummies['estado']!=0]
Base_Dummies_Respondieron.to_excel("Base_Dummies_Respondieron.xlsx", index=False)

#Para chequear que se hayan eliminado todas las observaciones que valian 0 en estado, hacemos el siguientes print para saber que solo se han eliminados los 0s al pasar de una base a otra
respondidos = Base_Dummies['estado'].value_counts()
print(respondidos)

respondidos_prueba = Base_Dummies_Respondieron['estado'].value_counts()
print(respondidos_prueba)


#########   EJERCICIO 2A   #########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering


Ingreso_Educacion = Base_Dummies_Respondieron[["ano_ed","ipcf_2024"]]
Ingreso_Educacion.to_excel("Ingreso_Educacion.xlsx", index=False)

#Ingreso_Educacion = Ingreso_Educacion[Ingreso_Educacion["ano_ed"] != 99]

#2 clusters
kmeans2 = KMeans(n_clusters=2, random_state=10, init="random", n_init=20).fit(Ingreso_Educacion)
kmeans2.labels_

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.scatter(Ingreso_Educacion["ano_ed"], Ingreso_Educacion["ipcf_2024"], c=kmeans2.labels_)
ax.set_title("K-Means Clustering con K=2")
ax.set_xlabel("Años de educación")
ax.set_ylabel("Ingreso (a precios de 2024)")
plt.show()


#4 clusters
kmeans2 = KMeans(n_clusters=4, random_state=10, init="random", n_init=20).fit(Ingreso_Educacion)
kmeans2.labels_

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.scatter(Ingreso_Educacion["ano_ed"], Ingreso_Educacion["ipcf_2024"], c=kmeans2.labels_)
ax.set_title("K-Means Clustering con K=4")
ax.set_xlabel("Años de educación")
ax.set_ylabel("Ingreso (a precios de 2024)")
plt.show()


#10 clusters
kmeans2 = KMeans(n_clusters=10, random_state=10, init="random", n_init=20).fit(Ingreso_Educacion)
kmeans2.labels_

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.scatter(Ingreso_Educacion["ano_ed"], Ingreso_Educacion["ipcf_2024"], c=kmeans2.labels_)
ax.set_title("K-Means Clustering con K=10")
ax.set_xlabel("Años de educación")
ax.set_ylabel("Ingreso (a precios de 2024)")
plt.show()


#########   EJERCICIO 2B   #########################################################
#Necesito Base_Dummies para correr lo siguiente

#### Nos quedamos solo con los OCUPADOS
Base_Ocupados = Base_Dummies[Base_Dummies['estado']==1]

#Nos quedamos solo con las variables ipcf y la nueva variable generada sobre anos de educacion para las perosnas ocupadas

Ingreso_Educacion_Ocupados = Base_Ocupados[["ano_ed","ipcf_2024"]]


#2 clusters
kmeans2 = KMeans(n_clusters=2, random_state=10, init="random", n_init=20).fit(Ingreso_Educacion_Ocupados)
kmeans2.labels_

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.scatter(Ingreso_Educacion_Ocupados["ano_ed"], Ingreso_Educacion_Ocupados["ipcf_2024"], c=kmeans2.labels_)
ax.set_title("K-Means Clustering con K=2 para ocupados")
ax.set_xlabel("Años de educación")
ax.set_ylabel("Ingreso (a precios de 2024)")
plt.show()


#DESOCUPADOS
Base_Desocupados = Base_Dummies[Base_Dummies['estado']==2]
Ingreso_Educacion_Desocupados = Base_Desocupados[["ano_ed","ipcf_2024"]]


#2 clusters
kmeans2 = KMeans(n_clusters=2, random_state=10, init="random", n_init=20).fit(Ingreso_Educacion_Desocupados)
kmeans2.labels_

fig, ax = plt.subplots(1, 1, figsize=(7,7))
ax.scatter(Ingreso_Educacion_Desocupados["ano_ed"], Ingreso_Educacion_Desocupados["ipcf_2024"], c=kmeans2.labels_)
ax.set_title("K-Means Clustering con K=2 desocupados")
ax.set_xlabel("Años de educación")
ax.set_ylabel("Ingreso (a precios de 2024)")
plt.show()


#########   EJERCICIO 3  #########################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris

Base_Dummies= pd.read_excel("Base_Dummies_Respondieron.xlsx")

#Para hacer el dendograma no tenemos que tener missings, nos fijamos con que variables podemos graficar dentro de las seleccionadas

Base_Dummies[vars_jerarquico].isnull().mean() * 100

percentage_data = Base_Dummies.notnull().mean() * 100

complete_vars = percentage_data[percentage_data == 100].index.tolist()

vars_jerarquico = ["ch04","ch07", "nivel_ed"]


# Paso 1: Seleccionamos las variables del DataFrame
data = Base_Dummies[vars_jerarquico]  

# Paso 2: Estandarización de los datos
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# Paso 3: Generamos la matriz de linkage para el clustering jerárquico
linkage_data = linkage(data_std, method='ward')  # Ward minimiza la varianza intra-cluster

# Paso 4: Graficamos el dendrograma
plt.figure(figsize=(12, 8))
dendrogram(linkage_data, labels=data.index.tolist(), leaf_rotation=90)
plt.title('Dendrograma del Clustering Jerárquico')
plt.xlabel('Observaciones')
plt.ylabel('Distancia')
plt.tight_layout()
plt.show()


#########   PARTE III: HISTOGRAMAS Y KERNELS   #########################################################
#Utilizamos la base dummies respondieron
Base_Dummies_Respondieron= pd.read_excel("Base_Dummies_Respondieron.xlsx")

import numpy as np
import matplotlib.pyplot as plt

#Defino dos variables, una para el ipcf de 2004 
ipcf_2004_24 = Base_Dummies_Respondieron['ipcf_2024'][(Base_Dummies_Respondieron['ano4']==2004)]


#Vemos su distribucion
distribucion_ipcf_2004 = ipcf_2004_24.value_counts()


# Definimos límites percentilares para 2004
lower_percentile_2004 = ipcf_2004_24.quantile(0.05)  # 5%
upper_percentile_2004 = ipcf_2004_24.quantile(0.95)  # 95%

### Histograma 2004 ###
# Filtramos según percentiles
ipcf_filtered_2004 = Base_Dummies_Respondieron['ipcf_2024'][
    (Base_Dummies_Respondieron['ipcf_2024'] >= lower_percentile_2004) & (Base_Dummies_Respondieron['ipcf_2024'] <= upper_percentile_2004) 
    & (Base_Dummies_Respondieron['ano4']==2004)]

# Graficamos el histograma
plt.figure(figsize=(10, 6))
plt.hist(ipcf_filtered_2004, bins=20, color='steelblue', alpha=0.7)
plt.xlabel('Ingreso per cápita (2004)')
plt.ylabel('Frecuencia')
plt.title('Distribución del ingreso per cápita (5° a 95° percentil)')
plt.grid(True)
plt.tight_layout()
plt.show()
    
### Histograma 2024 ###

#Defino una variable para el ipcf de 2024 
ipcf_2024 = Base_Dummies_Respondieron['ipcf_2024'][(Base_Dummies_Respondieron['ano4']==2024)]

#Vemos su distribucion
distribucion_ipcf_2024 = ipcf_2024.value_counts()

# Definimos límites percentilares para 2004
lower_percentile_2024 = ipcf_2024.quantile(0.05)  # 5%
upper_percentile_2024 = ipcf_2024.quantile(0.95)  # 95%

# Filtramos según percentiles
ipcf_filtered_2024 = Base_Dummies_Respondieron['ipcf_2024'][
    (Base_Dummies_Respondieron['ipcf_2024'] >= lower_percentile_2024) & (Base_Dummies_Respondieron['ipcf_2024'] <= upper_percentile_2024) 
    & (Base_Dummies_Respondieron['ano4']==2024)]

# Graficamos el histograma
plt.figure(figsize=(10, 6))
plt.hist(ipcf_filtered_2024, bins=20, color='steelblue', alpha=0.7)
plt.xlabel('Ingreso per cápita (2024)')
plt.ylabel('Frecuencia')
plt.title('Distribución del ingreso per cápita (45° a 95° percentil)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Definimos límites percentilares para 2004
lower_percentile_2024 = ipcf_2024.quantile(0.428)  # 5%
upper_percentile_2024 = ipcf_2024.quantile(0.95)  # 95%

# Filtramos según percentiles
ipcf_filtered_2024 = Base_Dummies_Respondieron['ipcf_2024'][
    (Base_Dummies_Respondieron['ipcf_2024'] >= lower_percentile_2024) & (Base_Dummies_Respondieron['ipcf_2024'] <= upper_percentile_2024) 
    & (Base_Dummies_Respondieron['ano4']==2024)]

# Graficamos el histograma
plt.figure(figsize=(10, 6))
plt.hist(ipcf_filtered_2024, bins=20, color='steelblue', alpha=0.7)
plt.xlabel('Ingreso per cápita (2024)')
plt.ylabel('Frecuencia')
plt.title('Distribución del ingreso per cápita (43° a 95° percentil)')
plt.grid(True)
plt.tight_layout()
plt.show()


######## Ej 2 ##############################################################

Base_Dummies_Respondieron = pd.read_excel("Base_Dummies_Respondieron.xlsx")

##### Ej 2a #####
#Para identificar los hogares debemos utilizar 3 variables: Codusu, nro_hogar y componente. Nuestra forma de pensarlo fue la siguiente, la j-esima persona va a pertenecer al i-esimo hogar dentro de la k-esima vivienda. 

# Creamos identificador de hogar
Base_Dummies_Respondieron['id_hogar'] = Base_Dummies_Respondieron['codusu'].astype(str) + "_" + Base_Dummies_Respondieron['nro_hogar'].astype(str)

# Creamos variable binaria: 1 si la persona está desocupada, 0 en otro caso
Base_Dummies_Respondieron['desocupado'] = (Base_Dummies_Respondieron['estado'] == 2).astype(int)

# Agrupar por hogar y ver si al menos un miembro está desocupado
hogares_desocupacion = Base_Dummies_Respondieron.groupby('id_hogar')['desocupado'].max().reset_index()
hogares_desocupacion.rename(columns={'desocupado': 'hogar_con_desocupado'}, inplace=True)

hogares_desocupacion.to_excel('hogares_desocupacion.xlsx')

# Unirlo de nuevo con la base original
Base_Dummies_Respondieron = Base_Dummies_Respondieron.merge(hogares_desocupacion, on='id_hogar', how='left')

# Ahora cada persona tiene una variable 'hogar_con_desocupado' que vale 1 si alguien en su hogar está desocupado


##### Ej 2b #####

# Filtramos personas ocupadas (ESTADO == 1) y con ingreso positivo (ipcf > 0)
Base_Dummies_Respondieron['ocupado_con_ingreso'] = ((Base_Dummies_Respondieron['estado'] == 1) & (Base_Dummies_Respondieron['ipcf_2024'] > 0)).astype(int)

# Creamos una nueva columna que contenga ipcf solo si la persona está ocupada y tiene ingreso
Base_Dummies_Respondieron['ipcf_filtrado'] = Base_Dummies_Respondieron['ipcf_2024'] * Base_Dummies_Respondieron['ocupado_con_ingreso']

# Sumamos ingresos por hogar
ingreso_hogar = Base_Dummies_Respondieron.groupby('id_hogar')['ipcf_filtrado'].sum().reset_index()
ingreso_hogar.rename(columns={'ipcf_filtrado': 'ingreso_hogar_ocupados'}, inplace=True)

# Unimos con la base original
Base_Dummies_Respondieron = Base_Dummies_Respondieron.merge(ingreso_hogar, on='id_hogar', how='left')

# Ahora cada persona tiene una variable 'ingreso_hogar_ocupados' que es la suma del ingreso individual (ipcf) sólo de los miembros del hogar ocupados y con ingreso positivo.
Base_Dummies_Respondieron.to_excel("Base_Dummies_Respondieron_xHogar.xlsx", index=False)


# Suponiendo que df es tu DataFrame con las columnas 'ingreso', 'ocupacion' y 'año'

##### Ej c #####
#Somos conscientes del equilibrio entre cumplir estrictamente con una consigna mediante herramientas automatizadas y, por otro lado, profundizar en el aprendizaje de los conceptos discutidos en clase. En este trabajo, optamos por priorizar nuestro proceso de aprendizaje, aun cuando ello pueda implicar una calificación más baja. Consideramos que este enfoque es más enriquecedor y coherente con los objetivos formativos del curso.
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.neighbors import KernelDensity

# Trabajamos sobre la base
df = Base_Dummies_Respondieron.copy()

# Nos quedamos con una fila por hogar (por id_hogar)
df_hogares = df.drop_duplicates('id_hogar')[['id_hogar', 'ano4', 'ingreso_hogar_ocupados', 'hogar_con_desocupado']].dropna()

# Función para estimar densidad y devolver valores para .plot()
def estimar_densidad_plot(data, bandwidth=500):
    kde = KernelDensity(kernel='tophat', bandwidth=bandwidth)
    kde.fit(data[:, None])
    x_plot = np.linspace(data.min(), data.max(), 1000)[:, None]
    log_dens = kde.score_samples(x_plot)
    return x_plot[:, 0], np.exp(log_dens)

# Colores consistentes
colores = {1: "#e74c3c", 0: "#27ae60"}  # rojo y verde estilo kernel plot
labels = {1: "Con desocupados", 0: "Sin desocupados"}

# Graficamos uno para cada año
for año in [2004, 2024]:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Distribución del ingreso por hogar en {año} - Kernel uniforme", fontsize=16)

    for estado in [1, 0]:
        datos = df_hogares[(df_hogares['ano4'] == año) & 
                           (df_hogares['hogar_con_desocupado'] == estado)]['ingreso_hogar_ocupados']
        datos = datos[datos > 0]

        if len(datos) > 0:
            x, y = estimar_densidad_plot(datos.values, bandwidth=500)
            ax.plot(x, y, alpha=0.7, label=labels[estado], color=colores[estado])

    ax.set_xlabel("Ingreso por hogar")
    ax.set_ylabel("Densidad estimada")
    ax.legend()
    ax.set_xlim(left=0, right=2_000_000)  # Limita el eje x hasta 5 millones
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Filtrar a las personas con ingresos menores a 25,000
df_filtrado = Base_Dummies_Respondieron[Base_Dummies_Respondieron['ipcf_2024'] < 25000]

# Calcular el total de personas con ingresos menores a 25,000 por ocupación y año
total_personas = df_filtrado.groupby(['hogar_con_desocupado', 'ano4']).size().reset_index(name='count')

# Calcular el total de personas por ocupación y año sin filtro de ingreso
total_personas_all = df.groupby(['hogar_con_desocupado', 'ano4']).size().reset_index(name='total')

# Unir ambas tablas para poder calcular los porcentajes
df_merged = pd.merge(total_personas, total_personas_all, on=['hogar_con_desocupado', 'ano4'])

# Calcular el porcentaje
df_merged['porcentaje'] = (df_merged['count'] / df_merged['total']) * 100

# Mostrar el resultado
print(df_merged)

#######################################################################

Base_Dummies_Respondieron.to_excel('Base_Dummies_Respondieron.xlsx')

# Filtrar datos solo para hogares con desocupados en 2004
año = 2004
estado = 1  # 1 = Con desocupados

# Preparamos el gráfico
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(f"Distribución del ingreso por hogar en {año} - Hogares con desocupados", fontsize=16)

# Seleccionamos los datos
datos = df_hogares[(df_hogares['ano4'] == año) & 
                   (df_hogares['hogar_con_desocupado'] == estado)]['ingreso_hogar_ocupados']
datos = datos[datos > 0]

# Solo si hay datos
if len(datos) > 0:
    x, y = estimar_densidad_plot(datos.values, bandwidth=50)
    ax.plot(x, y, alpha=0.7, label=labels[estado], color=colores[estado])

ax.set_xlabel("Ingreso por hogar")
ax.set_ylabel("Densidad estimada")
ax.legend()
ax.set_xlim(left=0)
ax.grid(True)
plt.tight_layout()
plt.show()


###################################################################################################
#### Ej d ######
#Calculamos el Bandwith optimo

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Asegurate de usar solo ingresos positivos
datos = df_hogares['ingreso_hogar_ocupados']
datos = datos[datos > 0].values[:, None]

# Definimos un rango de posibles valores de ancho de banda
bandwidths = np.linspace(50, 2000, 40)  # Ajustá el rango según tu escala

# Buscamos el mejor ancho de banda con validación cruzada
grid = GridSearchCV(KernelDensity(kernel='tophat'),
                    {'bandwidth': bandwidths},
                    cv=5)  # 5 folds
grid.fit(datos)

# Imprimimos el valor óptimo
print("Ancho de banda óptimo:", grid.best_params_['bandwidth'])

### El ancho de banda optimo es 50 #####

###volevemos a correr el inciso c pero ahora con el ancho de banda optimo #######

# Función para estimar densidad y devolver valores para .plot()
def estimar_densidad_plot(data, bandwidth=50):
    kde = KernelDensity(kernel='tophat', bandwidth=bandwidth)
    kde.fit(data[:, None])
    x_plot = np.linspace(data.min(), data.max(), 1000)[:, None]
    log_dens = kde.score_samples(x_plot)
    return x_plot[:, 0], np.exp(log_dens)

# Colores consistentes
colores = {1: "#e74c3c", 0: "#27ae60"}  # rojo y verde estilo kernel plot
labels = {1: "Con desocupados", 0: "Sin desocupados"}

# Graficamos uno para cada año
for año in [2004, 2024]:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Distribución del ingreso por hogar en {año} - Kernel uniforme", fontsize=16)

    for estado in [1, 0]:
        datos = df_hogares[(df_hogares['ano4'] == año) & 
                           (df_hogares['hogar_con_desocupado'] == estado)]['ingreso_hogar_ocupados']
        datos = datos[datos > 0]

        if len(datos) > 0:
            x, y = estimar_densidad_plot(datos.values, bandwidth=500)
            ax.plot(x, y, alpha=0.7, label=labels[estado], color=colores[estado])

    ax.set_xlabel("Ingreso por hogar")
    ax.set_ylabel("Densidad estimada")
    ax.legend()
    ax.set_xlim(left=0, right=2_000_000)  # Limita el eje x hasta 5 millones
    ax.grid(True)
    plt.tight_layout()
    plt.show()
######### Ej e ###########
#Vemos la cantidad de hogares segun si alguno de sus integrantes es desocupado para ambos años

estado_por_ano = df_hogares.groupby(['hogar_con_desocupado', 'ano4']).size().unstack()



################################### Ej 3 ########################################################
#volvemos a graficar los kernels pero ahora utilizando dos dstribuciones distintas. Mantenemos el ancho de banda optimo de 50.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

# Trabajamos sobre la base
df = Base_Dummies_Respondieron.copy()

# Nos quedamos con una fila por hogar (por id_hogar)
df_hogares = df.drop_duplicates('id_hogar')[['id_hogar', 'ano4', 'ingreso_hogar_ocupados', 'hogar_con_desocupado']].dropna()

# Función para estimar densidad y devolver valores para .plot()
def estimar_densidad_plot(data, kernel='tophat', bandwidth=50):
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(data[:, None])
    x_plot = np.linspace(data.min(), data.max(), 1000)[:, None]
    log_dens = kde.score_samples(x_plot)
    return x_plot[:, 0], np.exp(log_dens)

# Colores consistentes
colores = {1: "#e74c3c", 0: "#27ae60"}  # rojo y verde
labels = {1: "Con desocupados", 0: "Sin desocupados"}

# Parámetros generales
kernels = ['tophat', 'epanechnikov', 'gaussian']
kernel_nombres = {'tophat': 'Uniforme', 'epanechnikov': 'Epanechnikov', 'gaussian': 'Gaussiano'}
limite_x = 2_000_000  # cambiar si querés otro límite
bandwidth = 500

# Graficamos uno para cada año y para cada kernel
for kernel in kernels:
    for año in [2004, 2024]:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Distribución del ingreso por hogar en {año} - Kernel {kernel_nombres[kernel]}", fontsize=16)

        for estado in [1, 0]:
            datos = df_hogares[(df_hogares['ano4'] == año) & 
                               (df_hogares['hogar_con_desocupado'] == estado)]['ingreso_hogar_ocupados']
            # Filtramos los ingresos entre 0 y el límite que pusiste
            datos = datos[(datos > 0) & (datos <= limite_x)]

            if len(datos) > 0:
                x, y = estimar_densidad_plot(datos.values, kernel=kernel, bandwidth=bandwidth)
                ax.plot(x, y, alpha=0.7, label=labels[estado], color=colores[estado])

        ax.set_xlabel("Ingreso por hogar")
        ax.set_ylabel("Densidad estimada")
        ax.legend()
        ax.set_xlim(left=0, right=limite_x)
        ax.grid(True)
        plt.tight_layout()
        plt.show()













