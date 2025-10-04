# ╔════════════════════════════════════════════════════════════════╗
# ║                       TRABAJO PRÁCTICO N°3                     ║
# ║                          E337 - DATA BIG                       ║
# ╚════════════════════════════════════════════════════════════════╝
#
#   Integrantes:
#   • Rosario Namur
#   • Marcos Ohanian
#   • Juliana Belén
#
#
#   Fecha: 19 de mayo, 2025
#
# ══════════════════════════════════════════════════════════════════


###     Codigos iniciales    #########################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import sys
from scipy.stats import norm
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score, RocCurveDisplay, mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB
from scipy import stats
from ISLP import load_data
from sklearn.preprocessing import PolynomialFeatures 


import os
os.getcwd()

#os.chdir("/Users/Marcos/Desktop/Facu/Cuarto/Big Data/GitHub/E337-Grupo1/TP2/Bases")
#os.chdir("/Users/rosarionamur/Desktop/Big Data/TP3")
os.chdir("\\Users\\Usuario\\Desktop\\Big Data UdeSa\\GitHub\\E337-Grupo1\\TP3")

EPH = pd.read_excel("EPH.xlsx")




#########   PARTE A: Un poco más de estadística descriptiva   #########################################################

#########   EJERCICIO 1   #########################################################

#########   Creación de variables propias para predecir el desempleo######


#1. Porcentaje de algunos ingresos no laborables (subsidios, limosna y trabajo de menores)
#Porcentaje del ingreso que corresponden con 'v5_m', 'v18_m', 'v19_am'
# Agruparmos por hogar y sumar ingresos
hogares = EPH.groupby(['codusu', 'nro_hogar'], as_index=False).agg({
    'v5_m': 'sum',
    'v18_m': 'sum',
    'v19_am': 'sum',
    'itf': 'sum'})
# Creamos la variable de ingresos subsidiados x hogar
#Obs: no todos los ingresos son subsidios a pesar de que la varible se llama subsidios 
hogares['subsidios'] = hogares['v5_m'].fillna(0) + hogares['v18_m'].fillna(0) + hogares['v19_am'].fillna(0)

# Calcular el porcentaje de subsidiados sobre el total del hogar
hogares['porc_subsidios'] = hogares.apply(
    lambda row: (row['subsidios'] / row['itf'] * 100) if row['itf'] > 0 else None,
    axis=1)

# Hacemos el merge para incorporar la variable a la base individual
EPH = EPH.merge(
    hogares[['codusu', 'nro_hogar', 'porc_subsidios']],
    on=['codusu', 'nro_hogar'],
    how='left' )




#2. Edad^2 (edad_2)
#Tiene sentido que la edad se relacione de manera cuadrática con el trabajo porque: los niños no trabajan, luego adolescentes y adultos si y finalmente los adultos mayores no.
EPH['edad_2'] = EPH['ch06'] ** 2




#3. Años de educación
#Mientras más años de educación tenga, menos probabilidad de ser desocupado. 
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

EPH['ano_ed'] = EPH.apply(calcular_anos_estudio, axis=1)



#4. Años de educación/edad: es una proporción de cuánto de tu vida te educaste
# Creamos la variable solo para personas que tengan 35 años o menos como un proxy de educación "a tiempo"
EPH['ratio_educ_edad'] = np.where(
    (EPH['ch06'] > 0) & (EPH['ch06'] <= 35) & (EPH['ano_ed'] >= 0),
    EPH['ano_ed'] / EPH['ch06'],
    np.nan
)



#5. Movilidad geofráfica
#Considermaos que, por lo general, las perosnas que no reciden en su lugar de nacimiento se mudaorn en 
#busca de mejores oportuidades laborales. Por lo tanto, el hecho de no haber nacido en la localidad de 
#residencia actual puede ayudarnos a predecir la ocupacion de las perosnas. Como consecuencia observacional, 
#consideramos que las perosnas que no residen en su lugar de nacimiento tienen mas chances de NO ser 
#desocupadas pues emigraron en busca de mejores oportunidades laborales.

EPH['movilidad_geo'] = EPH['ch15'].apply(lambda x: 1 if x in [2, 3, 4, 5] else 0)



#6 Responsable único
#Si una persona es jefe de hogar y se encuentra "sola" en ese hogar, es probable que  
#deba trabajar para poder mantenerse a si misma. Por lo tanto, la correlacion con la ocupacion es muy alta.
EPH['responsable_único'] = ((EPH['ch03'] == 1) & (EPH['ch07'].isin([3, 4, 5]))).astype(int)



#7 Responsabilidad materna:  cuenta la cantidad de hijos de cada madre. 
#Muchas veces, el impacto de tener hijos en la empleabilidad es diferente según el género. 
#Queremos poder ssber cuantos hijos tiene esta madre en el hogar pues esto puede afectar su participacion en el mercado laboral. 
#Muchas madres puedne haber optado por dejar de trabjar para cuidar de sus hijos en caso de ser muchos o 
#puede haberse visto desplazada del mercado laboral (imposibilidad de reinsertarse en la vida laboral siendo una mujer que tuvo varios hijos).

#Primero, reconocemos la cantidad de hijos en el hogar
# Paso 1: marcar quiénes son hijos en la base
EPH['es_hijo'] = (EPH['ch03'] == 3).astype(int)

# Paso 2: contar cuántos hijos hay por hogar
hijos_por_hogar = EPH.groupby(['codusu', 'nro_hogar'])['es_hijo'].sum().reset_index()
hijos_por_hogar.rename(columns={'es_hijo': 'hijosXhogar'}, inplace=True)

# Paso 3: mergear esta información a cada individuo
EPH = EPH.merge(hijos_por_hogar, on=['codusu', 'nro_hogar'], how='left')

#Segundo, reconocemos las "madres" en el hogar
EPH['es_madre'] = (
    (EPH['ch04'] == 2) &                         # mujer
    (EPH['ch03'].isin([1, 2])) &                 # jefa o cónyuge
    (EPH['hijosXhogar'] >= 1)                    # al menos un hijo en el hogar
).astype(int)

#Tercero, creamos la variable que nos va a decir la cantidad de hijos que tiene la madre del hogar
EPH['responsabilidad_materna'] = EPH['es_madre'] * EPH['hijosXhogar']


# 8. Hijos_x_edad
#Esta es una interacción entre la cantidad de hijos en el hogar y la edad del progenitor.
#Se construye con el objetivo de capturar el hecho de que el efecto de tener hijos sobre la ocupación no es homogéneo a lo largo del ciclo de vida.
#Debemos tener en cuenta que esta variable se debe interpretar "ceteris-paribus", esto es, se deben comparar entre las personas de la misma edad. 


EPH['hijosXedad'] = np.where(EPH['ch03'].isin([1, 2]), EPH['ch06'] * EPH['hijosXhogar'], np.nan)
#Tiene sentido que ciertos valores sean 0 en vez de missing. Estas perosnas con jefes de hogar o conyuges que no hay tenido hijos. 




########################################################################
#Graficamos las variables creadas
########################################################################

#PORCENTAJE DE SUBSIDIOS POR HOGAR (Grafico kernel)
# Definimos función para recortar outliers usando percentiles
def recortar_outliers(serie, p_inf=1, p_sup=99):
    q_inf = np.percentile(serie.dropna(), p_inf)
    q_sup = np.percentile(serie.dropna(), p_sup)
    return serie[(serie >= q_inf) & (serie <= q_sup)]


# Filtrar datos de porc_subsidios para 2004 y 2024, eliminando negativos (si hubiese)
subsidios_2004 = EPH[(EPH['ano4'] == 2004) & (EPH['porc_subsidios'] >= 0)]['porc_subsidios']
subsidios_2024 = EPH[(EPH['ano4'] == 2024) & (EPH['porc_subsidios'] >= 0)]['porc_subsidios']

# Recortar outliers igual que antes
subsidios_2004_rec = recortar_outliers(subsidios_2004)
subsidios_2024_rec = recortar_outliers(subsidios_2024)

# Graficar densidad con clip para evitar negativos y con los mismos colores
plt.figure(figsize=(10, 6))
sns.kdeplot(subsidios_2004_rec, color='#A0E7E5', linewidth=2, label='2004', fill=True, alpha=0.4, clip=(0, None))
sns.kdeplot(subsidios_2024_rec, color='#CBAACB', linewidth=2, label='2024', fill=True, alpha=0.4, clip=(0, None))

plt.title('Distribución kernel del porcentaje de subsidios en el ingreso por hogar', fontsize=14)
plt.xlabel('Porcentaje de subsidios en el ingreso por hogar', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




#EDAD AL CUADRADO (histograma)
# Definimos la función para recortar outliers
def recortar_outliers(serie, p_inf=5, p_sup=95):
    q_inf = np.percentile(serie.dropna(), p_inf)
    q_sup = np.percentile(serie.dropna(), p_sup)
    return serie[(serie >= q_inf) & (serie <= q_sup)]

# Filtramos y recortamos los datos por año
edad_2004 = EPH[EPH['ano4'] == 2004]['edad_2']
edad_2024 = EPH[EPH['ano4'] == 2024]['edad_2']

edad_2004_recortada = recortar_outliers(edad_2004)
edad_2024_recortada = recortar_outliers(edad_2024)

# Cantidad de bins
bins = 30

# Creamos los histogramas
plt.figure(figsize=(12, 5))

# Histograma para 2004
plt.subplot(1, 2, 1)
plt.hist(edad_2004_recortada, bins=bins, color='#AEC6CF', edgecolor='black')
plt.title('Distribución de edad al cuadrado (2004)', fontsize=12)
plt.xlabel('Edad al cuadrado', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)

# Histograma para 2024
plt.subplot(1, 2, 2)
plt.hist(edad_2024_recortada, bins=bins, color='#FBB1BD', edgecolor='black')
plt.title('Distribución de edad al cuadrado (2024)', fontsize=12)
plt.xlabel('Edad al cuadrado', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)

plt.tight_layout()
plt.show()




#AÑOS DE EDUCACIÓN (función kernel)
# Función para recortar outliers entre percentiles p_inf y p_sup,
# asegurando que no haya valores negativos
def recortar_outliers(serie, p_inf=1, p_sup=99):
    # Calcular percentiles
    q_inf = np.percentile(serie.dropna(), p_inf)
    q_sup = np.percentile(serie.dropna(), p_sup)
    # Aseguramos que q_inf no sea negativo
    q_inf = max(q_inf, 0)
    # Filtrar la serie entre q_inf y q_sup
    return serie[(serie >= q_inf) & (serie <= q_sup)]

# Filtrar datos para años 2004 y 2024 y con años de educación >= 0
anoed_2004 = EPH[(EPH['ano4'] == 2004) & (EPH['ano_ed'] >= 0)]['ano_ed']
anoed_2024 = EPH[(EPH['ano4'] == 2024) & (EPH['ano_ed'] >= 0)]['ano_ed']

# Recortar outliers sin permitir valores negativos
anoed_2004_rec = recortar_outliers(anoed_2004)
anoed_2024_rec = recortar_outliers(anoed_2024)

# Graficar densidad con clip para evitar extrapolación negativa
plt.figure(figsize=(10, 6))
sns.kdeplot(anoed_2004_rec, color='#A0E7E5', linewidth=2, label='2004', fill=True, alpha=0.4, clip=(0, None))
sns.kdeplot(anoed_2024_rec, color='#CBAACB', linewidth=2, label='2024', fill=True, alpha=0.4, clip=(0, None))

plt.title('Distribución kernel de años de educación', fontsize=14)
plt.xlabel('Años de educación', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




#RATIO EDUCACIÓN/EDAD (Kernel)
# Filtrar y eliminar valores negativos
ratio_2004 = EPH[(EPH['ano4'] == 2004) & (EPH['ratio_educ_edad'] >= 0)]['ratio_educ_edad']
ratio_2024 = EPH[(EPH['ano4'] == 2024) & (EPH['ratio_educ_edad'] >= 0)]['ratio_educ_edad']

#Definimos la función para recortar outliers
def recortar_outliers(serie, p_inf=5, p_sup=95):
    q_inf = np.percentile(serie.dropna(), p_inf)
    q_sup = np.percentile(serie.dropna(), p_sup)
    return serie[(serie >= q_inf) & (serie <= q_sup)]

# Recortar outliers entre percentiles 1 y 95
ratio_2004_rec = recortar_outliers(ratio_2004, p_inf=1, p_sup=95)
ratio_2024_rec = recortar_outliers(ratio_2024, p_inf=1, p_sup=95)

# Graficar distribución
plt.figure(figsize=(10, 6))
sns.kdeplot(ratio_2004_rec, color='#A0E7E5', linewidth=2, label='2004', fill=True, alpha=0.4, clip=(0, None), bw_adjust=0.8)
sns.kdeplot(ratio_2024_rec, color='#CBAACB', linewidth=2, label='2024', fill=True, alpha=0.4, clip=(0, None), bw_adjust=0.8)

plt.title('Distribución kernel de la proporción de la vida dedicada al estudio', fontsize=14)
plt.xlabel('Ratio años de educación/ edad', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.legend()
plt.grid(True)
plt.xlim(0, 1) 
plt.tight_layout()
plt.show()


#MOVILIDAD_GEO (gráfico de barras)
# Agrupamos por año y la variable de movilidad
movilidad = EPH.groupby(['ano4', 'movilidad_geo']).size().unstack()

# Graficamos barras apiladas con colores pastel
movilidad.plot(kind='bar', stacked=True, figsize=(8, 7), 
               color=['#AEC6CF', '#FFB6C1'], width=0.4)

# Etiquetas y título del gráfico
plt.xlabel('Año', fontsize=10)
plt.ylabel('Cantidad', fontsize=10)
plt.title("Personas que no viven en el lugar donde nacieron (por año)", fontsize=12)
plt.legend(title='¿Vive en otro lugar?', labels=['No', 'Sí'], fontsize=10)
plt.xticks(rotation=0)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()



#RESPONSABLE ÚNICO
# Agrupamos por año y condición de responsable único
responsable = EPH.groupby(['ano4', 'responsable_único']).size().unstack()

# Graficamos barras apiladas con colores pastel
responsable.plot(kind='bar', stacked=True, figsize=(8, 7), 
                 color=['#AEC6CF', '#FFB6C1'], width=0.4)

# Etiquetas y título del gráfico
plt.xlabel('Año', fontsize=10)
plt.ylabel('Cantidad', fontsize=10)
plt.title("Jefes/as de hogar sin pareja (por año)", fontsize=12)
plt.legend(title='¿Responsable único del hogar?', labels=['No', 'Sí'], fontsize=10)
plt.xticks(rotation=0)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


#RESPONSABILIDAD MATERNA (histograma)
# Filtrar los datos por año y limitar a 8 hijos o menos
madre_2004 = EPH[(EPH['ano4'] == 2004) & (EPH['responsabilidad_materna'] <= 8)]
madre_2024 = EPH[(EPH['ano4'] == 2024) & (EPH['responsabilidad_materna'] <= 8)]

# Contar la frecuencia de cada cantidad de hijos
conteo_2004 = madre_2004['responsabilidad_materna'].value_counts().sort_index()
conteo_2024 = madre_2024['responsabilidad_materna'].value_counts().sort_index()

# Asegurar que ambos vectores tengan los mismos índices (0 a 8)
index = list(range(0, 9))
conteo_2004 = conteo_2004.reindex(index, fill_value=0)
conteo_2024 = conteo_2024.reindex(index, fill_value=0)

# Crear DataFrame con ambas series
df = pd.DataFrame({'2004': conteo_2004, '2024': conteo_2024})

# Crear gráfico de barras agrupadas
bar_width = 0.4
x = range(len(df))

plt.figure(figsize=(10, 6))
plt.bar([i - bar_width/2 for i in x], df['2004'], width=bar_width, color='#CE93D8', label='2004')  # lila pastel
plt.bar([i + bar_width/2 for i in x], df['2024'], width=bar_width, color='#A5D6A7', label='2024')  # verde agua pastel

plt.xlabel('Cantidad de hijos por madre')
plt.ylabel('Frecuencia')
plt.title('Distribución de madres con hijos a cargo (responsabilidad materna)')
plt.xticks(ticks=x, labels=index)
plt.legend()
plt.tight_layout()
plt.show()



#HIJOS X EDAD (histograma)
# Filtrar los datos por año
datos_2004 = EPH[(EPH['ano4'] == 2004)]
datos_2024 = EPH[(EPH['ano4'] == 2024)]

# Crear gráfico KDE
plt.figure(figsize=(10, 6))
sns.kdeplot(data=datos_2004, x='hijosXedad', fill=True, label='2004', color='cyan', alpha=0.3)
sns.kdeplot(data=datos_2024, x='hijosXedad', fill=True, label='2024', color='orchid', alpha=0.3)

plt.title('Distribución kernel de hijos por edad')
plt.xlabel('Hijos por edad')
plt.ylabel('Densidad')
plt.legend()
plt.tight_layout()
plt.show()



#########   PARTE B: Enfoque de validación   #########################################################

#########   EJERCICIO 2   #########################################################
#Creamos nuevamente las bases respondieron y no respondieron pues debemos incluir las nuevas variables generadas por nosotrso en EPH
#Creamos la base de datos con los que si respondieron 
respondieron = EPH[EPH['estado']!=0]

#Creamos la base de datos con los que no respondieron 
norespondieron = EPH[EPH['estado']==0]

#########  Inspección previa   ################################################
#### Inspeccion de duplicados ####

# Inspeccion de variables duplicadas
print("Duplicados:", respondieron.duplicated().sum())

#### Inspeccion de missing values ####
missings = respondieron.isnull().sum()
missings = missings[missings > 0].sort_values(ascending=False)
#112 variables deberian ser dropeadas si eliminamos aquellas con mas del 50% de missings

#Hemos utilizado el paper de Lin y Tsai (ver el archivo de respuestas), hemoas tomado las siguientes decisiones respecto a los missing values.
#Estas deciisones se han tomado para poder realizar la tabla de diferencia de medias entre las muestras testeo y entrenamiento.

# Paso 1: Dropear columnas con más del 50% de missing
threshold = 0.5
cols_to_drop = respondieron.columns[respondieron.isnull().mean() > threshold]
respondieron = respondieron.drop(columns=cols_to_drop)

#Paso 2: Reemplazamos los missing values de las variables.
#Para las var numericas imputamos la mediana mientras que para las categoricas la moda. 

numericas = ["ipcf_2024", "ch06", "edad_2", "porc_subsidios", "ano_ed", "ratio_educ_edad", "responsabilidad_materna", "hijosXedad"]

categoricas = ["ch03", "ch04", "ch07", "ch09", "ch10", "movilidad_geo", "responsable_único", "estado"]
#Al momento de incluir variables categoricas en nuestras muestras de entrenamiento y testeo, para evitar problemas de multicolinealidad perfecta,  incluiremos todas las dummies por catageoria -1.  
# Solo no vamos a incluir la primera dummy en cada categoria. por ejemplo, en ch09 solo incluiremos ch09_2 y ch09_3 en nuestras estimaciones (ej: nuestro modelo Logit).
# Imputar mediana para las variables numéricas continuas
for col in numericas:
    if col in respondieron.columns and respondieron[col].isnull().any():
        mediana = respondieron[col].median()
        respondieron[col].fillna(mediana, inplace=True)

# Imputar moda para las variables categóricas
for col in categoricas:
    if col in respondieron.columns and respondieron[col].isnull().any():
        moda = respondieron[col].mode().dropna()
        if not moda.empty:
            respondieron[col].fillna(moda.iloc[0], inplace=True)
                

# Le cambiamos el formato a la salida de la estadistica descriptiva 
pd.set_option('display.float_format', lambda x: '%.2f' % x) 

#### Creamos dummies (variables categóricas binarias) ####

for var in categoricas:
    for value in respondieron[var].dropna().unique():
        valor_str = int(value) if isinstance(value, float) and value.is_integer() else value
        nombre_columna = f"{var}_{valor_str}"
        respondieron[nombre_columna] = (respondieron[var] == value).astype(int)



#########  Entrenamiento y testeo para 2004   #################################
# Variable dependiente

# Subset para 2004
respondieron_2004 = respondieron[respondieron['ano4'] == 2004].copy()

# Variable dependiente
y_2004 = respondieron_2004['estado_2']

# Variables independientes
X_2004 = respondieron_2004[["ipcf_2024", "ch06", "edad_2", "porc_subsidios", "ano_ed", "ratio_educ_edad", 
                            "responsabilidad_materna", "hijosXedad", "ch03_2", 
                            "ch03_3", "ch03_4", "ch03_5", "ch03_6", "ch03_7", "ch03_8", 
                            "ch03_9", "ch03_10", "ch04_1", "ch07_2", "ch07_3", "ch07_4", "ch07_5",
                            "ch09_2", "ch09_3", "ch10_2", "ch10_3", "movilidad_geo_1", "responsable_único_1"]].copy()


# Agregamos columna de unos (intercepto)
X_2004.insert(0, 'constante', 1)

# División en train y test
x_train_2004, x_test_2004, y_train_2004, y_test_2004 = train_test_split(
    X_2004, y_2004, test_size=0.3, random_state=444
)

# Tabla de diferencias de medias
filas_2004 = []

for col in X_2004.columns:
    mean_train = x_train_2004[col].mean()
    mean_test = x_test_2004[col].mean()
    std_train = x_train_2004[col].std()
    std_test = x_test_2004[col].std()
    t_stat, p_val = stats.ttest_ind(x_train_2004[col], x_test_2004[col], nan_policy='omit')
    
    filas_2004.append({
        "Mean train": mean_train,
        "Mean test": mean_test,
        "SD train": std_train,
        "SD test": std_test,
        "t-stat": t_stat,
        "p-value": p_val
    })

tabla_dif_medias_2004 = pd.DataFrame(filas_2004, index=X_2004.columns)

# Exportar
tabla_dif_medias_2004.to_excel("tabla_diferencia_medias_2004.xlsx")


#########  Entrenamiento y testeo para 2024   #################################
# Subset para 2024
respondieron_2024 = respondieron[respondieron['ano4'] == 2024].copy()

# Variable dependiente
y_2024 = respondieron_2024['estado_2']

# Variables independientes
X_2024 = respondieron_2024[["ipcf_2024", "ch06", "edad_2", "porc_subsidios", "ano_ed", "ratio_educ_edad", 
                            "responsabilidad_materna", "hijosXedad", "ch03_2", 
                            "ch03_3", "ch03_4", "ch03_5", "ch03_6", "ch03_7", "ch03_8", 
                            "ch03_9", "ch03_10", "ch04_1", "ch07_2", "ch07_3", "ch07_4", "ch07_5",
                            "ch09_2", "ch09_3", "ch10_2", "ch10_3", "movilidad_geo_1", "responsable_único_1"]].copy()

# Agregamos columna de unos (intercepto)
X_2024.insert(0, 'constante', 1)


# División en train y test
x_train_2024, x_test_2024, y_train_2024, y_test_2024 = train_test_split(
    X_2024, y_2024, test_size=0.3, random_state=444
)

# Tabla de diferencias de medias
filas_2024 = []

for col in X_2024.columns:
    mean_train = x_train_2024[col].mean()
    mean_test = x_test_2024[col].mean()
    std_train = x_train_2024[col].std()
    std_test = x_test_2024[col].std()
    t_stat, p_val = stats.ttest_ind(x_train_2024[col], x_test_2024[col], nan_policy='omit')
    
    filas_2024.append({
        "Mean train": mean_train,
        "Mean test": mean_test,
        "SD train": std_train,
        "SD test": std_test,
        "t-stat": t_stat,
        "p-value": p_val
    })

tabla_dif_medias_2024 = pd.DataFrame(filas_2024, index=X_2024.columns)

# Exportar
tabla_dif_medias_2024.to_excel("tabla_diferencia_medias_2024.xlsx")



#########   PARTE C: Métodos de Clasificación y Performance   #################################################
#########   EJERCICIO 3   #########################################################
def run_logistic_regression(X, y, x_test, y_test, year_label):
    print(f"\n--- Resultados para {year_label} ---")

    # Regresión logística sin penalización
    log_reg = LogisticRegression(penalty=None).fit(X, y)

    # Probabilidades de predicción para la clase 1 ("desocupado")
    y_pred_proba = log_reg.predict_proba(x_test)[:, 1]

    # Clasificación binaria basada en un umbral de 0.5
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)

    # Matriz de confusión
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(conf_mat)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {auc:.4f}")

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Regresión Logística {year_label} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - Regresión Logística {year_label}')
    plt.legend()
    plt.grid()
    plt.show()

# Ejecutar para 2004
run_logistic_regression(x_train_2004, y_train_2004, x_test_2004, y_test_2004, "2004")

# Ejecutar para 2024
run_logistic_regression(x_train_2024, y_train_2024, x_test_2024, y_test_2024, "2024")

#2.LDA #######################################

def run_lda(X_train, y_train, X_test, y_test, year_label):
    print(f"\n--- Resultados LDA para {year_label} ---")

    # Entrenar modelo LDA
    lda = LDA()
    lda = lda.fit(X_train, y_train)

    # Predicciones
    y_test_pred = lda.predict(X_test)

    # Matriz de confusión y Accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)

    print("Matriz de confusión:\n", cm)
    print("Accuracy: %.4f" % accuracy)

    # Curva ROC y AUC
    y_pred_score = lda.predict_proba(X_test)[:, 1]  # Score para clase positiva

    auc = roc_auc_score(y_test, y_pred_score)
    print("AUC: %.4f" % auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_score, drop_intermediate=False)
    print("Thresholds:", thresholds)
    print("FPR:", fpr)
    print("TPR:", tpr)

    # Graficar curva ROC
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=f'LDA {year_label}')
    display.plot()
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=0.8)
    plt.title(f"Curva ROC - LDA {year_label}")
    plt.show()

# Ejecutar para 2004
run_lda(x_train_2004, y_train_2004, x_test_2004, y_test_2004, "2004")

# Ejecutar para 2024
run_lda(x_train_2024, y_train_2024, x_test_2024, y_test_2024, "2024")


#3.QDA ########################################

def run_qda(x_train, y_train, x_test, y_test, year_label):
    print(f"\n--- Resultados QDA para {year_label} ---")

    # Entrenar modelo QDA
    qda = QDA() 
    qda.fit(x_train, y_train)
    results_qda = qda.predict(x_test)

    # Predicciones
    y_pred_qda = pd.Series(results_qda.tolist())

    # Matriz de confusión
    conf_mat = confusion_matrix(y_test, y_pred_qda)
    print("Matriz de confusión:")
    print(conf_mat)   

    # Accuracy
    acc = accuracy_score(y_test, y_pred_qda)
    print(f"Accuracy: {acc:.4f}")

    # AUC y Curva ROC
    auc = roc_auc_score(y_test, y_pred_qda)
    print(f"AUC QDA: {auc:.2f}")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_qda)

    # Graficar curva ROC
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=f'QDA {year_label}')
    display.plot()  
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f"Curva ROC - QDA {year_label}")
    plt.show()

# Ejecutar para 2004
run_qda(x_train_2004, y_train_2004, x_test_2004, y_test_2004, "2004")

# Ejecutar para 2024
run_qda(x_train_2024, y_train_2024, x_test_2024, y_test_2024, "2024")


#4.KNN con K=5 ###############################

def run_knn(X_train, y_train, X_test, y_test, year_label, k=5):
    print(f"\n--- Resultados KNN (k={k}) para {year_label} ---")

    # Entrenar modelo KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predicciones
    y_test_pred_knn = knn.predict(X_test)
    y_test_prob_knn = knn.predict_proba(X_test)[:, 1]  # Probabilidades para ROC y AUC

    # Métricas
    accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
    cm_knn = confusion_matrix(y_test, y_test_pred_knn)
    auc_knn = roc_auc_score(y_test, y_test_prob_knn)
    fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_test_prob_knn)

    # Mostrar resultados
    print("Matriz de confusión (KNN):\n", cm_knn)
    print(f"Accuracy (KNN): {accuracy_knn:.4f}")
    print(f"AUC (KNN): {auc_knn:.4f}")

    # Graficar curva ROC
    display = RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn, roc_auc=auc_knn, estimator_name=f'KNN (k={k}) - {year_label}')
    display.plot()
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=0.8)
    plt.title(f"Curva ROC - KNN (k={k}) - {year_label}")
    plt.show()

# Ejecutar para 2004
run_knn(x_train_2004, y_train_2004, x_test_2004, y_test_2004, "2004")

# Ejecutar para 2024
run_knn(x_train_2024, y_train_2024, x_test_2024, y_test_2024, "2024")

#5.NAIVE BAYES ######################################

def run_naive_bayes(x_train, y_train, x_test, y_test, year_label):
    print(f"\n--- Resultados Naive Bayes para {year_label} ---")

    # Entrenar el modelo
    nb = GaussianNB()
    nb.fit(x_train, y_train)

    # Predicciones
    y_pred = nb.predict(x_test)
    y_pred_proba = nb.predict_proba(x_test)[:, 1]  # Probabilidad de ser desocupado

    # Matriz de confusión
    conf_mat = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(conf_mat)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {auc:.4f}")

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'Naive Bayes (AUC = {auc:.2f}) - {year_label}', color='purple')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - Naive Bayes {year_label}')
    plt.legend()
    plt.grid()
    plt.show()

# Ejecutar para 2004
run_naive_bayes(x_train_2004, y_train_2004, x_test_2004, y_test_2004, "2004")

# Ejecutar para 2024
run_naive_bayes(x_train_2024, y_train_2024, x_test_2024, y_test_2024, "2024")


#########   EJERCICIO 5   #########################################################
def procesar_base_norespondieron(df, numericas, categoricas, variables_modelo):
    df = df.copy()

    # Imputamos valores faltantes
    for col in numericas:
        if col in df.columns and df[col].isnull().any():
            mediana = df[col].median()
            df[col].fillna(mediana, inplace=True)

    for col in categoricas:
        if col in df.columns and df[col].isnull().any():
            moda = df[col].mode().dropna()
            if not moda.empty:
                df[col].fillna(moda.iloc[0], inplace=True)

    # Creamos las dummies
    for var in categoricas:
        for value in df[var].dropna().unique():
            valor_str = int(value) if isinstance(value, float) and value.is_integer() else value
            nombre_columna = f"{var}_{valor_str}"
            df[nombre_columna] = (df[var] == value).astype(int)
            
    # Creamos la matriz X con columnas en el mismo orden que las del modelo
    X = pd.DataFrame()
    for col in variables_modelo:
        if col == 'constante':
            X[col] = 1
        elif col in df.columns:
            X[col] = df[col]
        else:
            # Si la variable no está (por ejemplo, una dummy que no se generó), la agregamos en 0
            X[col] = 0

    # Luego, para más seguridad, reemplazamos NaNs remanentes por 0
    X = X.fillna(0)

    return X


# Reutilizamos nuestras listas anteriores
numericas = ["ipcf_2024", "ch06", "edad_2", "porc_subsidios", "ano_ed", "ratio_educ_edad", 
             "responsabilidad_materna", "hijosXedad"]

categoricas = ["ch03", "ch04", "ch07", "ch09", "ch10", "movilidad_geo", "responsable_único", "estado"]

variables_modelo = ["constante", "ipcf_2024", "ch06", "edad_2", "porc_subsidios", "ano_ed", "ratio_educ_edad", 
                    "responsabilidad_materna", "hijosXedad", "ch03_2", 
                    "ch03_3", "ch03_4", "ch03_5", "ch03_6", "ch03_7", "ch03_8", 
                    "ch03_9", "ch03_10", "ch04_1", "ch07_2", "ch07_3", "ch07_4", "ch07_5",
                    "ch09_2", "ch09_3", "ch10_2", "ch10_3", "movilidad_geo_1", "responsable_único_1"]

# Filtramos la base de norespondieron por año
norespondieron_2004 = norespondieron[norespondieron["ano4"] == 2004].copy()
norespondieron_2024 = norespondieron[norespondieron["ano4"] == 2024].copy()

# Preprocesamos
X_nopred_2004 = procesar_base_norespondieron(norespondieron_2004, numericas, categoricas, variables_modelo)
X_nopred_2024 = procesar_base_norespondieron(norespondieron_2024, numericas, categoricas, variables_modelo)

# Ahora, entrenamos nuestro modelo para cada año
lda_2004 = LDA().fit(x_train_2004, y_train_2004)
lda_2024 = LDA().fit(x_train_2024, y_train_2024)

# Predecimos
y_pred_nrespond_2004 = lda_2004.predict(X_nopred_2004)
y_pred_nrespond_2024 = lda_2024.predict(X_nopred_2024)

# Proporción de desocupados predichos
prop_desocup_2004 = (y_pred_nrespond_2004 == 1).mean()
prop_desocup_2024 = (y_pred_nrespond_2024 == 1).mean()


# Agregamos las predicciones al dataframe original
norespondieron_2004["estado_predicho"] = y_pred_nrespond_2004
norespondieron_2024["estado_predicho"] = y_pred_nrespond_2024





























