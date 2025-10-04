# ╔════════════════════════════════════════════════════════════════╗
# ║                       TRABAJO PRÁCTICO N°4                     ║
# ║                          E337 - DATA BIG                       ║
# ╚════════════════════════════════════════════════════════════════╝
#
#   Integrantes:
#   • Rosario Namur
#   • Marcos Ohanian
#   • Juliana Belén
#
#
#   Fecha: 5 de junio, 2025
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
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import rcParams


import os
os.getcwd()

#os.chdir("/Users/Marcos/Desktop/Facu/Cuarto/Big Data/GitHub/E337-Grupo1/TP2/Bases")
os.chdir("/Users/rosarionamur/Desktop/Big Data/TP4")
#os.chdir("\\Users\\Usuario\\Desktop\\Big Data UdeSa\\GitHub\\E337-Grupo1\\TP4")

#La base EPH a continuacion no tiene las variables creadas en el TP3.
#Por esto, incorporamos el codigo para hacer esas variables del trabajo anterior
#Para no tener que correrlo todo de vuelta cada vez, hicimos una nueva base EPH_variables (cargada en el Github)
#Si abris EPH.xlsx tenes que correr el codigo de variables
#Si no, esta el atajo de EPH_variables e ir directamente a linea 183

EPH = pd.read_excel("EPH.xlsx")
EPH = pd.read_excel("EPH_variables.xlsx")

###################Codigo para la creacion de variables (Si se usa EPH) ###############################################
#si usas EPH = pd.read_excel("EPH_variables.xlsx") podes ir directo a linea 183

#1. Porcentaje de algunos ingresos no laborables (subsidios, limosna y trabajo de menores)
hogares = EPH.groupby(['codusu', 'nro_hogar'], as_index=False).agg({
    'v5_m': 'sum',
    'v18_m': 'sum',
    'v19_am': 'sum',
    'itf': 'sum'})
hogares['subsidios'] = hogares['v5_m'].fillna(0) + hogares['v18_m'].fillna(0) + hogares['v19_am'].fillna(0)

hogares['porc_subsidios'] = hogares.apply(
    lambda row: (row['subsidios'] / row['itf'] * 100) if row['itf'] > 0 else None,
    axis=1)

EPH = EPH.merge(
    hogares[['codusu', 'nro_hogar', 'porc_subsidios']],
    on=['codusu', 'nro_hogar'],
    how='left' )

#2. Edad^2 (edad_2)
EPH['edad_2'] = EPH['ch06'] ** 2

#3. Años de educación
def calcular_anos_estudio(fila):
    ch12 = fila['ch12']
    ch13 = fila['ch13']
    ch14 = fila['ch14']

    base_dict = {
        1: 0,  
        2: 0,  
        3: 0,  
        4: 6,  
        5: 6,  
        6: 12, 
        7: 12, 
        8: 17  
    }
    
    base = base_dict.get(ch12, 0)
    
    if ch12 == 1:
        return 0

    def anios_completos(ch12):
        if ch12 == 2:   
            return 6
        elif ch12 == 3:      
            return 8
        elif ch12== 4: 
            return 6
        elif ch12 == 5:      
            return 3
        elif ch12 == 6:     
            return 3
        elif ch12 == 7:      
            return 5
        elif ch12 == 8:      
            return 3
        else:
            return 0

    if ch13 == 1:
        extra = anios_completos(ch12)
    elif ch13 in [2, 3]:  
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
EPH['ratio_educ_edad'] = np.where(
    (EPH['ch06'] > 0) & (EPH['ch06'] <= 35) & (EPH['ano_ed'] >= 0),
    EPH['ano_ed'] / EPH['ch06'],
    np.nan
)

#5. Movilidad geofráfica
EPH['movilidad_geo'] = EPH['ch15'].apply(lambda x: 1 if x in [2, 3, 4, 5] else 0)



#6 Responsable único
EPH['responsable_único'] = ((EPH['ch03'] == 1) & (EPH['ch07'].isin([3, 4, 5]))).astype(int)



#7 Responsabilidad materna:  cuenta la cantidad de hijos de cada madre. 
EPH['es_hijo'] = (EPH['ch03'] == 3).astype(int)
hijos_por_hogar = EPH.groupby(['codusu', 'nro_hogar'])['es_hijo'].sum().reset_index()
hijos_por_hogar.rename(columns={'es_hijo': 'hijosXhogar'}, inplace=True)

EPH = EPH.merge(hijos_por_hogar, on=['codusu', 'nro_hogar'], how='left')

EPH['es_madre'] = (
    (EPH['ch04'] == 2) &                         # mujer
    (EPH['ch03'].isin([1, 2])) &                 # jefa o cónyuge
    (EPH['hijosXhogar'] >= 1)                    # al menos un hijo en el hogar
).astype(int)

EPH['responsabilidad_materna'] = EPH['es_madre'] * EPH['hijosXhogar']


# 8. Hijos_x_edad
EPH['hijosXedad'] = np.where(EPH['ch03'].isin([1, 2]), EPH['ch06'] * EPH['hijosXhogar'], np.nan)

###################################################################
#============================== TP4 ==============================#
###################################################################

#Pasos previos, tratamos missings y creamos las muestras de entrenamiento y testeo

#Creamos nuevamente las bases respondieron y no respondieron pues debemos incluir las nuevas variables generadas por nosotros en EPH
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


#######################################################################
###################### METODOS DE REGULARIZACION ######################
#######################################################################


########## EJERCICIO 3 ##########


# Función para escalar X preservando la columna 'constante'
def escalar_X(X_train, X_test):
    columnas_a_escalar = X_train.columns.drop('constante')
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[columnas_a_escalar] = scaler.fit_transform(X_train[columnas_a_escalar])
    X_test_scaled[columnas_a_escalar] = scaler.transform(X_test[columnas_a_escalar])
    
    return X_train_scaled, X_test_scaled

# Estandarización para 2004
x_train_2004_scaled, x_test_2004_scaled = escalar_X(x_train_2004, x_test_2004)

# Estandarización para 2024
x_train_2024_scaled, x_test_2024_scaled = escalar_X(x_train_2024, x_test_2024)
   
  
def evaluar_modelo_logistico(X_train, y_train, X_test, y_test, year_label):
    print(f"\n--- Resultados para el año {year_label} ---")

    # Estandarización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for penalty_type, solver in [('l1', 'saga'), ('l2', 'lbfgs')]:
        print(f"\n--- Penalización {penalty_type.upper()} ---")

        # Inicializar y entrenar el modelo
        modelo = LogisticRegression(penalty=penalty_type, C=1.0, solver=solver, max_iter=10000)
        modelo.fit(X_train_scaled, y_train)

        # Predicción de probabilidades
        y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        y_pred = modelo.predict(X_test_scaled)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        conf_mat = confusion_matrix(y_test, y_pred)

        # Resultados
        print("Matriz de confusión:")
        print(conf_mat)
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC: {auc:.4f}")
        print("Reporte de clasificación:")
        print(classification_report(y_test, y_pred))

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'{penalty_type.upper()} (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Curva ROC - {penalty_type.upper()} - {year_label}")
        plt.legend()
        plt.grid(True)
        plt.show()


# Ejecutar para 2004
evaluar_modelo_logistico(x_train_2004, y_train_2004, x_test_2004, y_test_2004, "2004")

# Ejecutar para 2024
evaluar_modelo_logistico(x_train_2024, y_train_2024, x_test_2024, y_test_2024, "2024")


########## EJERCICIO 4 ##########

# KFold común para todos los casos
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Valores de lambda y Cs
n_values = np.arange(-5, 6)
lambdas = 10.0 ** n_values
Cs = 1 / lambdas


# ========== RIDGE 2004 ==========
logreg_cv = LogisticRegressionCV(Cs=Cs, cv=10, penalty='l2', solver='liblinear',
                                 scoring='accuracy', max_iter=1000)
logreg_cv.fit(x_train_2004_scaled, y_train_2004)
best_lambda_l2_2004 = 1 / logreg_cv.C_[0]
print("Best λ (Ridge 2004):", best_lambda_l2_2004)

avg_mse_ridge_2004 = []
for C in Cs:
    mse_folds = []
    for train_idx, test_idx in kf.split(x_train_2004_scaled):
        x_tr, x_te = x_train_2004_scaled.iloc[train_idx], x_train_2004_scaled.iloc[test_idx]
        y_tr, y_te = y_train_2004.iloc[train_idx], y_train_2004.iloc[test_idx]

        model = LogisticRegression(C=C, penalty='l2', solver='liblinear', max_iter=1000)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        mse_folds.append(mean_squared_error(y_te, pred))
    avg_mse_ridge_2004.append(np.mean(mse_folds))

plt.figure()
plt.plot(1 / np.array(Cs), avg_mse_ridge_2004, marker='o')
plt.xscale('log')
plt.xlabel("λ (lambda)")
plt.ylabel("MSE promedio")
plt.title("Ridge 2004: MSE vs λ")
plt.grid(True)
plt.tight_layout()
plt.show()


# ========== RIDGE 2024 ==========
logreg_cv = LogisticRegressionCV(Cs=Cs, cv=10, penalty='l2', solver='liblinear',
                                 scoring='accuracy', max_iter=1000)
logreg_cv.fit(x_train_2024_scaled, y_train_2024)
best_lambda_l2_2024 = 1 / logreg_cv.C_[0]
print("Best λ (Ridge 2024):", best_lambda_l2_2024)

avg_mse_ridge_2024 = []
for C in Cs:
    mse_folds = []
    for train_idx, test_idx in kf.split(x_train_2024_scaled):
        x_tr, x_te = x_train_2024_scaled.iloc[train_idx], x_train_2024_scaled.iloc[test_idx]
        y_tr, y_te = y_train_2024.iloc[train_idx], y_train_2024.iloc[test_idx]

        model = LogisticRegression(C=C, penalty='l2', solver='liblinear', max_iter=1000)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        mse_folds.append(mean_squared_error(y_te, pred))
    avg_mse_ridge_2024.append(np.mean(mse_folds))

plt.figure()
plt.plot(1 / np.array(Cs), avg_mse_ridge_2024, marker='o')
plt.xscale('log')
plt.xlabel("λ (lambda)")
plt.ylabel("MSE promedio")
plt.title("Ridge 2024: MSE vs λ")
plt.grid(True)
plt.tight_layout()
plt.show()


# ========== LASSO 2004 ==========
logreg_cv = LogisticRegressionCV(Cs=Cs, cv=10, penalty='l1', solver='liblinear',
                                 scoring='accuracy', max_iter=1000)
logreg_cv.fit(x_train_2004_scaled, y_train_2004)
best_lambda_l1_2004 = 1 / logreg_cv.C_[0]
print("Best λ (Lasso 2004):", best_lambda_l1_2004)

avg_mse_lasso_2004 = []
for C in Cs:
    mse_folds = []
    for train_idx, test_idx in kf.split(x_train_2004_scaled):
        x_tr, x_te = x_train_2004_scaled.iloc[train_idx], x_train_2004_scaled.iloc[test_idx]
        y_tr, y_te = y_train_2004.iloc[train_idx], y_train_2004.iloc[test_idx]

        model = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=1000)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        mse_folds.append(mean_squared_error(y_te, pred))
    avg_mse_lasso_2004.append(np.mean(mse_folds))

plt.figure()
plt.plot(1 / np.array(Cs), avg_mse_lasso_2004, marker='o')
plt.xscale('log')
plt.xlabel("λ (lambda)")
plt.ylabel("MSE promedio")
plt.title("Lasso 2004: MSE vs λ")
plt.grid(True)
plt.tight_layout()
plt.show()


# ========== LASSO 2024 ==========
logreg_cv = LogisticRegressionCV(Cs=Cs, cv=10, penalty='l1', solver='liblinear',
                                 scoring='accuracy', max_iter=1000)
logreg_cv.fit(x_train_2024_scaled, y_train_2024)
best_lambda_l1_2024 = 1 / logreg_cv.C_[0]
print("Best λ (Lasso 2024):", best_lambda_l1_2024)

avg_mse_lasso_2024 = []
for C in Cs:
    mse_folds = []
    for train_idx, test_idx in kf.split(x_train_2024_scaled):
        x_tr, x_te = x_train_2024_scaled.iloc[train_idx], x_train_2024_scaled.iloc[test_idx]
        y_tr, y_te = y_train_2024.iloc[train_idx], y_train_2024.iloc[test_idx]

        model = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=1000)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        mse_folds.append(mean_squared_error(y_te, pred))
    avg_mse_lasso_2024.append(np.mean(mse_folds))

plt.figure()
plt.plot(1 / np.array(Cs), avg_mse_lasso_2024, marker='o')
plt.xscale('log')
plt.xlabel("λ (lambda)")
plt.ylabel("MSE promedio")
plt.title("Lasso 2024: MSE vs λ")
plt.grid(True)
plt.tight_layout()
plt.show()



### Promedio de la proporción de variables descartadas en LASSO

## 2004
proporcion_descartadas_2004 = []

for C in Cs:
    proporciones = []
    for train_idx, test_idx in kf.split(x_train_2004_scaled):
        x_tr = x_train_2004_scaled.iloc[train_idx]
        y_tr = y_train_2004.iloc[train_idx]

        model = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=1000)
        model.fit(x_tr, y_tr)

        coef = model.coef_.ravel()
        proporciones.append(np.mean(coef == 0))  # Proporción de coeficientes en 0

    proporcion_descartadas_2004.append(np.mean(proporciones))

# Graficar
plt.figure()
plt.plot(1 / np.array(Cs), proporcion_descartadas_2004, marker='o')
plt.xscale('log')
plt.xlabel("λ (lambda)")
plt.ylabel("Proporción promedio de variables descartadas")
plt.title("Lasso 2004: Proporción de variables descartadas vs λ")
plt.grid(True)
plt.tight_layout()
plt.show()


## 2024
proporcion_descartadas_2024 = []

for C in Cs:
    proporciones = []
    for train_idx, test_idx in kf.split(x_train_2024_scaled):
        x_tr = x_train_2024_scaled.iloc[train_idx]
        y_tr = y_train_2024.iloc[train_idx]

        model = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=1000)
        model.fit(x_tr, y_tr)

        coef = model.coef_.ravel() 

        proporciones.append(np.mean(coef == 0))  # proporción de coeficientes en 0

    proporcion_descartadas_2024.append(np.mean(proporciones))

# Graficar
plt.figure()
plt.plot(1 / np.array(Cs), proporcion_descartadas_2024, marker='o')
plt.xscale('log')
plt.xlabel("λ (lambda)")
plt.ylabel("Proporción promedio de variables descartadas")
plt.title("Lasso 2024: Proporción de variables descartadas vs λ")
plt.grid(True)
plt.tight_layout()
plt.show()


########## EJERCICIO 5 ##########

### Variables descartadas para LASSO 2004

# Modelo final para 2004 con lambda = 100 ⇒ C = 0.01
modelo_lasso_final_2004 = LogisticRegression(C=0.01, penalty='l1', solver='liblinear', max_iter=1000)
modelo_lasso_final_2004.fit(x_train_2004_scaled, y_train_2004)

coef_2004 = modelo_lasso_final_2004.coef_.ravel()
nombres_vars_2004 = x_train_2004_scaled.columns

# Variables descartadas y seleccionadas
descartadas_2004 = nombres_vars_2004[coef_2004 == 0].tolist()
mantenidas_2004 = nombres_vars_2004[coef_2004 != 0].tolist()

print("2004 - Variables descartadas (coef = 0):")
print(descartadas_2004)

print("2004 - Variables mantenidas (coef ≠ 0):")
print(mantenidas_2004)




### Variables descartadas para LASSO 2024

# Modelo final para 2024 con lambda = 10 ⇒ C = 0.1
modelo_lasso_final_2024 = LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000)
modelo_lasso_final_2024.fit(x_train_2024_scaled, y_train_2024)

coef_2024 = modelo_lasso_final_2024.coef_.ravel()
nombres_vars_2024 = x_train_2024_scaled.columns

# Variables descartadas y seleccionadas
descartadas_2024 = nombres_vars_2024[coef_2024 == 0].tolist()
mantenidas_2024 = nombres_vars_2024[coef_2024 != 0].tolist()

print("2024 - Variables descartadas (coef = 0):")
print(descartadas_2024)

print("2024 - Variables mantenidas (coef ≠ 0):")
print(mantenidas_2024)


########## EJERCICIO 6 ##########

#Comparacion LASSO y RIDGE por año

# Función para entrenar y evaluar modelo logístico con penalización L1 o L2 

def evaluar_modelo_con_lambda(X_train, y_train, X_test, y_test, lambda_val, penalty_type, solver, year_label):
    C_val = 1 / lambda_val
    print(f"\n--- Año {year_label} | Penalización: {penalty_type.upper()} | Lambda: {lambda_val} ---")

    columnas_a_escalar = X_train.columns.drop('constante') if 'constante' in X_train.columns else X_train.columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[columnas_a_escalar] = scaler.fit_transform(X_train[columnas_a_escalar])
    X_test_scaled[columnas_a_escalar] = scaler.transform(X_test[columnas_a_escalar])

    modelo = LogisticRegression(penalty=penalty_type, C=C_val, solver=solver, max_iter=10000)
    modelo.fit(X_train_scaled, y_train)

    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    mse = mean_squared_error(y_test, y_proba)
    print(f"MSE de testeo: {mse:.4f}")

    return mse

lambdas = {
    'lasso_2004': 100,
    'lasso_2024': 10,
    'ridge_2004': 100,
    'ridge_2024': 100
}
    

# Evaluaciones

mse_lasso_2004 = evaluar_modelo_con_lambda(x_train_2004, y_train_2004, x_test_2004, y_test_2004,
                                           lambdas['lasso_2004'], 'l1', 'liblinear', "2004")
mse_ridge_2004 = evaluar_modelo_con_lambda(x_train_2004, y_train_2004, x_test_2004, y_test_2004,
                                           lambdas['ridge_2004'], 'l2', 'liblinear', "2004")

mse_lasso_2024 = evaluar_modelo_con_lambda(x_train_2024, y_train_2024, x_test_2024, y_test_2024,
                                           lambdas['lasso_2024'], 'l1', 'liblinear', "2024")
mse_ridge_2024 = evaluar_modelo_con_lambda(x_train_2024, y_train_2024, x_test_2024, y_test_2024,
                                           lambdas['ridge_2024'], 'l2', 'liblinear', "2024")

#####################################################
###################### ARBOLES ######################
#####################################################

#Ya realizamos la inspeccion de missing values para ejercicios anteriores.

########## EJERCICIO 7 ##########

def encontrar_mejor_profundidad(x_train, y_train, anio):
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    accuracies = []
    max_attributes = len(x_train.columns)
    depth_range = range(1, max_attributes + 1)

    for depth in depth_range:
        fold_accuracy = []
        tree_model = DecisionTreeClassifier(max_depth=depth, random_state=1)

        for train_idx, valid_idx in cv.split(x_train):
            x_train_fold = x_train.iloc[train_idx]
            x_valid_fold = x_train.iloc[valid_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_valid_fold = y_train.iloc[valid_idx]

            model = tree_model.fit(x_train_fold, y_train_fold)
            acc = model.score(x_valid_fold, y_valid_fold)
            fold_accuracy.append(acc)

        avg_acc = np.mean(fold_accuracy)
        accuracies.append(avg_acc)

    resultados = pd.DataFrame({
        "Max Depth": depth_range,
        "Average Accuracy": accuracies
    })

    best_idx = np.argmax(resultados["Average Accuracy"])
    best_depth = resultados.loc[best_idx, "Max Depth"]
    best_accuracy = resultados.loc[best_idx, "Average Accuracy"]

    print(f"\n Resultados para {anio}:")
    print(resultados.to_string(index=False))
    print(f"\n La mayor accuracy de cross-validation para {anio} es con max_depth = {best_depth}, y vale {best_accuracy:.4f}")
    
    return best_depth

# Aplicamos para ambos años (asumiendo que ya tenés estos datos cargados)
mejor_profundidad_2024 = encontrar_mejor_profundidad(x_train_2024, y_train_2024, 2024)
mejor_profundidad_2004 = encontrar_mejor_profundidad(x_train_2004, y_train_2004, 2004)


### Arboles de decision para 2004 y 2024
#2024 
# Cambiar 'ch06' por 'Edad' solo para graficar
x_train_2024_renamed = x_train_2024.rename(columns={'ch06': 'Edad'})

# Entrenar el árbol
decision_tree_2024 = DecisionTreeClassifier(max_depth=1, random_state=1).fit(x_train_2024_renamed, y_train_2024)

# Crear figura
fig_2024 = plt.figure(figsize=(10, 10))
ax_2024 = fig_2024.add_subplot(1, 1, 1)

# Dibujar árbol
plot_tree(decision_tree_2024,
          max_depth=1,
          impurity=True,
          feature_names=x_train_2024_renamed.columns,
          class_names=['Ocupado', 'Desocupado'],
          rounded=True,
          filled=True,
          ax=ax_2024)

        
fig_2024.savefig("Arbol_CART_2024_LILA.pdf", bbox_inches='tight')
plt.show()


# 2004
# Cambiar 'ch06' por 'Edad' solo para graficar
x_train_2004_renamed = x_train_2004.rename(columns={'ch06': 'Edad'})

# Entrenar el árbol
decision_tree_2004 = DecisionTreeClassifier(max_depth=1, random_state=1).fit(x_train_2004_renamed, y_train_2004)

# Crear figura
fig_2004 = plt.figure(figsize=(10, 10))
ax_2004 = fig_2004.add_subplot(1, 1, 1)

# Dibujar árbol
plot_tree(decision_tree_2004,
          max_depth=1,
          impurity=True,
          feature_names=x_train_2004_renamed.columns,
          class_names=['Ocupado', 'Desocupado'],
          rounded=True,
          filled=True,
          ax=ax_2004)


fig_2004.savefig("Arbol_CART_2004_LILA.pdf", bbox_inches='tight')
plt.show()


########## EJERCICIO 8 ##########
# Importancia de los predictores

# 2024
importancias = decision_tree_2024.feature_importances_
variables = x_train_2024.columns

# Crear DataFrame para ordenar y graficar
importancias_df = pd.DataFrame({
    'Variable': variables,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

# Graficar
plt.figure(figsize=(15, 10))
plt.barh(importancias_df['Variable'], importancias_df['Importancia'], color='#7e57c2')
plt.xlabel('Importancia')
plt.title('Importancia de cada predictor en el árbol CART (2024)')
plt.gca().invert_yaxis()  # Opcional, para que la más importante quede arriba
plt.tight_layout()
plt.savefig("Importancia_Predictores_2024.pdf")
plt.show()


# 2004
importancias = decision_tree_2004.feature_importances_
variables = x_train_2004.columns

# Crear DataFrame para ordenar y graficar
importancias_df = pd.DataFrame({
    'Variable': variables,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

# Graficar
plt.figure(figsize=(15, 10))
plt.barh(importancias_df['Variable'], importancias_df['Importancia'], color='#7e57c2')
plt.xlabel('Importancia')
plt.title('Importancia de cada predictor en el árbol CART (2004)')
plt.gca().invert_yaxis()  # Opcional, para que la más importante quede arriba
plt.tight_layout()
plt.savefig("Importancia_Predictores_2024.pdf")
plt.show()


########## EJERCICIO 9 ##########

# Matriz de confusión, la curva ROC, el AUC y el accuracy

def evaluar_arbol(X_test, y_test, modelo_arbol, year_label):
    print(f"\n--- Resultados para el año {year_label} ---")

    # Predicciones
    y_pred = modelo_arbol.predict(X_test)
    y_proba = modelo_arbol.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    conf_mat = confusion_matrix(y_test, y_pred)

    # Resultados
    print("Matriz de confusión:")
    print(conf_mat)
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'Arbol (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC - Árbol - {year_label}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Renombrar columnas para el test (igual que en el train)
x_test_2024_renamed = x_test_2024.rename(columns={'ch06': 'Edad'})
x_test_2004_renamed = x_test_2004.rename(columns={'ch06': 'Edad'})

# Evaluar árbol 2024
evaluar_arbol(x_test_2024_renamed, y_test_2024, decision_tree_2024, 2024)

# Evaluar árbol 2004
evaluar_arbol(x_test_2004_renamed, y_test_2004, decision_tree_2004, 2004)


# MSE test

# Predicciones
y_pred_2024 = decision_tree_2024.predict(x_test_2024_renamed)
y_pred_2004 = decision_tree_2004.predict(x_test_2004_renamed)

# MSE para arboles
mse_2024 = mean_squared_error(y_test_2024, y_pred_2024)
mse_2004 = mean_squared_error(y_test_2004, y_pred_2004)

print(f"MSE Árbol 2024: {mse_2024:.4f}")
print(f"MSE Árbol 2004: {mse_2004:.4f}")

# MSE para LASSO y RIDGE (esto lo hicimos antes para el ejercicio 3, linea 619)

mse_lasso_2004 = evaluar_modelo_con_lambda(x_train_2004, y_train_2004, x_test_2004, y_test_2004,
                                           lambdas['lasso_2004'], 'l1', 'liblinear', "2004")
mse_ridge_2004 = evaluar_modelo_con_lambda(x_train_2004, y_train_2004, x_test_2004, y_test_2004,
                                           lambdas['ridge_2004'], 'l2', 'liblinear', "2004")

mse_lasso_2024 = evaluar_modelo_con_lambda(x_train_2024, y_train_2024, x_test_2024, y_test_2024,
                                           lambdas['lasso_2024'], 'l1', 'liblinear', "2024")
mse_ridge_2024 = evaluar_modelo_con_lambda(x_train_2024, y_train_2024, x_test_2024, y_test_2024,
                                           lambdas['ridge_2024'], 'l2', 'liblinear', "2024")














