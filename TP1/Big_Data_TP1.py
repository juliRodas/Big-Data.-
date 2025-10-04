###     CÃ³digos iniciales    #########################################################

import pandas as pd

import os

os.getcwd()

os.chdir("/Users/Marcos/Desktop/Facu/Cuarto año/Big Data/GitHub/E337-Grupo1/TP1/Bases de Datos")
#os.chdir("/Users/rosarionamur/Desktop/Big Data/TP1/Bases de Datos")
#os.chdir("\\Users\\Usuario\\Desktop\\Big Data UdeSa\\GitHub\\E337-Grupo1\\TP1\\Bases de Datos")


#########   EJERCICIO 2A   #########################################################

#Abro y modifico la base de excel de 2024: nos quedamos solo con Gran BA 
Base2024= pd.read_excel("usu_individual_T124.xlsx")

Base2024 = Base2024[Base2024["AGLOMERADO"].isin([32, 33])]
Base2024 = Base2024[Base2024["AGLOMERADO"].isin([32, 33])]

Base2024.to_excel("usu_individual_T124_Modified.xlsx", index=False)


#Abro y modifico la base dta de 2004: nos quedamos solo con Gran BAy cambiamos a formato excel
Base2004 = pd.read_stata("Individual_t104.dta")

Base2004 = Base2004[Base2004["aglomerado"].isin(["Ciudad de Buenos Aires", "Partidos del GBA"])]

import numpy as np

Base2004["aglomerado"] = np.where(
    Base2004["aglomerado"] == "Ciudad de Buenos Aires", 32, 
    np.where(Base2004["aglomerado"] == "Partidos del GBA", 33, Base2004["aglomerado"]))

Base2004.to_excel("Individual_t104_Modified.xlsx", index=False)


#   REORGANIZAMOS LAS BASES DE DATOS PARA PODER APPENDEARLAS
import pandas as pd
import os
os.getcwd()
#os.chdir("/Users/Marcos/Desktop/Facu/Cuarto anÌo/Big Data/GitHub/E337-Grupo1/TP1/Bases de Datos")
os.chdir("/Users/rosarionamur/Desktop/Big Data/TP1/Bases de Datos")
#os.chdir("\\Users\\Usuario\\Desktop\\Big Data UdeSa\\GitHub\\E337-Grupo1\\TP1\\Bases de Datos")


Base2004 = pd.read_excel("Individual_t104_Modified.xlsx")
Base2024 = pd.read_excel("usu_individual_T124_Modified.xlsx")

# Uniformizamos los nombres de columnas a minúsculas
Base2004.columns = Base2004.columns.str.lower()
Base2024.columns = Base2024.columns.str.lower()
print(Base2024.columns)

# Seleccionamos las columnas comunes entre ambas bases
common_columns = Base2004.columns.intersection(Base2024.columns)

# Reordenamos las bases segÃºn las columnas comunes
Base2004 = Base2004[common_columns]
Base2024 = Base2024[common_columns]

# Concatenamos ambas bases en un Ãºnico DataFrame
BaseAppend1 = pd.concat([Base2004, Base2024], axis = 0, ignore_index=True)

# Exportamos el DataFrame combinado a un archivo xlsx
BaseAppend1.to_excel("BaseAppend1_Modified.xlsx", index=False)


#######  EJERCICIO 2B   ############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo
BaseAppend1 = pd.read_excel("BaseAppend1_Modified.xlsx")

# Seleccionamos las 10 variables de interés  
vars_interes = ["estado", "ch04", "ch06", "ch08", "ch09", "nivel_ed", "ipcf", "pp04g", "pp10a", "pp11t"]

# Diccionario para renombrar las variables
nuevos_nombres = {
    "estado": "Condición de actividad"	
    "ch04": "Sexo",
    "ch06": "Edad",
    "ch08": "Cobertura médica",
    "ch09": "Leer y escribir",
    "nivel_ed": "Educación",
    "ipcf": "Ingreso per cápita",
    "pp04g": "Lugar de trabajo",
    "pp10a": "Búsqueda laboral",
    "pp11t": "Seguro de desempleo"
}

# Renombrar las columnas
BaseAppend1  = BaseAppend1 .rename(columns=nuevos_nombres)

# Calcular el porcentaje de datos nulos por año con las variables de interés
percentage_data = BaseAppend1 .groupby('ano4')[list(nuevos_nombres.values())].apply(lambda x: x.isnull().mean() * 100)

# Crear el heatmap  
plt.figure(figsize=(12, 8))
sns.heatmap(percentage_data.T, annot=False, cmap="coolwarm", cbar_kws={'label': 'Porcentaje'})
plt.title("Datos faltantes en las variables de interés (por año)")
plt.xlabel("Año")
plt.ylabel("Variables")
plt.show()


####### EJERCICIO 2C/D   ############################################################

import pandas as pd
import numpy as np

# Cargar el DataFrame
BaseAppend1 = pd.read_excel("BaseAppend1_Modified.xlsx")

##################################################################################################################
#Antes de empezar a modificar nuestras variables con criterios específicos para cada una, hemos tomado una decisión arbitraria que aplica a toda la base: Vamos a reemplazar todos los "Ns./Nr." por missing values. Además, empezamos a unificar los lenguajes entre ambos diccionarios, modificamos los "Si" por 1 y los "No" por 
BaseAppend1 = BaseAppend1.replace({"Sí": 1, "Si": 1, "No": 2, "Ns./Nr.": np.nan,"Ns./Nr": np.nan })
#Observacion: no podemos poner todos los 0 como missing pues en la base ciertas variables -muy pocas- lo utilizan como un valor en si en vez de como missing
               
#####################################################################################################################################                    
# Primer lista de variables a modificar a la que solo le cambiamos los 0, 9, 99, 999, 9999 y -9 por missing
variables_a_cambiar = ['pp07g1', 'pp07g2', 'pp07g3', 'pp07g4', 'pp07h', 'pp07i', 'pp07j', 'pp07k',
                       'pp09a', 'pp10a', 'pp10c', 'pp10d', 'pp10e', 'pp11a', 'pp11c99', 'pp11l', 
                       'pp11l1', 'pp11m', 'pp11n', 'pp11p', 'pp11q', 'pp11r', 'pp11s', 'pp11t', 
                       'rdecindr', 'gdecindr', 'adecindr', 'v2_m', 'v3_m', 'v4_m', 'v5_m', 'v8_m', 
                       'v9_m', 'v10_m', 'v11_m', 'v18_m', 'tot_p12', 'p21', 'pp08d1', 'pp08f1', 
                       'pp08f2', 'pp08j1', 'pp08j2', 'pp05c_2', 'pp05c_3', 'pp05e', 'pp05f', 
                       'pp06a', 'intensi', 'pp06e', 'pp07f2', 'pp07f3', 'pp07f4', "ch09", "ch11", 
                       "pp07c", "ch13", "ch16", "t_vi", "v21_m", "nivel_ed", "cat_ocup", "pp11g_ano", 
                       "pp11g_mes", "pp11g_dia", "pp11b2_mes", "pp11b2_ano", "pp11b2_dia", "pp11d_cod", 
                       "cat_inac", "pp3f_tot", "pp03g", "pp03h", "pp03d", "pp03i", "pp03j", "intensi", 
                       "ch07", "pp03c", "ch14", "pp3e_tot", "pp04b3_ano" , "pp04b3_dia", "pp04b3_mes", 
                       "pp04c", "pp04c99", "pp05b2_mes", "pp05b2_ano", "pp05b2_dia", "pp05h", "pp07d", 
                       "pp07f1", "pp08j3", "pp11b2_mes", "pp11c", "pp08j3", "p47t", "v12_m","ch10"]


#Modificamos los 0, 9, 99, 999, 9999 y -9 por missing values para un grupo de variables en las que estos valores no tienen sentido
def replace_with_nan(df, variables_a_cambiar):
    
    for var in variables_a_cambiar:
        if var in df.columns:
            df[var] = df[var].replace({0: np.nan, 9: np.nan, 99: np.nan, 999: np.nan, 9999: np.nan, -9: np.nan})
    return df  # Se devuelve el DataFrame modificado


# Aplicar la funciÃ³n correctamente
BaseAppend1 = replace_with_nan(BaseAppend1, variables_a_cambiar)


##################################################################################################################
# Segunda lista de variables a modificar a la que solo le cmabiamos los 0 por missing -pero que no les sacamos los 9s-
variables_pp_a_cambiar = ['pp02c1', 'pp02c2','pp02c3','pp02c4','pp02c5','pp02c6','pp02c7','pp02c8','pp02e','pp02h',
                          'pp02i','pp03c','pp03d','pp3e_tot','pp3f_tot','pp03g','pp03h','pp03i','pp03j','pp04a','pp04b1',
                          'pp04b1','pp04b2', 'pp04c','pp04c99','pp04g', 'pp05c_1', 'pp05c_2','pp05c_3','pp05e','pp05f','pp05h',
                          'pp06a','pp06c','pp06d','pp06e','pp06h','pp07a','pp07d','pp07e','pp07f1','pp07f2','pp07f3',
                          'pp07f4','pp07f5','pp07g_59','pp07g2','pp08d1','pp08d4','pp08f1','pp08f2','pp08j1','pp08j2',
                          'pp09b','pp09c','pp11b1','pp11c','pp11o','pp11o', "ch11", "decifr", "rdecifr", "gdecifr", "pdecifr", 
                          "adecifr", "deccfr", "rdeccfr", "gdeccfr", "adeccfr", "pp09c", "v19_am", "rdecocur", "adecocur",
                          "gdecocur", "rdecocur", "decocur", "decindr"]

#Ahora cambiamos los 0 por missings en las variables restantes
def replace_with_nan(df, variables_a_cambiar):
    for var in variables_pp_a_cambiar:
        if var in df.columns:
            df[var] = df[var].replace({0: np.nan})
    return df  # Se devuelve el DataFrame modificado


# Aplicar la funciÃ³n correctamente
BaseAppend1 = replace_with_nan(BaseAppend1, variables_pp_a_cambiar)

##################################################################################################################   
    

#Esta función nos permite elgir a nsootros el valor a modifcar en cada variables separando por casos específicos 
def reemplazar_por_nan(df, variable, valor_a_reemplazar):
    # Reemplaza el valor especÃ­fico por NaN en la columna indicada
    df[variable] = df[variable].replace(valor_a_reemplazar, np.nan)
    

reemplazar_por_nan(BaseAppend1, "pp04b2", 31) 
reemplazar_por_nan(BaseAppend1, "pp04b1", 2) 
reemplazar_por_nan(BaseAppend1, "pp06c", [-7,-8,-9]) 
reemplazar_por_nan(BaseAppend1, "ch06", [-1]) 
reemplazar_por_nan(BaseAppend1, "pp06d", [-7,-8,-9])
reemplazar_por_nan(BaseAppend1, "ch14", 99)
reemplazar_por_nan(BaseAppend1, "decocur", 00)
reemplazar_por_nan(BaseAppend1, "rdecocur", 00)
reemplazar_por_nan(BaseAppend1, "gdecocur", 00)
reemplazar_por_nan(BaseAppend1, "adecocur", 00)
reemplazar_por_nan(BaseAppend1, "ch14", 00)


#####################################################################################################################################

#A continuación, haremos todos los cmabios necesarios -detallados en el word- para poder unificar los valores que toma cada variables pues en 2004 y 2024 se utilizaron distintos criterios.

#Para la variable edad (que tiene datos en sting pero el diccionario no tiene codigos asociados) la corregimos manualmente
BaseAppend1["ch06"].unique() 
BaseAppend1['ch06'] = BaseAppend1['ch06'].replace({'Menos de 1 año': 0, '98 y más años': 99})


##Cambios de Marcos
BaseAppend1['trimestre'] = BaseAppend1['trimestre'].replace({ "1er. Trimestre": 1})
BaseAppend1['region'] = BaseAppend1['region'].replace({ "Gran Buenos Aires": 1})
BaseAppend1['ch03'] = BaseAppend1['ch03'].replace({ "Jefe": 1, 'Cónyuge/Pareja': 2, 'Hijo/Hijastro': 3, 'Yerno/Nuera' : 4, 'Nieto' : 5, 'Madre/Padre':6, "Suegro":7, "Hermano":8, "Otros familiares":9, "No familiares":10  })
BaseAppend1['ch04'] = BaseAppend1["ch04"].replace({ "Varón": 1, 'Mujer': 2 })
BaseAppend1['ch07'] = BaseAppend1["ch07"].replace({ "Unido": 1, 'Casado': 2, 'Separado o divorciado': 3, 'Viudo' : 4, 'Soltero' : 5})
BaseAppend1['ch08'] = BaseAppend1["ch08"].replace({ "Obra social (incluye PAMI)": 1, 'Mutual/Prepaga/Servicio de emergencia': 2, 'Planes y seguros públicos': 3, 'No paga ni le descuentan' : 4, 'Obra social y mutual/prepaga/servicio de emergencia' : 12, 'Mutual/prepaga/servicio de emergencia/planes y seguros públi':23 })
BaseAppend1['ch09'] = BaseAppend1['ch09'].replace({'Sí­': 1, 'No': 2, "Menor de 2 años": 3})
BaseAppend1['ch10'] = BaseAppend1['ch10'].replace({'Sí, asiste': 1, 'No asiste, pero asistió': 2, 'Nunca asistió': 3})
BaseAppend1['ch11'] = BaseAppend1["ch11"].replace({ "Público": 1, 'Privado': 2})
BaseAppend1['ch12'] = BaseAppend1['ch15'].replace({'Jardí­n/Preescolar': 1, 'Primario': 2, 'EGB': 3, "Secundario" : 4, "Polimodal" : 5, 'Terciario':6,  "Universitario": 7, "Posgrado Universitario":8, "Educación especial (discapacitado)":9 })
BaseAppend1['ch14'] = BaseAppend1["ch14"].replace({ 1: "01", 2:" 02", 3:" 03", 4: "04", 5: "05", 6: "06",  7:" 07", 8: "08", 9:" 09"})
BaseAppend1['ch15'] = BaseAppend1['ch15'].replace({'En esta localidad': 1, 'En otra localidad': 2, 'En otra provincia (especificar)': 3, 'En un país limítrofe' : 4, 'En otro país' : 5})
BaseAppend1['ch16'] = BaseAppend1['ch16'].replace({ "En esta localidad": 1, 'En otra localidad de esta provincia': 2, 'En otra provincia (especificar)': 3, 'En un país limítrofe' : 4, 'En otro país' : 5, 'No había nacido':6,  })
BaseAppend1['nivel_ed'] = BaseAppend1['nivel_ed'].replace({ "Primaria Incompleta (incluye educación especial)": 1, 'Primaria Completa': 2, 'Secundaria Incompleta': 3, 'Secundaria Completa' : 4, 'Superior Universitaria Incompleta' : 5, 'Superior Universitaria Completa':6, "Sin instrucción":7 })
BaseAppend1['estado'] = BaseAppend1['estado'].replace({'Desocupado': 2, 'Entrevista individual no realizada (no respuesta al cuestion': 0, 'Inactivo': 3, 'Ocupado' : 1, 'Menor de 10 años' : 4 })
BaseAppend1['cat_ocup'] = BaseAppend1['cat_ocup'].replace({ "Patrón": 1, 'Cuenta propia': 2, 'Obrero o empleado': 3, 'Trabajador familiar sin remuneración' : 4  })
BaseAppend1['cat_inac'] = BaseAppend1['cat_inac'].replace({ "Jubilado/pensionado": 1, 'Rentista': 2, 'Estudiante': 3, 'Ama de casa' : 4, 'Menor de 6 años' : 5, 'Discapacitado':6, "Otros":7  })
BaseAppend1['pp02e'] = BaseAppend1['pp02e'].replace({'Ya tiene trabajo asegurado': 2, 'Se cansó de buscar trabajo': 3, 'Hay poco trabajo en esta época del año' : 4, 'Por otras razones' : 5})
BaseAppend1['pp03c'] = BaseAppend1['pp03c'].replace({ "...un sólo empleo/ocupación/actividad?": 1, '...más de un empleo/ocupación/actividad?': 2 })
BaseAppend1['pp03h'] = BaseAppend1["pp03h"].replace({ "...podía trabajarlas esa semana?": 1, '...podía empezar a trabajarlas en dos semanas a más tardar?': 2, '...no podía trabajar más horas?': 3})
BaseAppend1['intensi'] = BaseAppend1["intensi"].replace({ "Subocupación horaria No Demandante": 1, "Subocupación horaria No Demandante":1,'Ocupación plena': 2, 'Sobreocupación horaria': 3, 'Ocupado que no trabajó en la semana' : 4 })
BaseAppend1['intensi'] = BaseAppend1["intensi"].replace({ "Subocupación horaria Demandante": 1})
BaseAppend1['pp04a'] = BaseAppend1["pp04a"].replace({ "...estatal?": 1, '...privada?': 2, '...de otro tipo? (especificar)': 3})
BaseAppend1['pp04b1'] = BaseAppend1["pp04b1"].replace({ "Casa de familia": 1})

## Cambios de Juli
BaseAppend1['pp04c'] = BaseAppend1['pp04c'].replace({ "1 persona": 1, '2 personas': 2, '3 personas': 3, '4 personas' : 4, '5 personas' : 5, 'de 6 a 10 personas':6, 'de 11 a 25 personas':7, 'de 26 a 40 personas':8, 'de 41 a 100 personas':9, 'de 101 a 200 personas':10, 'de 201 a 500 personas':11, 'más de 500 personas':12})  
BaseAppend1['pp04c99'] = BaseAppend1['pp04c99'].replace({ "Hasta 5": 1, 'De 6 a 40': 2, 'Más de 40': 3})  
BaseAppend1['pp04g'] = BaseAppend1['pp04g'].replace({ "En un local/oficina/establecimiento/negocio/taller/chacra/fi": 1, 'En puesto o kiosco fijo callejero': 2, 'En vehículos: bicicleta/moto/auto/barco/bote (no incluye ser': 3, 'En vehículo para transporte de personas y mercaderías-aéreo,' : 4, 'En obras en construcción, de infraestructura, minería o simi' : 5, 'En esta vivienda':6,'En la vivienda del socio o del patrón':7,'En el domicilio/local de los clientes':8,'En la calle/espacios públicos/ambulante/de casa en casa/pues':9,'En otro lugar':10})
BaseAppend1['pp05c_1'] = BaseAppend1['pp05c_1'].replace({ "Propio (del negocio)": 1, 'Prestado/alquilado': 2, 'No tiene': 3})
BaseAppend1['pp05c_2'] = BaseAppend1['pp05c_2'].replace({ "Propio (del negocio)": 1, 'Prestado/alquilado': 2, 'No tiene': 3})
BaseAppend1['pp05c_3'] = BaseAppend1['pp05c_3'].replace({ "Propio (del negocio)": 1, 'Prestado/alquilado': 2, 'No tiene': 3})
BaseAppend1['pp05f'] = BaseAppend1['pp05f'].replace({ "Distintos clientes? (incluye público en general)": 6, 'Un solo cliente? (persona, empresa)': 7})
BaseAppend1['pp05h'] = BaseAppend1['pp05h'].replace({ "Menos de un mes": 1, 'De 1 a 3 meses': 2, 'Más de 3 a 6 meses': 3, 'Más de 6 meses a 1 año': 4, 'Más de 1 a 5 años': 5, 'Más de 5 años':6})
BaseAppend1['pp06e'] = BaseAppend1['pp06e'].replace({ "Es una sociedad jurídicamente constituida? (S.A., S.R.L., Co": 1, 'Es una sociedad de otra forma legal?': 2, 'O es una sociedad convenida de palabra?': 3})
BaseAppend1['pp07a'] = BaseAppend1['pp07a'].replace({ "Menos de un mes": 1, '1 a 3 meses': 2, 'Más de 3 a 6 meses': 3, 'Más de 6 a 12 meses': 4, 'Más de 1 a 5 años': 5, 'Más de 5 años':6})
BaseAppend1['pp07c'] = BaseAppend1['pp07c'].replace({ "Sí (incluye changa, trabajo transitorio, por tarea u obra, s": 1, 'No (Incluye permanente, fijo, estable, de planta)': 2})
BaseAppend1['pp07d'] = BaseAppend1['pp07d'].replace({ "Sólo fue esa vez/sólo cuando lo llaman": 1, 'Hasta 3 meses': 2, 'Más de 3 a 6 meses': 3, 'Más de 6 a 12 meses': 4, 'Más de 1 año': 5})
BaseAppend1['pp07e'] = BaseAppend1['pp07e'].replace({ "...un plan de empleo?": 1, '...un período de prueba?': 2, '...una beca/pasantía/aprendizaje?': 3, 'Ninguno de estos': 4})
BaseAppend1['pp07e'] = BaseAppend1['pp07e'].replace({ "...un plan de empleo?": 1, '...un período de prueba?': 2, '...una beca/pasantía/aprendizaje?': 3, 'Ninguno de estos': 4})

##Cambios de RO
BaseAppend1['pp07f5'] = BaseAppend1['pp07f5'].replace({ '5': 1 })
BaseAppend1['pp07g_59'] = BaseAppend1['pp07g_59'].replace({ '5': 1})
BaseAppend1['pp07j'] = BaseAppend1['pp07j'].replace({ '...de día? (mañana/tarde)': 1, '...de noche': 2, '...de otro tipo? (rotativo, día y noche, guardias con franco': 3})
BaseAppend1['pp07k'] = BaseAppend1['pp07k'].replace({ '...entrega una factura?': 3, '...le dan recibo con sello/membrete/firma del empleador?': 1, '...le dan un papel/recibo sin nada?' : 2, '...no le dan ni entrega nada?' : 4, 'no cobra, es trabajador sin pago, ad-honorem': 5  })
BaseAppend1['pp09a'] = BaseAppend1['pp09a'].replace({ "Ciudad de Buenos Aires": 1, 'Partidos del GBA': 2, 'Ambos': 3, 'En otro lugar' : 4 })
BaseAppend1['pp10a'] = BaseAppend1['pp10a'].replace({ '...de 1 a 3 meses?': 2, '...más de 1 año?':  5, '...más de 3 a 6 meses?': 3, '...más de 6 a 12 meses?' : 4, '...menos de 1 mes?' : 1 })
BaseAppend1['pp10e'] = BaseAppend1['pp10e'].replace({ '...de 1 a 3 meses?': 2, '...más de 1 a 3 años?': 5, '...más de 3 a 6 meses?': 3, '...más de 3 años?' : 6, '...más de 6 a 12 meses?' : 4, '...menos de 1 mes?': 1  })
BaseAppend1['pp11a'] = BaseAppend1['pp11a'].replace({ '...de otro tipo?': 3, '...estatal?': 1, '...privada?': 2})
BaseAppend1['pp11b1'] = BaseAppend1['pp11b1'].replace({ 'Casa de familia': 1 })
BaseAppend1['pp11c'] = BaseAppend1['pp11c'].replace({ '1 persona': 1, '2 personas': 2, '3 personas': 3, '4 personas' : 4, '5 personas' : 5, 'de 101 a 200 personas': 10, 'de 11 a 25 personas': 7, 'de 201 a 500 personas': 11, 'de 26 a 40 personas': 8, 'de 41 a 100 personas': 9, 'de 6 a 10 personas': 6 , 'más de 500 personas':  12 })
BaseAppend1['pp11c99'] = BaseAppend1['pp11c99'].replace({ 'De 6 a 40': 2, 'Hasta 5': 1, 'Más de 40': 3})
BaseAppend1['pp11l'] = BaseAppend1['pp11l'].replace({ 'Falta de clientes/clientes que no pagan': 1, 'Falta de capital/equipamiento': 2, 'Causas personales (matrimonio, embarazo, cuidado de hijos o': 7, 'Otras causas laborales (especificar)' : 5, 'Tenía gastos demasiado altos' : 4, 'Trabajo estacional': 3  })
BaseAppend1['pp11l1'] = BaseAppend1['pp11l1'].replace({ '...un trabajo permanente, fijo estable, de planta, etc.?': 2, '...una changa, trabajo transitorio, por tarea u obra, suplen': 1})
BaseAppend1['pp11m'] = BaseAppend1['pp11m'].replace({ '...otro tipo de trabajo?': 3, '...un período de prueba?': 2, '...un plan de empleo?': 1 })
BaseAppend1['pp11o'] = BaseAppend1['pp11o'].replace({ 'Despido/cierre': 1, 'Fin del trabajo temporario/estacional': 4, 'Le pagaban poco/no le pagaban': 5, 'Malas relaciones laborales/malas condiciones de trabajo (ins' : 6, 'Otras causas laborales' : 8, 'Por jubilación': 3, 'Por razones personales': 9, 'Por retiro voluntario del sector público': 2 , 'Renuncia obligada/pactada': 7 })
BaseAppend1['decocur'] = BaseAppend1['decocur'].replace({ '01': 1, '02': 2, '03': 3, '04' : 4, '05' : 5, '06': 6, '07': 7, '08': 8, '09': 9  })
BaseAppend1['rdecocur'] = BaseAppend1['rdecocur'].replace({ '01': 1, '02': 2, '03': 3, '04' : 4, '05' : 5, '06': 6, '07': 7, '08': 8, '09': 9  })
BaseAppend1['gdecocur'] = BaseAppend1['gdecocur'].replace({ '01': 1, '02': 2, '03': 3, '04' : 4, '05' : 5, '06': 6, '07': 7, '08': 8, '09': 9  })
BaseAppend1['adecocur'] = BaseAppend1['adecocur'].replace({ '01': 1, '02': 2, '03': 3, '04' : 4, '05' : 5, '06': 6, '07': 7, '08': 8, '09': 9  })


##Guardamos los cmabios realizados 
BaseAppend1.to_excel("BaseEPH_limpia.xlsx", index=False)

#Chequeamos que los todas las variables hayan quedado homogenizadas

for var in BaseAppend1:
    valores_var= BaseAppend1[var].value_counts()
    print(valores_var)
    

#######  EJERCICIO 2E   ############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el archivo
BaseEPH_limpia = pd.read_excel("BaseEPH_limpia.xlsx")

# Seleccionamos las 10 variables de interés  
vars_interes = ["estado", "ch04", "ch06", "ch08", "ch09", "nivel_ed", "ipcf", "pp04g", "pp10a", "pp11t"]

# Diccionario para renombrar las variables
nuevos_nombres = {
    "estado": "Condición de actividad"	
    "ch04": "Sexo",
    "ch06": "Edad",
    "ch08": "Cobertura médica",
    "ch09": "Leer y escribir",
    "nivel_ed": "Educación",
    "ipcf": "Ingreso per cápita",
    "pp04g": "Lugar de trabajo",
    "pp10a": "Búsqueda laboral",
    "pp11t": "Seguro de desempleo"
}

# Renombrar las columnas
BaseEPH_limpia = BaseEPH_limpia.rename(columns=nuevos_nombres)

# Calcular el porcentaje de datos nulos por año con las variables de interés
percentage_data = BaseEPH_limpia.groupby('ano4')[list(nuevos_nombres.values())].apply(lambda x: x.isnull().mean() * 100)

# Crear el heatmap  
plt.figure(figsize=(12, 8))
sns.heatmap(percentage_data.T, annot=False, cmap="coolwarm", cbar_kws={'label': 'Porcentaje'})
plt.title("Datos faltantes en las variables de interés (por año)")
plt.xlabel("Año")
plt.ylabel("Variables")
plt.show()


####### Parte 2   ############################################################
#######  EJERCICIO 3   ############################################################

# Cargar el archivo
BaseEPH_limpia = pd.read_excel("BaseEPH_limpia.xlsx")

#Revisamos que nuestra variable tenga solo los valores 1 y 2 (para hombre y mujer, respectivamente)
BaseEPH_limpia["ch04"].unique()

#Lo mismo para el año
BaseEPH_limpia["ano4"].unique()


# Creamos un gráfico de barras para mostrar la composición por sexo en 2004 y 2024
composicion_sexo = BaseEPH_limpia.groupby(['ano4', 'ch04']).size().unstack()

# Graficamos la composición por sexo con barras apiladas
composicion_sexo.plot(kind='bar', stacked=True, figsize=(8, 7), 
                      color=['#3CB371', '#9370DB'], width=0.4)  

# Etiquetas y título del gráfico
plt.xlabel('Año', fontsize=10)  
plt.ylabel('Observaciones', fontsize=10)
plt.title('Composición por Sexo en 2004 y 2024', fontsize=12)
plt.legend(title='Sexo', labels=['Hombre', 'Mujer'], fontsize=10)
plt.xticks(rotation=0)  
plt.yticks(fontsize=10)
plt.tight_layout()

# Mostramos el gráfico
plt.show()

#######  EJERCICIO 4   ############################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BaseEPH_limpia = pd.read_excel("BaseEPH_limpia.xlsx")

variables = ["ch04", "ch06", "ch07", "ch08", "nivel_ed", "estado", "cat_inac", "ipcf"]

BaseEPH_limpia['ano4'] = pd.to_numeric(BaseEPH_limpia['ano4'], errors='coerce')


#Vamos a hacer dummies para las variables categgoricas 
#Para ello, haremos diccionariamos en donde cada numero este asociado a se correspondiente categoria de modo tal que la matriz de varianzas sea interpretable y vaya en linea con los tres principios de visualización de datos discutidos en la Clase
#Cabe destacar que, al haber hecho una profunda limpieza inciial, debemos reintroducir las categorias asociadas con cada -acorde a los criterios de los diccionarios de la EPH- para realizar esta consigna
dicc_ch07 = {
    1: 'Unido',
    2: 'Casado',
    3: 'Separado o divorciado',
    4: 'Viudo',
    5: 'Soltero'
}

dicc_ch08 = {
    1: 'Obra social (incluye PAMI)',
    2: 'Mutual/Prepaga/Servicio de emergencia',
    3: 'Planes y seguros públicos',
    4: 'No paga ni le descuentan',
    9: 'Ns./Nr.',
    12: 'Obra social y mutual/prepaga/servicio de emergencia'
}

dicc_nivel_ed = {
    1: 'Primaria Incompleta (incluye educación especial)',
    2: 'Primaria Completa',
    3: 'Secundaria Incompleta',
    4: 'Secundaria Completa',
    5: 'Superior Universitaria Incompleta',
    6: 'Superior Universitaria Completa',
    7: 'Sin instrucción'
}

dicc_estado = {
    0: 'Entrevista individual no realizada (no respuesta al cuestion)',
    1: 'Ocupado',
    2: 'Desocupado',
    3: 'Inactivo',
    4: 'Menor de 10 años'
}

dicc_cat_inac = {
    1: 'Jubilado/pensionado',
    2: 'Rentista',
    3: 'Estudiante',
    4: 'Ama de casa',
    5: 'Menor de 6 años',
    6: 'Discapacitado',
    7: 'Otros'
}


# Generamos variables dummy para cada una de las categorías presentes en las columnas seleccionadas
 # Para la variable ch07
for i, j in dicc_ch07.items():  
    BaseEPH_limpia[j] = BaseEPH_limpia['ch07'].apply(lambda x: 1 if x == i else 0)

# Para la variable ch08
for i, j in dicc_ch08.items():  
    BaseEPH_limpia[j] = BaseEPH_limpia['ch08'].apply(lambda x: 1 if x == i else 0)

# Para la variable nivel_ed
for i, j in dicc_nivel_ed.items():  
    BaseEPH_limpia[j] = BaseEPH_limpia['nivel_ed'].apply(lambda x: 1 if x == i else 0)

# Para la variable estado
for i, j in dicc_estado.items():  
    BaseEPH_limpia[j] = BaseEPH_limpia['estado'].apply(lambda x: 1 if x == i else 0)

# Para la variable cat_inac
for i, j in dicc_cat_inac.items():  
    BaseEPH_limpia[j] = BaseEPH_limpia['cat_inac'].apply(lambda x: 1 if x == i else 0)


# Definimos las variables que queremos incluir en la matriz de correlación 
from itertools import chain

variables = list(chain(
    dicc_ch07.values(), 
    dicc_ch08.values(), 
    dicc_nivel_ed.values(), 
    dicc_estado.values(), 
    dicc_cat_inac.values()
))


# Creamos una lista con los nombres truncados de las variables para facilitar su visualización
variables_truncadas = [col[:24] for col in variables]

# Truncamos los nombres de todas las columnas del DataFrame a un máximo de 24 caracteres para uniformar el formato
BaseEPH_limpia.columns = [col[:24] for col in BaseEPH_limpia.columns]


# Filtramos los datos por año y seleccionamos las variables correctas
df_2004 = BaseEPH_limpia.loc[BaseEPH_limpia['ano4'] == 2004, variables_truncadas].drop(['Entrevista individual no', 'Ns./Nr.'], axis=1)
df_2024 = BaseEPH_limpia.loc[BaseEPH_limpia['ano4'] == 2024, variables_truncadas].drop(['Entrevista individual no', 'Ns./Nr.'], axis=1)

# Calcular las matrices de correlación
corr_matrix_2004 = df_2004.corr()
corr_matrix_2024 = df_2024.corr()

# Graficamos la matriz de correlación para el año 2004
plt.figure(figsize=(20, 16))  # Ajustamos el tamaño de la figura para que sea más legible
sns.heatmap(corr_matrix_2004, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Matriz de Correlación (2004)')
plt.xticks(rotation=45, ha='right')  # Rotamos las etiquetas de los ejes para mejor visualización
plt.yticks(rotation=0)
plt.show()

# Graficamos la matriz de correlación para el año 2024
plt.figure(figsize=(20, 16))  # Ajustamos el tamaño de la figura
sns.heatmap(corr_matrix_2024, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Matriz de Correlación (2024)')
plt.xticks(rotation=45, ha='right')  # Rotamos las etiquetas de los ejes para mejor visualización
plt.yticks(rotation=0)
plt.show()

#######  Parte III ##########

#Ejercicio 5

#Medimos la cantidad de desocupados  e inactivos 
desocupados_2004 = len(BaseEPH_limpia[(BaseEPH_limpia['estado'] == 2) & (BaseEPH_limpia['ano4'] == 2004)])  
desocupados_2024 = len(BaseEPH_limpia[(BaseEPH_limpia['estado'] == 2) & (BaseEPH_limpia['ano4'] == 2024)])  
inactivos_2004 = len(BaseEPH_limpia[(BaseEPH_limpia['estado'] == 3) & (BaseEPH_limpia['ano4'] == 2004)])  
inactivos_2024 = len(BaseEPH_limpia[(BaseEPH_limpia['estado'] == 3) & (BaseEPH_limpia['ano4'] == 2024)])  

media_ipcf_por_estado = BaseEPH_limpia.groupby(['estado', 'ano4'])['ipcf'].mean()


# Mostramos los resultados
print(f'desocupados en 2004: {desocupados_2004}')
print(f'desocupados en 2024: {desocupados_2024}')
print(f'inactivos en 2004: {inactivos_2004}')
print(f'inactivos en 2024: {inactivos_2024}')
print(media_ipcf_por_estado)


#Ejercicio 6

#Contamos la cantidad de personas que no respondieron 
no_responde_estado = (BaseEPH_limpia['estado']==0).sum()

no_responde_2004 = ((BaseEPH_limpia['estado'] == 0) & (BaseEPH_limpia['ano4'] == 2004)).sum()
no_responde_2024 = ((BaseEPH_limpia['estado'] == 0) & (BaseEPH_limpia['ano4'] == 2024)).sum()

print(f"No respondieron en 2004: {no_responde_2004}")
print(f"No respondieron en 2024: {no_responde_2024}")

#Creamos la base de datos con los que si respondieron 
respondieron = BaseEPH_limpia[BaseEPH_limpia['estado']!=0]

#Creamos la base de datos con los que no respondieron 
norespondieron = BaseEPH_limpia[BaseEPH_limpia['estado']==0]

#Guardamos las bases de datos
respondieron.to_excel("respondieron.xlsx", index=False)
norespondieron.to_excel("norespondieron.xlsx", index=False)


#Ejercicio 7
#Agregamos a la base respondieron una columna llamada “PEA”(Población Económicamente Activa) que toma valor 1 si están ocupados o desocupados en ESTADO
respondieron['PEA'] = respondieron['estado'].apply(lambda x: 1 if x in [1, 2] else 0)


# Realizamos un gráfico de barras mostrando la composición por PEA para 2004 y 2024
PEA = respondieron.groupby(['ano4', 'PEA']).size().unstack()

# Graficamos la composición por sexo con barras apiladas
PEA.plot(kind='bar', stacked=True, figsize=(8, 7), 
                      color=['#FF6347', '#FFD700'], width=0.4)  

# Etiquetas y título del gráfico
plt.xlabel('Año', fontsize=10)  
plt.ylabel('Cantidad', fontsize=10)
plt.title("Composición de PEA en 2004 y 2024", fontsize=12)
plt.legend(title='Estado', labels=["Activa", 'Inactiva'], fontsize=10)
plt.xticks(rotation=0)  
plt.yticks(fontsize=10)
plt.tight_layout()

# Mostramos el gráfico
plt.show()




#Ejercicio 8

#Creamos una #Agregamos a la base respondieron una columna llamada “PET” (Población en Edad para Trabajar) que toma valor 1 si están si están la persona tiene entre 15 y 65 años cumplidos.
respondieron["ch06"] = pd.to_numeric(respondieron["ch06"], errors="coerce")
respondieron["PET"] = respondieron["ch06"].apply(lambda x: 1 if 15 <= x <= 65 else 0)

PET = respondieron.groupby(['ano4', 'PET']).size().unstack()


# Graficamos la composición por sexo con barras apiladas
PET.plot(kind='bar', stacked=True, figsize=(8, 7), 
                      color=['#FF6347', '#FFD700'], width=0.4)  

# Etiquetas y título del gráfico
plt.xlabel('Año', fontsize=10)  
plt.ylabel('Cantidad', fontsize=10)
plt.title("Composición de PET en 2004 y 2024", fontsize=12)
plt.legend(title='Estado', labels=["En edad", 'No en edad'], fontsize=10)
plt.xticks(rotation=0)  
plt.yticks(fontsize=10)
plt.tight_layout()


# Mostramos el gráfico
plt.show()


#Ejercicio 9
respondieron["desocupados"] = respondieron["estado"].apply(lambda x: 1 if x==2 else 0)

print('desocupados en 2004:', len(respondieron[(respondieron['desocupados'] == 1) & (respondieron['ano4'] == 2004)]))
print('desocupados en 2024:', len(respondieron[(respondieron['desocupados'] == 1) & (respondieron['ano4'] == 2024)]))


# a 
desocupados_estado = respondieron.groupby(['ano4', 'nivel_ed', 'desocupados']).size().unstack()

# Sumamos los valores de ocupados y desocupados por año y nivel educativo
desocupados_estado['Total'] = desocupados_estado[0] + desocupados_estado[1]

# Calculamos la proporción de desocupados
desocupados_estado['Proporcion desocupados'] = (desocupados_estado[1] / desocupados_estado['Total']) * 100

# Limpiamos el DataFrame dejando solo las columnas necesarias
desocupados_estado = desocupados_estado[['Proporcion desocupados']].reset_index()

# Mostramos el resultado
print(desocupados_estado)


#b
#Creamos la variable categórica "CH06" en donde agruparemos la cantidad de perosnas que han cumplido años en subgrupos cuyo rango será de 10 años

respondieron["CH06"] = respondieron["ch06"].apply( lambda x:
    1 if 10 <= x < 20 else
    2 if 20 <= x < 30 else
    3 if 30 <= x < 40 else 
    4 if 40 <= x < 50 else
    5 if 50 <= x < 60 else
    6 if 60 <= x < 70 else 
    7 if 70 <= x < 80 else
    8 if 80 <= x < 90 else
    9 if 90 <= x < 100 else 0
 )

desocupados_edad = respondieron.groupby(['ano4', 'CH06', 'desocupados']).size().unstack()
#Con el fin de medir la proporción de desocupados por edad agrupada comparando 2004 vs 20024, tal como en el ejercicio anterior, generaremos una matriz que almacena la cantidad en cada uno de lso subgrupos de las variables que representan la edad -categorica CH06-, el año y si es desocupado o no (habra 2x2x11 celdas)
#Luego, tal como en el punto anterior, calculamos las cantidades en cada subgrupo para luego poder inferir las proporciones de desocupados por edad agrupada entre ambos años.

# Calcular el total de cada grupo de edad por año
desocupados_edad['Total'] = desocupados_edad[0] + desocupados_edad[1]

# Calcular la proporción de desocupados
desocupados_edad['Proporcion desocupados'] = (desocupados_edad[1] / desocupados_edad['Total']) * 100

# Limpiar el DataFrame para mostrar solo las columnas necesarias
desocupados_edad = desocupados_edad[['Proporcion desocupados']].reset_index()

# Mostrar el resultado
print(desocupados_edad)


#c
#hemos elejido utilizar la variable ch07 que nos indica el estado civil de las personas. Encontramos interesante la relación que puede haber entre el estado civil, la desocupación y la edad, compararemos las diferencias entre estas proporciones entre los años 2004 y 2024
#Para ello, utilizaremos el código descrito en ambos incisos anteriores
# Calcular la proporción de desocupados por estado civil y año
desocupados_civil = (
    respondieron
    .groupby(["ano4", "ch07"])["desocupados"]
    .mean() * 100  # Convertimos a porcentaje
).reset_index()

# Renombramos la columna para mayor claridad
desocupados_civil.rename(columns={"desocupados": "Proporcion desocupados"}, inplace=True)

# Mostramos el resultado
print(desocupados_civil)




#10

df_2004 = respondieron[respondieron['ano4'] == 2004]
df_2024 = respondieron[respondieron['ano4'] == 2024]

# Definimos función para calcular ambas tasas
def calcular_tasas(df):
    desocupados = df[df['desocupados'] == 1].shape[0]
    PEA = df[df['PEA'] == 1].shape[0]  # PEA: Población Económicamente Activa
    PET = df.shape[0]  # PET: Población en Edad de Trabajar (total de la muestra)

    tasa_INDEC = (desocupados / PEA) * 100 if PEA > 0 else 0
    tasa_alternativa = (desocupados / PET) * 100

    return tasa_INDEC, tasa_alternativa

# Calculamos tasas para cada año
tasa_INDEC_2004, tasa_alt_2004 = calcular_tasas(df_2004)
tasa_INDEC_2024, tasa_alt_2024 = calcular_tasas(df_2024)

# Creamos un DataFrame con los resultados
tasas_df = pd.DataFrame({
    'Año': [2004, 2024],
    'Tasa INDEC': [tasa_INDEC_2004, tasa_INDEC_2024],
    'Tasa Alternativa': [tasa_alt_2004, tasa_alt_2024]
})

# Mostramos la tabla con resultados
print(tasas_df)

# Guardamos el resultado en un Excel
tasas_df.to_excel("Output/tasas_desocupacion.xlsx", index=False)








