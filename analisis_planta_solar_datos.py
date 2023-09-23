"""CASO 2 BA: ANALISIS DETECCIÓN INEFICIENCIAS PLANTA SOLAR
En este caso estaremos trabajando para una compañía de generación de energía solar fotovoltaica.

Han detectado comportamientos anómalos en 2 de las plantas y la subcontrata de mantenimiento no es capaz de identificar el motivo.

Antes de desplazar a un equipo de ingenieros nos piden al equipo de data science que analicemos los datos de los sensores y medidores para ver si podemos detectar el problema.

En este caso entre otras cosas vamos a aprender:

- cómo funcionan este tipo de plantas solares
- análisis a realizar en datasets donde la variable tiempo tiene mucha importancia
- cómo enfocar el análisis en proyectos donde los datos son recogidos por sensores o medidores

Por tanto, mucho de lo que aprendamos aquí es de aplicación general en proyectos de industria e IoT:

- análisis de producción en fábricas
- otros tipos de energía
- smart cities
- IoT en agricultura
etc.

Siguiendo la metodología de Discovery:

OBJETIVO

Analizar los datos disponibles para intentar intuir donde pueden estar los problemas y si es necesario desplazar o no a un equipo de ingenieros a las plantas.

PALANCAS

En este tipo de proyectos en el que hay un proceso claro la parte más IMPORTANTE es conocer y entender ese proceso.
Vamos a ver por ejemplo cómo en este caso, que parece a priori fácil por la aparente sencillez de los datos, si no hacemos un diseño del proyecto guiado por el proceso, nos podríamos meter en un bucle infinito de análisis sin llegar a ningún lado.
Una vez entendido cómo funciona el negocio y el proceso las palancas nos van a salir solas.

Fuente del gráfico: https://upload.wikimedia.org/wikipedia/commons/b/bb/How_Solar_Power_Works.png

from IPython import display
display.Image("../../../99_Media/How_Solar_Power_Works.png")

Por tanto las palancas que influyen sobre el objetivo de negocio (en este caso generar corriente AC) son:

- Irradiación: a mayor irradiación mayor DC generada. Pero no es monotónica, a partir de ciertos valores mayor temperatura puede mermar la capacidad de generación
- Estado de los paneles: deben estar limpios y con un correcto funcionamiento para generar la mayor energía DC posible
- Eficiencia de los inverters: siempre hay una pérdida en la transformación de DC a AC, pero debe ser la mínima posible. También deben estar en correcto estado y funcionamiento.
- Medidores y sensores: si se estropean y no miden bien perdemos la trazabilidad y la posibilidad de detectar fallos

KPIs

- Irradiación: mide la energía solar que llega
- Temperatura ambiente y del módulo: medida por los sensores de la planta en grados Celsius
- Potencia DC: medida los kw de corriente contínua
- Potencia AC: medida los kw de corriente alterna
- Eficiencia del inverter (lo crearemos nosotros): mide la capacidad de transformación de DC a AC. Se calcula como AC / DC * 100

ENTIDADES Y DATOS

Para determinar las entidades es necesario conocer de qué se compone una planta solar.

La unidad mínima es la celda, es ahí donde se produce la generación de energía por reacción con los fotones del sol.

Las celdas se encapsulan en unos "rectángulos" que se llaman módulos.

Varios módulos forman un panel.

Los paneles se organizan en filas que se llaman arrays.

Un inverter recibe corriente contínua de varios arrays.

Una planta puede tener varios inverters.

Además están los medidores y los sensores, que puede haber uno o varios.

display.Image("../../../99_Media/paneles_solares.jpeg")

En nuestro caso las entidades que tenemos en la granularidad de los datos son:

- Ventanas de 15 minutos durante un período de 34 días
- Plantas: son 2
- Inverters: varios por planta
- Sólo un sensor de irradiación por planta
- Sólo un sensor de temperatura ambiente por planta
- Sólo un sensor de temperatura del módulo por planta

Esto condiciona que podremos saber por ejemplo si un inverter de una planta tiene menor rendimiento del esperado, pero no sabremos qué array, panel o módulo lo puede estar causando

PREGUNTAS SEMILLA

Habiendo entendido las palancas, kpis y entidades ya podemos plantear las preguntas semilla:

Sobre la irradiación:

- ¿Llega suficiente irradiación todo los días?
- ¿Es similar en ambas plantas?
- ¿Cómo es su distribución por hora?
- ¿Cómo se relaciona con la temperatura ambiente y la temperatura del módulo?

Sobre las plantas:

- ¿Les llega la misma cantidad de irradiación?
- ¿Tienen similar número de inverters?
- ¿Generan similar cantidad de DC?
- ¿Generan similar cantidad de AC?

Sobre la generación de DC:

- ¿Cual es la relación entre irradiación y generación de DC?
- ¿Se ve afectada en algún momento por la temperatura ambiente o del módulo?
- ¿Es similar en ambas plantas?
- ¿Cómo se distribuye a lo largo del día?
- ¿Es constante a lo largo de los días?
- ¿Es constante en todos los inverters?
- ¿Ha habido momentos de fallos?

Sobre la generación de AC:

- ¿Cual es la relación entre DC y generación de AC?
- ¿Es similar en ambas plantas?
- ¿Cómo se distribuye a lo largo del día?
- ¿Es constante a lo largo de los días?
- ¿Es constante en todos los inverters?
- ¿Ha habido momentos de fallos?

Sobre los medidores y sensores:

- ¿Son fiables los datos de irradiación?
- ¿Son fiables los datos de temperatura?
- ¿Son fiables los datos de DC?
- ¿Son fiables los datos de AC?
- ¿Son similares los datos entre ambas plantas?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline # para que los gráficos aparezcan en Jupyter Notebook
# %config IPCompleter.greedy=True # cuando pulsamos la tecla tabuladora que autocomplete

# ENTENDER LOS FICHEROS

# Este caso se compone de 4 ficheros:

# - Planta 1, datos de generación
# - Planta 1, datos de sensor ambiental
# - Planta 2, datos de generación
# - Planta 2, datos de sensor ambiental

# CARGA DE LOS DATOS PLANTA 1 - DATOS DE GENERACIÓN

p1g = pd.read_csv('Datos/Plant_1_Generation_Data.csv')
print(p1g)
p1g.info()

# CARGA DE LOS DATOS PLANTA 1 - DATOS DE SENSOR AMBIENTAL

p1w = pd.read_csv('Datos/Plant_1_Weather_Sensor_Data.csv')
print(p1w)
p1w.info()

# CARGA DE LOS DATOS PLANTA 2 - DATOS DE GENERACIÓN

p2g = pd.read_csv('Datos/Plant_2_Generation_Data.csv')
print(p2g)
p2g.info()

# CARGA DE LOS DATOS PLANTA 2 - DATOS DE SENSOR AMBIENTAL

p2w = pd.read_csv('Datos/Plant_2_Weather_Sensor_Data.csv')
print(p2w)
p2w.info()

# CALIDAD DE DATOS

# CALIDAD DE PLANTA 1 - DATOS DE GENERACIÓN

p1g.info()

# Vemos que no hay nulos.
# Vemos que DATE_TIME está como object.
# Convertimos DATE_TIME a tipo datetime

p1g['DATE_TIME'] = pd.to_datetime(p1g.DATE_TIME,dayfirst=True)

print(p1g.head())

# Comprobamos que el identificador de planta sea único.

print(p1g.PLANT_ID.unique())

# Vamos a reemplazarlo por un literal más legible.

p1g['PLANT_ID'] = p1g.PLANT_ID.replace(4135001, 'p1')

# Revisamos los descriptivos.

print(p1g.describe().T)

# Vamos a quitar la visualización de notación científica.

pd.options.display.float_format = '{:15.2f}'.format
print(p1g.describe().T)

# Resulta extraño la diferencia de medias entre DC y AC. Vamos a visualizarlo.

p1g[['DC_POWER','AC_POWER']].plot(figsize = (16,12));

# La diferencia es muy grande.
# Primero vamos a comprobar si van en la misma dirección aunque sea a disinta escala (con una correlación), y después vamos a comprobar cual es el ratio medio entre ambas medidas.

print(p1g.DC_POWER.corr(p1g.AC_POWER))
print((p1g.DC_POWER / p1g.AC_POWER).describe())

# Parece que los Inverters están transformando solo el 10% de DC a AC, lo cual a priori es muy bajo.
# De todas formas desde la calidad llegamos hasta aquí y seguiremos explorando esto en la parte de análisis y comparándolo con la Planta 2 a ver si pasa lo mismo.
# Analizamos la variable categórica, que es el identificador de los inverters.

print(p1g.SOURCE_KEY.nunique())
print(p1g.SOURCE_KEY.value_counts())

# CONCLUSIONES:

# - La planta 1 tiene 22 inverters
# - Todos tienen un número similar de medidas aunque no exactamente igual
# - Podrían ser paradas por mantenimientos, o simples pérdidas de datos pero lo apuntamos para la fase de análisis

# Vamos a analizar las variables DAILY_YIELD, ya que los metadatos nos dicen que la variable TOTAL_YIELD es el total acumulado por inverter, pero en DAILY_YIELD no lo especifica, por lo que no sabemos si es un acumulado por inverter o por planta.
# La hipótesis es la siguiente: si es por planta no debería haber diferencias entre el dato de los diferentes inverters en el mismo momento puntual.
# Y por consiguiente si vemos que sí hay diferencias entonces es que el dato es por inverter.
# Para comprobarlo nos sirve con coger una muestra de inverters.

seleccion = list(p1g.SOURCE_KEY.unique()[:5])
temp = p1g[p1g.SOURCE_KEY.isin(seleccion)].set_index('DATE_TIME')
print(temp)

# En los datos ya vemos que es diferente, pero vamos a comprobar sobre más datos para que no sea un efecto de esos registros en concreto.
# Vamos a verlo gráficamente, y por simplificar vamos a coger solo una muestra de días.
# Como tenemos la fecha como index recordamos que podemos usar indexación parcial y slice.

temp = temp.loc['2020-06-01':'2020-06-05']
print(temp)

plt.figure(figsize = (16,12))
sns.lineplot(data = temp.reset_index(), x = temp.reset_index().DATE_TIME, y = 'DAILY_YIELD', hue = 'SOURCE_KEY');

# Definitivamente diferentes inverters tienen diferentes datos en el mismo momento temporal, por lo que concluímos que esa variable es por inverter
# Por último vamos a analizar el período en el que tenemos datos y si el número de mediciones diarias es constante.

p1g.DATE_TIME.dt.date.value_counts().sort_index().plot.bar(figsize = (12,8));

# CONCLUSIONES

# - El período de datos es entre el 15 de Mayo del 2020 y el 17 de Junio de 2020
# - Tenemos datos para todos los días, no falta ninguno intermedio
# - Pero algunos días como el 21/05 o el 29/05 tienen menos mediciones
# - Por lo que no parece 100% regular

# CALIDAD DE PLANTA 1 - DATOS DE SENSOR AMBIENTAL

p1w.info()

# Corregimos el tipo de DATE_TIME

p1w.DATE_TIME = pd.to_datetime(p1w.DATE_TIME)
print(p1w.head())

# Reemplazamos el nombre de la planta

p1w['PLANT_ID'] = p1w.PLANT_ID.replace(4135001,'p1')
print(p1w)

# Revisamos los estadísticos

print(p1w.describe().T)

# Revisamos la variable categórica, que es el identificador del sensor.

print(p1w.SOURCE_KEY.nunique())

# Solo hay un sensor de variables ambientales en la planta.
# Revisamos la fecha.

p1w.DATE_TIME.dt.date.value_counts().sort_index().plot.bar(figsize = (12,8));

# CONCLUSIONES

# - El período de datos es entre el 15 de Mayo del 2020 y el 17 de Junio de 2020
# - Tenemos datos para todos los días, no falta ninguno intermedio
# - Pero algunos días como el 21/05 o el 29/05 tienen menos mediciones
# - Por lo que no parece 100% regular

# CALIDAD DE PLANTA 2 - DATOS DE GENERACIÓN

p2g.info()

p2g['DATE_TIME'] = pd.to_datetime(p2g.DATE_TIME)
p2g['PLANT_ID'] = p2g.PLANT_ID.replace(4136001, 'p2')

print(p2g.head())
print(p2g.describe().T)

# En este caso los valores de DC y AC están mucho más cercanos entre sí. Vamos a calcular el ratio.

print((p2g.DC_POWER / p2g.AC_POWER).describe())

# Ahora los valores del ratio sí están muy próximos a uno.
# Analizamos la variable categórica, que es el identificador de los inverters.

print(p2g.SOURCE_KEY.nunique())
print(p2g.SOURCE_KEY.value_counts())

# CONCLUSIONES

# - La planta 2 tiene 22 inverters también
# - Todos tienen un número similar de medidas aunque no exactamente igual
# - A excepción de 4 que tienen unas 800 medidas menos
# - Lo apuntamos para la fase de análisis

# Por último vamos a analizar la fecha.

p2g.DATE_TIME.dt.date.value_counts().sort_index().plot.bar(figsize = (12,8));

# CONCLUSIONES

# - El período de datos es entre el 15 de Mayo del 2020 y el 17 de Junio de 2020
# - Tenemos datos para todos los días, no falta ninguno intermedio
# - Pero algunos días como el 20/05 y varios más tienen menos mediciones
# - Por lo que no parece 100% regular

# CALIDAD DE PLANTA 2 - DATOS DE SENSOR AMBIENTAL

p2w.info()

# Corregimos el tipo de DATE_TIME

p2w.DATE_TIME = pd.to_datetime(p2w.DATE_TIME)
print(p2w.head())

# Reemplazamos el nombre de la planta

p2w['PLANT_ID'] = p2w.PLANT_ID.replace(4136001,'p2')
print(p2w)

# Revisamos los estadísticos

print(p2w.describe().T)

# Analizamos la variable categórica, que es el identificador del sensor.

print(p2w.SOURCE_KEY.nunique())

# Solo hay un sensor de variables ambientales en la planta.
# Revisamos la fecha.

p2w.DATE_TIME.dt.date.value_counts().sort_index().plot.bar(figsize = (12,8));

# CONCLUSIONES

# - El período de datos es entre el 15 de Mayo del 2020 y el 17 de Junio de 2020
# - Tenemos datos para todos los días, no falta ninguno intermedio
# - Pero algunos días como el 15/05 u otros tienen menos mediciones, aunque faltan mucho menos que en los otros datasets
# - Pero no parece 100% regular

# TEMAS PENDIENTES DE LA CALIDAD DE DATOS PARA ANALIZAR POSTERIORMENTE

# - En la planta 1 parece que los Inverters están transformando solo el 10% de DC a AC, lo cual a priori es muy bajo.
# - En la planta 2 el ratio es mucho más cercano a 1.
# - Los intervalos de medida no son 100% regulares. Hay días con menos medidas, y hay también diferencias por inverters.

# CREACIÓN DEL DATAMART ANALITICO

# Vamos a hacer una unión por partes.

# Primero los dos datasets de generación. Que será una apilación de registros ya que los campos son iguales.
# Después los dos de medidas ambientales. Que será una apilación de registros ya que los campos son iguales.
# Y por último cruzaremos ambos parciales mediante la integración por campos clave.

# UNIÓN DE LOS DASTASETS DE GENERACIÓN

gener = pd.concat([p1g,p2g],axis = 'index')
print(gener)

# Vamos a renombrar ya las variables para hacerlas más descriptivas y usables.

gener.columns = ['fecha','planta','inverter_id','kw_dc','kw_ac','kw_dia','kw_total']
print(gener)

# Ahora que tenemos las 2 plantas unidas vamos a hacer lo que se llama un análisis de coherencia, dado que según la documentación kw_dia y kw_total están directamente relacionados con kw_dc y kw_ac.
# Vamos a intentar replicar los datos de kw_dia y kw_total.

gener2 = gener.copy()

# Creamos una variable date para poder agregar por ella.

gener2['date'] = gener2.fecha.dt.date
print(gener2)

# La suma por planta, date e inverter de kw_dc o de kw_ac debería coincidir con el máximo de kw_dia.

gener2 = gener2.groupby(['planta','date','inverter_id']).agg({'kw_dc':sum,
                                                              'kw_ac':sum,
                                                              'kw_dia':max,
                                                              'kw_total':max}).reset_index()
print(gener2)

# Ordenamos para poder analizar.

gener2 = gener2.sort_values(['planta','inverter_id','date'])
print(gener2)

# Kw_dia no concuerda para nada ni con kw_dc ni con kw_ac.
# Vamos a ver si concuerda con kw_total, para ello calculamos el incremento diario de kw_total que debería coincidir con el máximo de kw_dia del día anterior.

gener2['lag1'] = gener2.groupby(['planta','inverter_id']).kw_total.shift(1)
gener2['incremento'] = gener2.kw_total - gener2.lag1
print(gener2)

# Comprobamos en la planta 1.
print(gener2[gener2.planta == 'p1'].head(50))

# Comprobamos en la planta 2.
print(gener2[gener2.planta == 'p2'].head(50))

# CONCLUSIONES

# - kw_dia tiene coherencia con kw_total
# - pero éstas no tienen coherencia con kw_dc ni con kw_ac
# - es como si estuvieran en diferentes unidades o hubiera algún cálculo del que no somos conscientes
# - por tanto tendremos 2 bloques a poder usar: o bien kw_dc con kw_ac, o bien kw_dia con kw_total, pero no podemos mezclarlas entre sí

# UNIÓN DE LOS DATASETS DE MEDICIONES AMBIENTALES

temper = pd.concat([p1w,p2w], axis = 'index')
print(temper)

# Vamos a renombrar ya las variables para hacerlas más descriptivas y usables.

temper.columns = ['fecha','planta','sensor_id','t_ambiente','t_modulo','irradiacion']
print(temper)

# CREACIÓN DEL DATAMART ANALÍTICO

# En este caso el campo clave es compuesto de fecha y planta y manda el dataset de generación, ya que el de temperatura solo nos aporta variables adicionales.

df = pd.merge(left = gener, right = temper, how = 'left', on = ['fecha','planta'])
print(df)

# Tras una integración siempre es conveniente comprobar si se han generado nulos.

print(df.isna().sum())

# Buscamos si los nulos cumplen algún patrón.

nulos = df[df.sensor_id.isna()]
print(nulos)

# Se trata del día 3 de Junio a las 14:00, que por algún motivo no tiene datos de temperatura pero solo para 4 inverters de la planta 1.
#Vamos a buscar en el dataset de temperatura si existe ese datetime.

print(temper[temper.fecha.between('2020-06-03 13:30:00', '2020-06-03 14:30:00')])

# Efectivamente vemos que falta ese tramo en ambas plantas. Pero sin embargo solo hay mediciones en esa hora en la planta 1, y solo en 4 inverters.

# Por tanto habría dos soluciones:

# - imputar esos datos para esos invertes
# - eliminar esos 4 registros

# Dado que parece una franja de medición propia solo de 4 inverters de la planta 1 vamos a optar por eliminarlos.

df.dropna(inplace = True)
print(df)

# Por último vamos a pasar la fecha al index para poder usar toda la potencia de Pandas.

df.set_index('fecha', inplace = True)
print(df)

# GUARDAMOS EL DATAMART

"""
Hasta ahora habíamos usado csv y bases de datos para guardar los datos.
Son formatos muy útiles y sobre todo portables.
Pero tienen el problema de son externos a Python, y por tanto no son capaces de almacenar metadatos propios de Python como algunos tipos de variables (ej datetime).
Por eso se creó un formato propio de Python: el pickle.
Con este formato podemos almacenar cualquier objeto de Python, desde un dataset hasta un modelo de machine learning.
Y tiene la ventaja de que cuando lo recuperas tiene exactamente todas las propiedades del momento en el que lo guardaste.
Además de que está bastante optimizado en cuanto al tamaño en disco.
El inconveniente es que no es tan portable. No puedes abrirlo desde Excel por ejemplo.
Puedes ponerle la extensión que quieras al archivo, aunque se suele usar la convencion .pickle
Para guardar en pickle desde Pandas usamos df.to_pickle('ruta_en_disco')
Y para cargar un pickle usamos pd.read_pickle('ruta_en_disco')
"""

df.to_pickle('Datos/df.pickle')
