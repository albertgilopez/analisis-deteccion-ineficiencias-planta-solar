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

# Para que no nos muestre la notación científica

pd.options.display.float_format = '{:15.2f}'.format

# CARGA DE LOS DATOS

df = pd.read_pickle('Datos/df.pickle')

print(df)
df.info()

# CREACIÓN DE VARIABLES

# Comenzamos por extraer los componentes de la fecha e incorporarlos como nuevas variables.

def componentes_fecha(dataframe):

    mes = dataframe.index.month
    dia = dataframe.index.day
    hora = dataframe.index.hour
    minuto = dataframe.index.minute
    
    return(pd.DataFrame({'mes':mes, 'dia':dia, 'hora':hora, 'minuto':minuto}))

df = pd.concat([df.reset_index(),componentes_fecha(df)], axis = 1).set_index('fecha')
print(df)

# Vamos a crear la variable eficiencia del inverter, que consiste en el porcentaje de DC que transforma a AC satisfactoriamente.
# Pero se nos presenta una dificultad muy habitual en los ratios, que el denominador puede ser cero.
# Si fuera el caso, al hacer el ratio nos devolvería un nulo.
# En nuestro caso el denominador es DC, por tanto si la generación de DC fuera cero la de AC debería ser cero también.
# Podemos corregir eso simplemente imputando los nulos que salgan por ceros.

def eficiencia_inverter(AC,DC):
    temp = AC / DC * 100
    return(temp.fillna(0))

df['eficiencia'] = eficiencia_inverter(df.kw_ac, df.kw_dc)

# Comprobamos que no haya generado nulos.

print(df.eficiencia.isna().sum())

# Visualizamos la eficiencia a nivel global.

df.eficiencia.plot.kde();

# Aquí hay algo importante.

# Hay dos grupos claramente diferenciados y uno de ellos es claramente ineficiente.
# Pero de momento lo dejamos apuntado y más adelante revisaremos que entidad es la que está teniendo problemas: planta, inverter, etc.

# REORDENACION DEL DATAFRAME

# En este caso es muy importante no empezar a analizar por analizar, si no seguir el plan definido en el diseño del proyecto, ya que existe un orden muy claro en el proceso: factores ambientales --> kw_dc --> kw ac.
# Así que vamos a reorganizar las columnas del df para que nos ayude a interpretar en este orden.

orden = ['planta','mes','dia','hora','minuto','sensor_id','irradiacion','t_ambiente','t_modulo','inverter_id','kw_dc','kw_ac','eficiencia','kw_dia','kw_total']

df = df[orden]
print(df)

# DATAFRAME DIARIO

# En nivel de análisis al que tenemos los datos es cada 15 minutos, lo cual puede ser demasiado desagregado para ciertos análisis.
# Vamos a dejar construída una versión del dataframe agregada a nivel dia. Para ello usamos resample para hacer downgrading.
# Deberemos agregar por planta e inverter que son los campos clave de nuestro dataset.
# Como tenemos variables a las que aplican diferentes funciones de agregación podemos usar el formato de diccionario de agg()

print(df.head())

df_dia = df.groupby(['planta', 'inverter_id']).resample('D') \
    .agg({'irradiacion': [min,np.mean,max],
          't_ambiente': [min,np.mean,max],
          't_modulo': [min,np.mean,max],
          'kw_dc': [min,np.mean,max,sum],
          'kw_ac': [min,np.mean,max,sum],
          'eficiencia': [min,np.mean,max],
          'kw_dia': max,
          'kw_total': max})

print(df_dia)

# Nos lo ha generado con multi índice, tanto en filas como en columnas.
# Para quitar el de las columnas podemos aplanar los nombres con .to_flat_index(), que es un método relativamente reciente de Pandas.

# https://pandas.pydata.org/docs/reference/api/pandas.MultiIndex.to_flat_index.html

# Esto devuelve los niveles en tuplas, que luego podemos unir con un list comprehension.
# Vamos a revisar como devuelve los nombres de columnas to_flat_index()

tuplas = df_dia.columns.to_flat_index()
print(tuplas)

# Y unimos ambas partes del par con un guión bajo usando .join

df_dia.columns = ["_".join(par) for par in tuplas]
print(df_dia)

# Ahora tenemos que pasar planta e inverter_id a columnas, y dejar la fecha como el índice.

df_dia = df_dia.reset_index().set_index('fecha')
print(df_dia)

# Ya tenemos preparados nuestros datasets por hora y por día. Los guardamos.

df.to_pickle('Datos/df.pickle')
df_dia.to_pickle('Datos/df_dia.pickle')
