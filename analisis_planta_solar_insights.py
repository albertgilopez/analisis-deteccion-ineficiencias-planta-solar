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

# Formato de graficos
sns.set_style('darkgrid')

# CARGA DE LOS DATOS

df = pd.read_pickle('Datos/df.pickle')

print(df)
df.info()

# ANÁLISIS E INSIGHTS

# La primera palanca es la recepción de la energía solar.

# Tenemos 3 KPIs con los que medir esta palanca: irradiación que llega, temperatura ambiente y temperatura del módulo.
# Estos KPIs se miden con un único sensor por planta, así que el dato es el mismo para todos los inverters.
# Tenemos que entender cómo funcionan estas variables entre sí antes de pasar a ver cómo interactúan con el siguiente nivel.
# Dado que da igual el inverter y solo necesitamos esas 3 variables vamos a crear un dataset más pequeño con solo un inverter de cada planta para trabajar sobre el.

recepcion = df.loc[(df.inverter_id == '1BY6WEcLGh8j5v7') | (df.inverter_id == 'q49J1IKaHRwDQnt'), 'planta':'t_modulo']
print(recepcion)

# PREGUNTA: ¿Las dos plantas reciben la misma cantidad de energía?

temp = recepcion.groupby('planta').agg({'irradiacion':sum,'t_ambiente':np.mean,'t_modulo':np.mean})
print(temp)

f, ax = plt.subplots(nrows=1, ncols=3, figsize = (18,5))

ax[0].bar(temp.index, temp.irradiacion, color = ['red','blue'], alpha = 0.3)
ax[1].bar(temp.index, temp.t_ambiente, color = ['red','blue'], alpha = 0.3)
ax[2].bar(temp.index, temp.t_modulo, color = ['red','blue'], alpha = 0.3)
ax[0].set_title('Irradiación por planta')
ax[1].set_title('Temperatura ambiente por planta')
ax[2].set_title('Temperatura módulo por planta');

# CONCLUSIONES:

# - En general la planta 2 recibe más energía solar que la 1
# - Pero esta diferencia no puede implicar el problema de rendimiento que supuestamente existe

# PREGUNTA: ¿Cómo se relacionan esas tres variables?

temp = recepcion.loc[:,['planta','irradiacion','t_ambiente','t_modulo']]
print(temp)

temp = temp.drop(columns=["planta"])
sns.heatmap(temp.corr(), annot=True);

temp = recepcion.loc[:,['planta','irradiacion','t_ambiente','t_modulo']]
sns.pairplot(temp.reset_index(), hue = 'planta', height=3, plot_kws={'alpha': 0.1});

# CONCLUSIONES:

# - La irradiación correlaciona mucho con la temperatura del módulo
# - Pero no tanto con la temperatura ambiente
# - Por tanto una primera forma de identificar módulos defectuosos o sucios es localizar los que produzcan poco cuando la irradiación es alta

# PREGUNTA: ¿Cómo se distribuye la irradiación y la temperatura a lo largo del día?

temp = pd.crosstab(recepcion.hora,recepcion.planta,values = recepcion.irradiacion,aggfunc='mean')
print(temp)

plt.figure(figsize=(10,10))
sns.heatmap(temp, annot=True, fmt=".2f");

temp = pd.crosstab(recepcion.hora,recepcion.planta,values = recepcion.t_ambiente,aggfunc='mean')
print(temp)

plt.figure(figsize=(10,10))
sns.heatmap(temp, annot=True, fmt=".1f");

# CONCLUSIONES:

# - Ambas plantas tienen patrones similares. Podríamos pensar que están en zonas geográficas no muy alejadas
# - Existe irradiación (y por tanto a priori las plantas deberían producir) entre las 7 y las 17
# - La irradiación máxima se produce entre las 11 y las 12
# - La temperatura ambiente máxima se produce entre las 14 y las 16

# PREGUNTA: ¿Ambas plantas son igual de capaces de generar DC a partir de la irradiación?

plt.figure(figsize = (12,8))
sns.scatterplot(data = df, x = df.irradiacion, y = df.kw_dc);

# Existen 2 patrones claramente diferentes. ¿Serán las plantas?

plt.figure(figsize = (12,8))
sns.scatterplot(data = df, x = df.irradiacion, y = df.kw_dc, hue = 'planta');

# La planta número 2 produce muchos menos kw ante los mismos niveles de irradiación.
# Pero antes habíamos visto que la relación entre dc y ac en la planta 1 era rara.
# Y también que los datos de dc y ac no cuadraban con los de kw_dia.

# Hay algo raro en los datos.

# Vamos a ver la relación entre la irradiación y kw_dia a ver si nos da luz.

plt.figure(figsize = (12,10))
sns.scatterplot(data = df, x = df.irradiacion, y = df.kw_dia, hue = 'planta');

# Es muy extraño. Parece que la relación es que a más irradiación menos kw generados. Lo cual no tiene sentido.
# Incluso parece que los máximos de kw se producen en horas de irradiación cero.

# ¿Te imaginas qué puede estar pasando?

# CUIDADO: la variable kw_dia es un ACUMULADO. Eso significa que debería alcanzar su máximo cuando llega la última hora del día, por ej las 23:45, donde obviamente la irradiación es cero.
# Y no tener datos hasta pasadas las 7 que es cuando vemos que hay irradiación.

# Vamos a comprobarlo.

df.groupby('hora')[['kw_dia']].mean().plot.bar();

# De nuevo algo no cuadra. Hay generación entre las 00 y las 06.
# Y además a partir de las 18 comienza a decaer, lo cual no debería pasar si es un acumulado.

# CONCLUSIONES:

# No nos fiamos de estas variables acumuladas como kw_dia y kw_total.
# Pero la verdad es que tampoco nos fiamos mucho de las otras.
# En una situación real yo pararía el proyecto hasta ser capaz de ver qué pasa con los datos.
# Pero para poder continuar vamos a asumir que los datos de dc y ac son correctos.
# Y bajo esa asunción obtendremos nuestras conclusiones.

# INSIGHT #1: La planta 2 genera niveles mucho más bajos de DC incluso a niveles similares de irradiación

# PREGUNTA: ¿La generación es constante a lo largo de los días?

# Podemos usar el df_dia para graficar la visión global de generación de DC durante el período de análisis.

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
print(df.info())

plt.figure(figsize = (10,8))

# Esto devuelve los niveles en tuplas, que luego podemos unir con un list comprehension.
# Vamos a revisar como devuelve los nombres de columnas to_flat_index()

tuplas = df_dia.columns.to_flat_index()
# Y unimos ambas partes del par con un guión bajo usando .join

df_dia.columns = ["_".join(par) for par in tuplas]

print(df_dia.info())

sns.lineplot(data = df_dia.reset_index(), x = df_dia.reset_index().fecha, y = 'kw_dc_sum', hue = 'planta');

# Vemos que la planta 1 tiene mucha más variabilidad mientras que la planta 2 es mucho más constante.
# Pero sobre todo nos extraña los bajos niveles de generacion de DC en de la planta 2 en comparación con la 1.

# Vamos a examinar la generación de cada día a ver si vemos algo raro.
# Generamos una variable date para poder agregar por ella.

df['date'] = df.index.date
print(df)

# Creamos un dataframe temporal para analizar la generación de DC horaria en cada día en la planta 1.

dc_constante_p1 = df[df.planta == 'p1'].groupby(['planta','date','hora']).kw_dc.sum()
print(dc_constante_p1)

# Vamos a pasar date a columnas, para poder respresentar cada columna (que son los dates) como una variable y por tanto como un gráfico independiente.

dc_constante_p1.unstack(level = 1).plot(subplots = True, layout = (17,2), sharex=True, figsize=(20,30));

# CONCLUSIONES:

# - En la planta 1 sí se mantienen unos patrones similares durante todos los días
# - A excepción de un parón el día 20 de Mayo y una caída extraña el 05 de Junio
# - Pero ninguna parece ser estructural
# - Por tanto aunque cada día pudiera tener diferentes totales de producción los patrones intradía son similares y parecen correctos

# Repetimos el análisis en la planta 2

dc_constante_p2 = df[df.planta == 'p2'].groupby(['planta','date','hora']).kw_dc.sum()
print(dc_constante_p2)

# Vamos a pasar date a columnas, para poder respresentar cada columna (que son los dates) como una variable y por tanto como un gráfico independiente.

dc_constante_p2.unstack(level = 1).plot(subplots = True, layout = (17,2), sharex=True, figsize=(20,30));

# CONCLUSIONES:

# - De nuevo el día 20 de Mayo aparece con un comportamiento raro
# Los niveles de producción son constantes durante los días, pero siempre unas 10 veces por debajo de los nivels de la planta 1

# INSIGHT #2: Los niveles bajos de la planta 2 son constantes y presentan unas curvas diarias que parecen normales.

# PREGUNTA: ¿La conversión de DC a AC se genera correctamente?

sns.scatterplot(data = df, x = df.kw_dc, y = df.kw_ac, hue = df.planta);

# De nuevo los patrones son clarísimos: la planta 2 transforma la corriente de forma mucho más eficiente.
# Vamos a ampliar analizando la variable eficiencia que habíamos creado.

temp = df.groupby(['planta','hora'],as_index = False).eficiencia.mean()
print(temp)

sns.lineplot(data = temp, x = 'hora', y = 'eficiencia', hue = 'planta');

# INSIGHT #3

# La planta 1 tiene una capacidad de transformar DC a AC bajísima, lo cual sugiere problemas con los inverters

# Otras conclusiones:

# - Entrar en el detalle de los inverters de la planta 1, a ver si son todos o hay algunos que sesgan la media
# - Revisar por qué la planta 2 pierde eficiencia durante las horas de más irradiación

# Vamos a empezar por la segunda, comparando la producción de DC con la de AC en la planta 2.

temp = df[['planta','hora','kw_dc','kw_ac']].melt(id_vars= ['planta','hora'])
print(temp)

plt.figure(figsize = (12,8))
sns.lineplot(data = temp[temp.planta == 'p2'], x = 'hora', y = 'value', hue = 'variable', ci = False);

# Vemos que efectivamente en las horas centrales hay pérdida de eficiencia. Pero ni de lejos el nivel de pérdida que habíamos visto en el análisis anterior.
# Vamos a analizar la distribución de la eficiencia en esas horas.

temp = df.between_time('08:00:00','15:00:00')
temp = temp[temp.planta == 'p2']

temp.eficiencia.plot.density();

# Hay un conjunto de datos con eficiencia cero, y es lo que genera el problema. ¿Pero cual es la causa de esa eficiencia cero?

# Vamos a seleccionar esos casos y revisarlos.

temp[temp.kw_dc == 0]

# Parece que no es problema del inverter, si no de que en esos momentos no se ha generado DC.
# Vamos a poner la condición de que DC > 0 y ver ahí cual es la eficiencia.

temp[temp.kw_dc > 0].eficiencia.plot.density();

# Efectivamente cuando hay DC la eficiencia es superior al 96%.
# La pregunta entonces es ¿por qué no hay DC? ¿Hay algún patrón?

# Vamos a crear un indicador de DC = 0 para poder analizarlo.

temp['kw_dc_cero'] = np.where(temp['kw_dc'] == 0, 1, 0)
print(temp)

# Empezamos por las variables numéricas.

temp.groupby('kw_dc_cero')[['irradiacion','t_ambiente','t_modulo']].mean()

# En la temperatura ambiente no hay mucha diferencia, pero en la del módulo y en la irradiación sí.
# ¿Podría ser que si se calienta demasiado el módulo deje de generar DC?

# Vamos a verlo comparando la temperatura del módulo con la generación de DC.

sns.scatterplot(data = temp, x = 't_modulo', y = 'kw_dc',hue = 'kw_dc_cero');

# La hipótesis anterior no se confirma, ya que hay muchos casos de temperaturas altas donde se genera DC, y también de kw_dc igual a cero en casi todos los rangos de temperaturas.

# Vamos a analizar ahora las categóricas, empezando por el inverter.

temp.groupby('inverter_id').kw_dc_cero.mean().sort_values(ascending = False).plot.bar();

# Existe gran diferencia en el porcentaje de producción cero de DC por inverter.
# Desde algunos que tienen menos del 5% hasta algunos que superan el 30%.

# INSIGHT #4:: En la planta 2 existen varios inverters a los que no está llegando suficiente producción de DC, y por tanto cuyos módulos necesitan revisión.

# Vamos a analizar los inverters desde el punto de vista de la eficiencia media para ver si hay "buenos y malos".

temp[temp.kw_dc > 0].groupby(['inverter_id','date'],as_index = False).eficiencia.mean().boxplot(column = 'eficiencia', by = 'inverter_id', figsize = (14,10))
plt.xticks(rotation = 90);

# INSIGHT #5:: Una vez descontando el problema de la no generación de DC, los inverters de la planta 2 sí funcionan bien y hacen bien el trabajo de transformación a AC.

# Para terminar de analizar la eficiencia de los inverters podemos ver su rendimiento en cada uno de los días para ver si han posido existir problemas puntuales

temp[temp.kw_dc > 0].groupby(['inverter_id','date']).eficiencia.mean().unstack(level = 0).plot(subplots = True, sharex=True, figsize=(20,40))
plt.xticks(rotation = 90);

# Para tener un término de comparación vamos a repetir los análisis con la planta 1.

temp = df.between_time('08:00:00','15:00:00')
temp = temp[temp.planta == 'p1']
temp['kw_dc_cero'] = np.where(temp['kw_dc'] == 0, 1, 0)
print(temp)

temp.eficiencia.plot.density();

# Vemos que no, aquí todos los inverters tienen una eficiencia constante (aunque muy baja)

temp.groupby(['inverter_id','date'],as_index = False).eficiencia.mean().boxplot(column = 'eficiencia', by = 'inverter_id', figsize = (14,10))
plt.xticks(rotation = 90);

# Vemos que salvo días puntuales en algunos inverters en el resto la eficiencia es constante.
# Vamos a revisar la eficiencia media diaria por cada inverter.

temp.groupby(['inverter_id','date']).eficiencia.mean().unstack(level = 0).plot(subplots = True, sharex=True, figsize=(20,40))
plt.xticks(rotation = 90);

# En el análisis por inverter vemos de nuevo que todos los datos son constantes.
# Vamos a comprobar que entonces no hay fallos en la generación de DC.

temp.groupby('inverter_id').kw_dc_cero.mean().sort_values(ascending = False).plot.bar();

# Vemos que aunque hay algunos inverters que han tenido fallos su magnitud es inferior al 2% de las mediciones.
# Por tanto la generación de DC en la planta 1 sí es correcta, y el fallo está en la transformación de DC a AC.

# CONCLUSIONES:

"""
Tras un ananálisis de los datos podemos concluir que:

- Existen graves problemas de calidad de datos. Se debería revisar en qué parte de la cadena se generan estos problemas, incluyendo los medidores de las plantas.
- El hecho de que la generación en DC sea unas 10 veces superior en la planta 1 que en la 2, sumado al hecho de que la eficiencia en la planta 1 esté sobre el 10% nos lleva a pensar que el dato de generación de DC en la planta 1 puede estar artificialmente escalado por algún motivo.
- Pero de momento a falta de comprobación vamos a asumir que los datos son correctos.
- La dos plantas han recibido altas cantidades de irradiación, no hemos localizado ningún problema en esta fase
- Aunque la temperatura ambiente es superior en la planta 2 y sus módulos se calientan más que los de la planta 1 esto no parece tener un impacto significativo
- La generación de DC de la planta 1 funciona bien, los módulos parecen llevar DC a los inverters.
- La generación de DC de la planta 2 NO funciona bien, algunos módulos llevan muy poco DC a los inverters incluso en las horas de mayor irradiación.
- La transformación de DC a AC de la planta 1 NO funciona bien, solo se transforma en torno al 10%, eso sí, de forma constante. Y esta baja eficiencia no es debida a momentos de no recepción de DC ni se concentra en inverters concretos, si no que parece más estructural (de nuevo tener en cuenta que podría deberse a un problema de calidad de datos en kw_dc de la planta 1
- La transformación de DC a AC de la planta 2 funciona bien, ya que una vez eliminados los períodos de generación cero de DC el resto tienen una eficiencia superior al 97%

RECOMENDACIONES:

- Revisar la captación de datos y su fiabilidad
- Revisión de mantenimiento en los módulos de los inverters de la planta 2 en los que hay muchos momentos de generación cero de DC
- Revisión de mantenimiento de los inverters de la Planta 1
"""

