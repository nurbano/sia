import argparse
from src.tools import import_json, calcular_atributos_totales, plot_band_error_generation, plot_band_error_aptitud
from src.poblacion import crear_poblacion_inicial
from src.eve import calcular_aptitud, calcular_fitness_generacion, calcular_fitness_relativo
from src.seleccion import seleccionar_padres, realizar_reemplazo, seleccionar_metodo, mostrar_seleccionados, seleccionar_nueva_generacion
from src.crossover import realizar_crossover, generar_hijos
from src.mutacion import aplicar_mutacion
from src.tools import encontrar_mejor_cromosoma, calcular_diversidad_genetica, plot_tasa_convergencia, plot_diversidad_genetica

import time
import sys
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="TP2_AG"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--config_json", default="config.json", help="Path to json config file")

args= parser.parse_args(cmd_args)

data= import_json(args.config_json)

print(f'Puntos totales: {data["total_puntos"]}')
print(f'Población: {data["poblacion"]}')
print(f'Clase:  {data["clase"]}')

K = int(data["porcentaje_hijos"] * data["poblacion"])
atributos = ["fuerza", "destreza", "inteligencia", "vigor", "constitución"]
#Creo población Inicial
poblacion= crear_poblacion_inicial(data["poblacion"],data["total_puntos"])

mejor_aptitud_anterior = None
estancamiento = 0

mejores_aptitudes = []
aptitud_minima_historial = []
aptitud_desviacion_historial = []
diversidad_genetica_historial = []
tasa_convergencia_historial = []
diversidad_por_gen_historial = []
APTITUD_PROMEDIO= []
APTITUD_DESVIACION= []
DIVERSIDAD_DESVIACION= []
atributos_mean= np.zeros((data["max_generaciones"],6))
atributos_std= np.zeros((data["max_generaciones"],6))
# Lista para guardar los atributos del mejor cromosoma en cada generación
mejores_atributos_historial = {atributo: [] for atributo in atributos}

# Lista para almacenar los mejores cromosomas completos con sus aptitudes
mejores_cromosomas_por_generacion = []

# Inicializar el mejor cromosoma de todas las generaciones
mejor_cromosoma_todas_generaciones = None
mejor_aptitud_todas_generaciones = float('-inf')

start = time.time()

for generacion_actual in range(data["max_generaciones"]):
    # Calcular aptitudes actuales
    aptitudes = [calcular_aptitud(data["clase"], cromosoma) for cromosoma in poblacion]
    
    #print(aptitudes)
    mejor_cromosoma, mejor_aptitud_actual = encontrar_mejor_cromosoma(poblacion, calcular_aptitud, data["clase"])
    aptitud_promedio = np.mean(aptitudes)
    aptitud_minima = np.min(aptitudes)
    aptitud_desviacion = np.std(aptitudes)
    APTITUD_PROMEDIO.append(aptitud_promedio)
    APTITUD_DESVIACION.append(aptitud_desviacion)
    # Calcular diversidad genética
    diversidad, diversidad_desviacion= calcular_diversidad_genetica(poblacion, atributos)

    df=pd.DataFrame({'fuerza': [], 'destreza': [], 'inteligencia': [], 'vigor': [], 'constitución': [], 'altura': []})

    for pob in poblacion:
        df_aux= pd.DataFrame([pob])
        df= pd.concat([df, df_aux])
    df= df.reset_index(drop=True)
    for j, atributo in enumerate(df.columns):
        atributos_mean[generacion_actual,j]=df[str(atributo)].mean()
        atributos_std [generacion_actual,j]=df[str(atributo)].std()

    # Tasa de convergencia
    if mejor_aptitud_anterior is not None:
        tasa_convergencia = abs(mejor_aptitud_actual - mejor_aptitud_anterior)
    else:
        tasa_convergencia = 0

    # Guardar los atributos del mejor cromosoma de esta generación
    for atributo in atributos:
        mejores_atributos_historial[atributo].append(mejor_cromosoma[atributo])

    # Guardar el mejor cromosoma con sus atributos y aptitud
    mejores_cromosomas_por_generacion.append({
        'cromosoma': mejor_cromosoma,
        'aptitud': mejor_aptitud_actual
    })

    # Actualizar el mejor cromosoma de todas las generaciones si es necesario
    if mejor_aptitud_actual > mejor_aptitud_todas_generaciones:
        mejor_aptitud_todas_generaciones = mejor_aptitud_actual
        mejor_cromosoma_todas_generaciones = mejor_cromosoma

    # Guardar métricas para análisis posterior
    mejores_aptitudes.append(mejor_aptitud_actual)
    aptitud_minima_historial.append(aptitud_minima)
    aptitud_desviacion_historial.append(aptitud_desviacion)
    diversidad_genetica_historial.append(diversidad)
    DIVERSIDAD_DESVIACION.append(diversidad_desviacion)
    tasa_convergencia_historial.append(tasa_convergencia)

    print(f"Generación {generacion_actual + 1} | Mejor Aptitud: {mejor_aptitud_actual: .4f} | Promedio: {aptitud_promedio:.4f} | Diversidad: {diversidad:.4f} | Tasa Convergencia: {tasa_convergencia:.4f}")

    # Verificar si la aptitud ha dejado de mejorar significativamente
    if mejor_aptitud_anterior is not None and abs(mejor_aptitud_actual - mejor_aptitud_anterior) < data["delta_fitness_min"]:
        estancamiento += 1
    else:
        estancamiento = 0

    # Detener si se ha alcanzado el estancamiento permitido
    if estancamiento >= data["max_estancamiento"]:
        print(f"El algoritmo se ha detenido por estancamiento después de {estancamiento} generaciones sin mejora significativa.")
        break

    # 2. Seleccionar padres
    fitness_rel= np.asarray(calcular_fitness_relativo(poblacion, calcular_aptitud, data["clase"]))
    padres = seleccionar_padres(poblacion, calcular_aptitud, fitness_rel, data["clase"], 
                                data["total_puntos"],K, data["prop_sel_padres"])

    # 3. Generar hijos
    hijos = generar_hijos(padres, K, data["total_puntos"])

    # 4. Seleccionar la nueva generación
    fitness_rel_nueva_gen= fitness_rel= np.asarray(calcular_fitness_relativo(poblacion+hijos, 
                                                                             calcular_aptitud, 
                                                                             data["clase"]))
    nueva_generacion = seleccionar_nueva_generacion(poblacion, hijos,
                                                    calcular_aptitud,
                                                    fitness_rel_nueva_gen,
                                                    data["metodo_seleccion_1"],
                                                    data["metodo_seleccion_2"],
                                                    data["clase"], 
                                                    data["total_puntos"],
                                                    data["poblacion"],
                                                    data["prop_reemplazo"],
                                                    M= data["tam_torneo"],
                                                    T0= data["T0"],
                                                    Tc= data["Tc"],
                                                    t= data["t"],
                                                    threshold= data["th_torneo"])

    # 5. Aplicar mutación
    nueva_generacion_mutada =  aplicar_mutacion(nueva_generacion,  data["prob_mutacion"], data["total_puntos"])

    # Actualizar aptitud y población para la siguiente generación
    mejor_aptitud_anterior = mejor_aptitud_actual
    poblacion = nueva_generacion_mutada
    t_parcial= time.time()-start
    print("Tiempo parcial: ", t_parcial, " segundos")
    if t_parcial > data["tiempo"]:
        print("El algoritmo se detuvo por tiempo.")
        break



end = time.time()
print(f'Tiempo: {end - start} segundos')    
for i, registro in enumerate(mejores_cromosomas_por_generacion):
    registro
    #print(f"Generación {i + 1}: Cromosoma: {registro['cromosoma']}, Aptitud: {registro['aptitud']:.4f}")


print(registro)
# Imprimir el mejor cromosoma de todas las generaciones
print(f"\nMejor cromosoma encontrado en todas las generaciones: {mejor_cromosoma_todas_generaciones}, Aptitud: {mejor_aptitud_todas_generaciones:.4f}")

cant_generaciones= len(mejores_cromosomas_por_generacion)
# Graficar los resultados
x= np.arange(len(APTITUD_PROMEDIO))
y= np.array(APTITUD_PROMEDIO)
y_upper= np.array(APTITUD_PROMEDIO)+np.array(APTITUD_DESVIACION)
y_lower= np.array(APTITUD_PROMEDIO)-np.array(APTITUD_DESVIACION)
y_max= np.array(mejores_aptitudes)
y_min= np.array(aptitud_minima_historial)
plot_band_error_aptitud(x[:cant_generaciones], y[:cant_generaciones], y_upper[:cant_generaciones], y_lower[:cant_generaciones], y_max[:cant_generaciones], y_min[:cant_generaciones])

x= np.arange(data["max_generaciones"])
y= atributos_mean
y_max= atributos_mean+atributos_std
y_min= atributos_mean-atributos_std

plot_band_error_generation(x[:cant_generaciones],y[:cant_generaciones], y_max[:cant_generaciones], y_min[:cant_generaciones] )
plot_tasa_convergencia(tasa_convergencia_historial)


plot_band_error_generation(x[:cant_generaciones],y[:cant_generaciones], y_max[:cant_generaciones], y_min[:cant_generaciones] )
y= np.array(diversidad_genetica_historial)
y_upper= y+np.array(DIVERSIDAD_DESVIACION)
y_lower= y-np.array(DIVERSIDAD_DESVIACION)
plot_diversidad_genetica(y, y_upper, y_lower)
