import argparse
from src.tools import import_json, calcular_atributos_totales, plot_band_error_generation
from src.poblacion import crear_poblacion_inicial
from src.eve import calcular_aptitud, calcular_fitness_generacion, calcular_fitness_relativo
from src.seleccion import seleccionar_padres, realizar_reemplazo, seleccionar_metodo, mostrar_seleccionados, seleccionar_nueva_generacion
from src.crossover import realizar_crossover, generar_hijos
from src.mutacion import aplicar_mutacion
from src.tools import encontrar_mejor_cromosoma, calcular_diversidad_genetica

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

# Lista para guardar los atributos del mejor cromosoma en cada generación
mejores_atributos_historial = {atributo: [] for atributo in atributos}

# Lista para almacenar los mejores cromosomas completos con sus aptitudes
mejores_cromosomas_por_generacion = []

# Inicializar el mejor cromosoma de todas las generaciones
mejor_cromosoma_todas_generaciones = None
mejor_aptitud_todas_generaciones = float('-inf')

for generacion_actual in range(data["max_generaciones"]):
    # Calcular aptitudes actuales
    aptitudes = [calcular_aptitud(data["clase"], cromosoma) for cromosoma in poblacion]
    mejor_cromosoma, mejor_aptitud_actual = encontrar_mejor_cromosoma(poblacion, calcular_aptitud, data["clase"])
    aptitud_promedio = np.mean(aptitudes)
    aptitud_minima = np.min(aptitudes)
    aptitud_desviacion = np.std(aptitudes)

    # Calcular diversidad genética
    diversidad = calcular_diversidad_genetica(poblacion, atributos)

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
    tasa_convergencia_historial.append(tasa_convergencia)

    print(f"Generación {generacion_actual + 1} | Mejor Aptitud: {mejor_aptitud_actual} | Promedio: {aptitud_promedio:.4f} | Diversidad: {diversidad:.4f} | Tasa Convergencia: {tasa_convergencia:.4f}")

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
                                                 fitness_rel_nueva_gen,                                                   data["clase"], 
                                                   data["total_puntos"],
                                                   data["poblacion"],
                                                     data["prop_reemplazo"],
                                                       data["tam_torneo"])

    # 5. Aplicar mutación
    nueva_generacion_mutada =  aplicar_mutacion(nueva_generacion,  data["prob_mutacion"], data["poblacion"])

    # Actualizar aptitud y población para la siguiente generación
    mejor_aptitud_anterior = mejor_aptitud_actual
    poblacion = nueva_generacion_mutada
    
for i, registro in enumerate(mejores_cromosomas_por_generacion):
    print(f"Generación {i + 1}: Cromosoma: {registro['cromosoma']}, Aptitud: {registro['aptitud']:.4f}")

# Imprimir el mejor cromosoma de todas las generaciones
print(f"\nMejor cromosoma encontrado en todas las generaciones: {mejor_cromosoma_todas_generaciones}, Aptitud: {mejor_aptitud_todas_generaciones:.4f}")

# Graficar los resultados

# Gráficas existentes
plt.figure(figsize=(10, 5))
plt.plot(mejores_aptitudes, label="Mejor Aptitud")
plt.title("Mejor Aptitud a lo largo de las Generaciones")
plt.xlabel("Generación")
plt.ylabel("Mejor Aptitud")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(aptitud_minima_historial, label="Aptitud Mínima")
plt.title("Aptitud Mínima a lo largo de las Generaciones")
plt.xlabel("Generación")
plt.ylabel("Aptitud Mínima")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(aptitud_desviacion_historial, label="Desviación Estándar de Aptitud")
plt.title("Desviación Estándar de la Aptitud a lo largo de las Generaciones")
plt.xlabel("Generación")
plt.ylabel("Desviación Estándar")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(tasa_convergencia_historial, label="Tasa de Convergencia")
plt.title("Tasa de Convergencia a lo largo de las Generaciones")
plt.xlabel("Generación")
plt.ylabel("Tasa de Convergencia")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(diversidad_genetica_historial, label="Diversidad Genética")
plt.title("Diversidad Genética a lo largo de las Generaciones")
plt.xlabel("Generación")
plt.ylabel("Diversidad Genética")
plt.legend()
plt.show()

# Nuevas gráficas de los atributos del mejor cromosoma a lo largo de las generaciones
for atributo in atributos:
    plt.figure(figsize=(10, 5))
    plt.plot(mejores_atributos_historial[atributo], label=f"{atributo.capitalize()}")
    plt.title(f"Evolución de {atributo.capitalize()} del Mejor Cromosoma a lo largo de las Generaciones")
    plt.xlabel("Generación")
    plt.ylabel(f"{atributo.capitalize()}")
    plt.legend()
    plt.show()

# fitness=calcular_fitness_generacion(poblacion,data["clase"])
# fitness_rel= np.asarray(calcular_fitness_relativo(poblacion, calcular_aptitud, data["clase"]))

# padres = seleccionar_padres(poblacion, calcular_aptitud, fitness_rel,
#                              data["clase"], data["total_puntos"], K, data["prop_sel_padres"])
# mostrar_seleccionados("Padres Seleccionados", padres, calcular_aptitud, data["clase"])

# hijos = generar_hijos(padres, K, data["total_puntos"])
# mostrar_seleccionados("Hijos Generados", hijos, calcular_aptitud,  data["clase"])
# fitness_rel_nueva_gen= fitness_rel= np.asarray(calcular_fitness_relativo(poblacion+hijos, calcular_aptitud, data["clase"]))
# print(fitness_rel_nueva_gen)
# nueva_generacion = seleccionar_nueva_generacion(poblacion, hijos,
#                                                  calcular_aptitud,
#                                                  fitness_rel_nueva_gen,                                                   data["clase"], 
#                                                    data["total_puntos"],
#                                                    data["poblacion"],
#                                                      data["prop_reemplazo"],
#                                                        data["tam_torneo"])
# mostrar_seleccionados("Nueva Generación", nueva_generacion, calcular_aptitud, data["clase"])

# nueva_generacion_mutada = aplicar_mutacion(nueva_generacion,  data["prob_mutacion"], data["poblacion"])
# mostrar_seleccionados("Nueva Generación Mutada", nueva_generacion_mutada, calcular_aptitud, data["clase"])

# print(f'Fitness generación incial: {fitness.mean():.3f} (mean) | {fitness.std():.3f}(std)')
# print(f'Fitness Relativo generación incial: {fitness_rel.mean():.3f} (mean) | {fitness_rel.std():.3f}(std)')

# seleccionados = seleccionar_metodo(poblacion, 'torneo_probabilistico', calcular_aptitud, 
#                                    fitness_rel, data["clase"], K
#                                    , threshold=data["th_torneo"])
# mostrar_seleccionados("torneo_deterministico", seleccionados, calcular_aptitud, data["clase"])

# padres = seleccionar_padres(poblacion, fitness, data["porc_ruleta"], data["porc_padres"])

# #print(f'Len padres: {len(padres)}')
# fitness_padres =calcular_fitness_generacion(padres,data["clase"])  
# print(f'Fitness generación seleccionada: {fitness_padres.mean():.3f} (mean) | {fitness_padres.std():.3f}(std)')

# nueva_poblacion= realizar_reemplazo(poblacion, fitness, padres, fitness_padres, data["porc_elite"], data["poblacion"])
# fitness_nueva_poblacion =calcular_fitness_generacion(nueva_poblacion,data["clase"])  
# print(f'Fitness nueva población: {fitness_nueva_poblacion.mean():.3f} (mean) | {fitness_nueva_poblacion.std():.3f}(std)')
# # print("hola")
# descendientes = realizar_crossover(padres, data["total_puntos"])
# fitness_descendientes = calcular_fitness_generacion(descendientes, data["clase"])
# print(f'Fitness descendientes: {fitness_descendientes.mean():.3f} (mean) | {fitness_descendientes.std():.3f}(std)')
