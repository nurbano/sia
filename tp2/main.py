import argparse
from src.tools import import_json, calcular_atributos_totales, plot_band_error_generation
from src.poblacion import crear_poblacion_inicial
from src.eve import calcular_aptitud, calcular_fitness_generacion, calcular_fitness_relativo
from src.seleccion import seleccionar_padres, realizar_reemplazo, seleccionar_metodo, mostrar_seleccionados
from src.crossover import realizar_crossover
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

# genes_mean= np.zeros((20,6))
# genes_std= np.zeros((20,6))
# fuerza= np.zeros((20,2))
# for i in range(20):
#     poblacion= crear_poblacion_inicial(data["poblacion"],data["total_puntos"])

#     df=pd.DataFrame({'fuerza': [], 'destreza': [], 'inteligencia': [], 'vigor': [], 'constitución': [], 'altura': []})
#     #print(poblacion[0]['fuerza'])
#     # print(df.head())
#     for pob in poblacion:
#         df_aux= pd.DataFrame([pob])
#         df= pd.concat([df, df_aux])
#     df= df.reset_index(drop=True)
    
#     for j, atributo in enumerate(df.columns):

#         genes_mean[i,j]=df[str(atributo)].mean()
#         genes_std [i,j]=df[str(atributo)].std()


# # for i in range(6):
# #     plt.plot(genes[::,i])
# print(df.columns)
# # plt.show()
# # x= np.arange(20)
# # y= genes_mean[:,5]
# # y_max= genes_mean[:,5]+genes_std[:,5]
# # y_min= genes_mean[:,5]-genes_std[:,5]
# x= np.arange(20)
# y= genes_mean
# y_max= genes_mean+genes_std
# y_min= genes_mean-genes_std
# print(y_max)
# plot_band_error_generation(x,y, y_max, y_min )
# fit_mean= np.zeros(20)
# for j in range(20):
#     poblacion= crear_poblacion_inicial(data["poblacion"],data["total_puntos"])
#     fit_gene= np.zeros(len(poblacion))
#     for i in range(len(poblacion)):
#         cromosoma= [poblacion[i][pob] for pob in poblacion[0]]
#         fit= fitness(data["clase"], cromosoma)
#         fit_gene[i]= fit


#     print(f'Finess: min: {fit_gene.min(): .4f} - max: {fit_gene.max(): .4f} - mean: {fit_gene.mean(): .4f} - std: {fit_gene.std(): .4f}')
#     fit_mean[j]= fit_gene.mean()
# plt.plot(fit_mean)
# plt.show()
# poblacion= crear_poblacion_inicial(data["poblacion"],data["total_puntos"])
# fit_gene=calcular_fitness_generacion(poblacion,data["clase"])
# print(f'Fitness generación incial: {fit_gene.mean():.3f} (mean) | {fit_gene.std():.3f}(std)')
# for i in range(10):

#     padres = seleccionar_padres(poblacion, fit_gene, data["porc_ruleta"], data["porc_padres"])


#     fit_padres=calcular_fitness_generacion(padres,data["clase"])
#     print(f'Fitness generación seleccionada: {fit_padres.mean():.3f} (mean) | {fit_padres.std():.3f}(std)')
#     poblacion= padres
#     fit_gene= fit_padres


poblacion= crear_poblacion_inicial(data["poblacion"],data["total_puntos"])
fitness=calcular_fitness_generacion(poblacion,data["clase"])

print(f'Fitness generación incial: {fitness.mean():.3f} (mean) | {fitness.std():.3f}(std)')
fitness_rel= np.asarray(calcular_fitness_relativo(poblacion, calcular_aptitud, data["clase"]))
print(f'Fitness Relativo generación incial: {fitness_rel.mean():.3f} (mean) | {fitness_rel.std():.3f}(std)')

seleccionados = seleccionar_metodo(poblacion, 'torneo_probabilistico', calcular_aptitud, 
                                   fitness_rel, data["clase"], K
                                   , threshold=data["th_torneo"])
mostrar_seleccionados("torneo_deterministico", seleccionados, calcular_aptitud, data["clase"])

padres = seleccionar_padres(poblacion, fitness, data["porc_ruleta"], data["porc_padres"])

#print(f'Len padres: {len(padres)}')
fitness_padres =calcular_fitness_generacion(padres,data["clase"])  
print(f'Fitness generación seleccionada: {fitness_padres.mean():.3f} (mean) | {fitness_padres.std():.3f}(std)')

nueva_poblacion= realizar_reemplazo(poblacion, fitness, padres, fitness_padres, data["porc_elite"], data["poblacion"])
fitness_nueva_poblacion =calcular_fitness_generacion(nueva_poblacion,data["clase"])  
print(f'Fitness nueva población: {fitness_nueva_poblacion.mean():.3f} (mean) | {fitness_nueva_poblacion.std():.3f}(std)')
# print("hola")
descendientes = realizar_crossover(padres, data["total_puntos"])
fitness_descendientes = calcular_fitness_generacion(descendientes, data["clase"])
print(f'Fitness descendientes: {fitness_descendientes.mean():.3f} (mean) | {fitness_descendientes.std():.3f}(std)')
