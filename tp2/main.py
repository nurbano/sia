import argparse
from src.tools import import_json, calcular_atributos_totales, plot_band_error_generation
from src.poblacion import crear_poblacion_inicial
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

genes_mean= np.zeros((20,6))
genes_std= np.zeros((20,6))
fuerza= np.zeros((20,2))
for i in range(20):
    poblacion= crear_poblacion_inicial(data["poblacion"],data["total_puntos"])

    df=pd.DataFrame({'fuerza': [], 'destreza': [], 'inteligencia': [], 'vigor': [], 'constitución': [], 'altura': []})
    #print(poblacion[0]['fuerza'])
    # print(df.head())
    for pob in poblacion:
        df_aux= pd.DataFrame([pob])
        df= pd.concat([df, df_aux])
    df= df.reset_index(drop=True)
    
    for j, atributo in enumerate(df.columns):

        genes_mean[i,j]=df[str(atributo)].mean()
        genes_std [i,j]=df[str(atributo)].std()


# for i in range(6):
#     plt.plot(genes[::,i])
print(df.columns)
# plt.show()
# x= np.arange(20)
# y= genes_mean[:,5]
# y_max= genes_mean[:,5]+genes_std[:,5]
# y_min= genes_mean[:,5]-genes_std[:,5]
x= np.arange(20)
y= genes_mean
y_max= genes_mean+genes_std
y_min= genes_mean-genes_std
print(y_max)
plot_band_error_generation(x,y, y_max, y_min )

