from src.tools import import_json, cross_validation
from src.tools import cargar_datos, normalizar
from src.layers import Perceptron

import argparse
import time
import sys
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="TP3 Perceptrón"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--config_json", default="config.json", help="Path to json config file")

args= parser.parse_args(cmd_args)

data= import_json(args.config_json)

name_config= args.config_json.split(".json")[0].split("./config/")[1] #En windows usar .\\config\\ en Linux ./config/


print(f'Tamaño de entrada: {data["input_size"]} | Épocas: {data["epochs"]} | Tasa de Aprendizaje: {data["learning_rate"]} | Activación: {data["activation_function"]}')

if data["file"]=="None":
    x= data["x"]
    y= data["y"]

else:
    datos = cargar_datos(data["file"])
    x = [fila[:-1] for fila in datos]  # Las tres primeras columnas son las entradas
    y = [fila[-1] for fila in datos]   # La última columna es la salida
    



x= np.asarray(x)
y= np.asarray(y)
if data["normalizar"]=="True":
        #x= normalizar(x)
        y= normalizar(y)

# print(f'x: {x}')
# print(f'y: {y}')


for i in range(data["iteraciones"]):
     # Inicializamos el perceptrón con función escalón
     perceptron= Perceptron(input_size=data["input_size"],
                                learning_rate=data["learning_rate"], 
                                epochs=data["epochs"], 
                                activation=data["activation_function"])
     errors, df_aux,  weight_history= perceptron.train(x, y)
     df_aux["Iteración"]= i
     if i==0:
          df= df_aux
     else:
        df= pd.concat([df, df_aux]).reset_index(drop=True)
    


df.to_excel(f'./results/{name_config}.xlsx', index=False)
print(df)
plt.plot(errors, marker='o', color='red')
plt.title(f'Evolución del Error para la config {name_config}')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig(f'./images/plot_error_{name_config}.png')
plt.show()

kf= KFold(n_splits=5)
MSE_TRAIN=[]
MSE_TEST=[]
for i, (train_index, test_index) in enumerate(kf.split(x)):

    print(f"Fold {i}:")

    #print(f"  Train: index={train_index}")
    errors, df_aux,  weight_history= perceptron.train(x[train_index], y[train_index],verbose="False")
    
    print("MSE train", errors[-1])
    MSE_TRAIN.append(errors[-1])
    #print(f"  Test:  index={test_index}")
    outputs= perceptron.predict(x[test_index])
    error = y[test_index] - outputs  #Salida esperada menos la salida obtenida
    
    total_error =np.sum( error ** 2)
    mse = total_error / len(x) #El error depende de todos
    MSE_TEST.append(mse)
    print("MSE test", mse)

print(MSE_TRAIN, MSE_TEST)
print(f"MSE Train Promedio: {sum(MSE_TRAIN)/len(MSE_TRAIN)}")
print(f"MSE Test Promedio: {sum(MSE_TEST)/len(MSE_TEST)}")