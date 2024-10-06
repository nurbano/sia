from src.tools import import_json, cross_validation
from src.tools import cargar_datos, normalizar, cargar_datos_digitos, cargar_datos_digitos_ruido
from src.layers import Perceptron, MLPParidad

import argparse
import time
import sys
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="TP3 Perceptrón"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--config_json", default="config.json", help="Path to json config file")

args= parser.parse_args(cmd_args)

data= import_json(args.config_json)
print(args.config_json)
name_config= args.config_json.split(".json")[0].split("./config/")[1] #En windows usar .\\config\\ en Linux ./config/


print(f'Tamaño de entrada: {data["input_size"]} | Épocas: {data["epochs"]} | Tasa de Aprendizaje: {data["learning_rate"]} | Activación: {data["activation_function"]}')

if data["file"]=="None":
    x= data["x"]
    y= data["y"]

else:
    datos = cargar_datos_digitos_ruido(data["file"])
    datos= np.array(datos)
    x = [fila[:-1] for fila in datos]  # Las tres primeras columnas son las entradas
    y = [fila[-1] for fila in datos]   # La última columna es la salida
    print(datos)
    print(f"Forma de los datos cargados: {datos.shape}")  # Debería ser (10, 35)

# Preparar etiquetas para la paridad (0 si es impar, 1 si es par)
y_paridad = np.array([1 if i % 2 == 0 else 0 for i in range(10)])  # 0: impar, 1: par

# Preparar etiquetas para la clasificación de dígitos (codificación one-hot)
y_digitos = np.eye(10)

x= np.asarray(x)
y= np.asarray(y)
if data["normalizar"]=="True":
        x= normalizar(x)
        y= normalizar(y)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))  # Crear una figura con un subplot de 2 filas y 5 columnas
for i, ax in enumerate(axes.flat):
    ax.imshow(np.reshape(datos[i], (7, 5)), cmap='Grays')
    ax.set_title(f"Dígito: {i}")
    ax.axis('off')  # Ocultar los ejes
plt.tight_layout()
plt.show()
# print(f'x: {x}')
# print(f'y: {y}')


# for i in range(data["iteraciones"]):
#      # Inicializamos el perceptrón con función escalón
#      perceptron= Perceptron(input_size=data["input_size"],
#                                 learning_rate=data["learning_rate"], 
#                                 epochs=data["epochs"], 
#                                 activation=data["activation_function"])
#      errors, df_aux= perceptron.train(x, y)
#      df_aux["Iteración"]= i
#      if i==0:
#           df= df_aux
#      else:
#         df= pd.concat([df, df_aux]).reset_index(drop=True)
    
mlp_paridad = MLPParidad(
    input_size=data["input_size"],
    hidden_size=data["hidden_size"],
    learning_rate=data["learning_rate"],
    epochs=data["epochs"],
    weight_update_method=data["weight_update_method"]  # Opciones: 'GD', 'Momentum', 'Adam'
)

# Entrenar la red neuronal con los datos limpios
mlp_paridad.train(X=datos, y=y_paridad)

plt.figure(figsize=(10,5))
plt.plot(mlp_paridad.errors, label='Error Paridad')
plt.title('Evolución del Error durante el Entrenamiento - Paridad')
plt.xlabel('Épocas')
plt.ylabel('Error')
plt.legend()
plt.show()

mlp_paridad.train(X=datos, y=y_paridad)

plt.figure(figsize=(10,5))
plt.plot(mlp_paridad.accuracies, label='Precisión Paridad')
plt.title('Evolución del precisión durante el Entrenamiento - Paridad')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# def añadir_ruido_gaussiano(X, mean=0, std=0.5):
#     ruido = np.random.normal(mean, std, X.shape)
#     X_con_ruido = X + ruido
#     # Asegurarse de que los valores estén entre 0 y 1
#     X_con_ruido = np.clip(X_con_ruido, 0, 1)
#     return X_con_ruido

# datos_con_ruido = añadir_ruido_gaussiano(datos)

# print(datos_con_ruido)
# nuevo_array= np.zeros((70,5))
# for i in range(len(datos_con_ruido)):
#      dato=datos_con_ruido[i].reshape((7,5))
#      for j in range(7):
#           nuevo_array[i*7+j]= dato[j]

# print(nuevo_array)
    
# np.savetxt("./data/datos_con_ruido.csv", nuevo_array, delimiter=",", fmt='%f')  # fmt='%d' para enteros

