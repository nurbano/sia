from src.tools import import_json, cross_validation
from src.tools import cargar_datos, normalizar, cargar_datos_digitos
from src.layers import Perceptron, MLPParidad, MLPDigitos

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
    datos = cargar_datos_digitos(data["file"])
    x = [fila[:-1] for fila in datos]  # Las tres primeras columnas son las entradas
    y = [fila[-1] for fila in datos]   # La última columna es la salida
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

    
mlp_clasificacion = MLPDigitos(
    input_size=data["input_size"],
    hidden_size=data["hidden_size"],
    learning_rate=data["learning_rate"],
    epochs=data["epochs"],
    weight_update_method=data["weight_update_method"]  # Opciones: 'GD', 'Momentum', 'Adam'
)

# Entrenar la red neuronal con los datos limpios
mlp_clasificacion.train(X=datos, y=y_digitos)

plt.figure(figsize=(10,5))
plt.plot(mlp_clasificacion.errors, label='Error de Clasificación')
plt.title('Evolución del Error durante el Entrenamiento - Clasificación')
plt.xlabel('Épocas')
plt.ylabel('Error')
plt.legend()
plt.show()



plt.figure(figsize=(10,5))
plt.plot(mlp_clasificacion.accuracies, label='Precisión Paridad')
plt.title('Evolución del precisión durante el Entrenamiento - Paridad')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()
