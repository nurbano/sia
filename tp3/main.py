from src.tools import import_json

from src.layers import Perceptron

import argparse
import time
import sys
import timeit
import numpy as np
import matplotlib.pyplot as plt

cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="TP3 Perceptrón"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--config_json", default="config.json", help="Path to json config file")

args= parser.parse_args(cmd_args)

data= import_json(args.config_json)

name_config= args.config_json.split(".json")[0].split("./config/")[1] #En windows usar .\\config\\


print(f'Tamaño de entrada: {data["input_size"]} | Épocas: {data["epochs"]} | Tasa de Aprendizaje: {data["learning_rate"]} | Activación: {data["activation_function"]}')


x= data["x"]
y= data["y"]
print(f'x: {x}')
print(f'y: {y}')
# Inicializamos el perceptrón con función escalón
perceptron= Perceptron(input_size=data["input_size"],
                             learning_rate=data["learning_rate"], 
                             epochs=data["epochs"], 
                             activation_function=None)

errors= perceptron.train(x, y)

plt.plot(errors, marker='o', color='red')
plt.title(f'Evolución del Error para la config {name_config}')
plt.xlabel('Épocas')
plt.ylabel('Error total')
plt.grid(True)
plt.savefig(f'./images/plot_error_{name_config}.png')
plt.show()
