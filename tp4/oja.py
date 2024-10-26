import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from src.networks import OjaNetwork
from src.tools import import_json, standarization
from src.pca import PCA
import sys
import argparse



cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="OJA"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--config_json", default="config.json", help="Path to json config file")

args= parser.parse_args(cmd_args)

data= import_json(args.config_json)

df = pd.read_csv(data["dataset_path"])
oja= OjaNetwork(input_size= data["input_size"], 
                learning_rate= data["learning_rate"], 
                epochs=data["epochs"])
#print(df.head())
X= np.array(df.drop("Country", axis=1))
features= np.array(df.drop("Country", axis=1).columns)
countries= np.array(df["Country"])
X_scaled= standarization(X)

oja.train(X_scaled)
oja.plot_training()
oja_weights = oja.get_weights()
oja_indices = np.dot(X_scaled, oja_weights)
sign_oja= np.sign(oja_weights[0] )
pca= PCA(n=1)
pca.train(X_scaled)
pca_loadings= pca.autovectores()
PC1= pca.calc_PC1(X_scaled)
sign_pca= np.sign(pca_loadings[0][0])

signo= 1
#print(sign_oja, sign_pca)
if sign_oja!= sign_pca:
    signo=-1
dif= np.mean(pca_loadings-oja_weights*signo)
print("Diferencia: ", dif)
#Comparar loadings
plt.figure(figsize=(12, 6))
index = np.arange(len(features))
bar_width = 0.35
plt.bar(index, oja_weights*signo, bar_width, label='Oja weights')
plt.bar(index + bar_width, pca_loadings[0], bar_width, label='PCA1 loadings')
plt.xlabel('Features')
plt.ylabel('Loadings')
plt.title('Comparación loadings por features')
plt.xticks(index + bar_width / 2, features, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

#Comparar PC1 vs OJA
plt.figure(figsize=(12, 6))
index = np.arange(len(countries))
bar_width = 0.35
plt.bar(index, oja_indices*signo, bar_width, label='Oja Index')
plt.bar(index + bar_width, PC1, bar_width, label='PCA Index')
plt.xlabel('País')
plt.ylabel('Índice de la Primera Componente Principal')
plt.title('Comparación de Índices por País')
plt.xticks(index + bar_width / 2, countries, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()