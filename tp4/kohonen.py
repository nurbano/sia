import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from src.networks import Kohonen
from src.tools import import_json, standarization

import sys
import argparse



cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="Kohonen"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--config_json", default="config.json", help="Path to json config file")

args= parser.parse_args(cmd_args)

data= import_json(args.config_json)

df = pd.read_csv(data["dataset_path"])

# X= np.array(df.drop("Country", axis=1))
# features= np.array(df.drop("Country", axis=1).columns)
# countries= np.array(df["Country"])
# X_scaled= standarization(X)

# Instanciar la clase Kohonen
kohonen_net = Kohonen(
    K=6,
    input_dim=data["input_dim"],  # Restamos 1 porque la primera columna es 'Country'
    learning_rate=0.1,
    radius=4,
    iterations=1000
)
normalized_data = kohonen_net.normalize_data(df)
kohonen_net.train(normalized_data)

# Visualizar el Quantization Error
kohonen_net.plot_quantization_error()

# Visualizar la U-Matrix
kohonen_net.plot_u_matrix()

# Visualizar el mapa de calor de frecuencias
kohonen_net.plot_heatmap()

# Visualizar la representación bidimensional
kohonen_net.plot_2d_representation(normalized_data)

# Mostrar los países agrupados por clúster
kohonen_net.show_clustered_countries()