import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from src.tools import plot_comb, plot_dataset, calculate_orto, plot_estados, obtain_letter_matrix
from src.networks import Hopfield
from src.tools import import_json
import sys
import argparse

df = pd.read_csv("./data/full_alphabet_complete_dataset.csv")

cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="Hopfield"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--config_json", default="config.json", help="Path to json config file")

args= parser.parse_args(cmd_args)

data= import_json(args.config_json)

#plot_dataset(df)

#Para calcular la ortogonalidad de todas las combinaciones:
#df_orto= calculate_orto(df, True)
df_orto= pd.read_csv("./data/calc_orto.csv")
print(df_orto.head())
print(df_orto.tail())

best= df_orto.sort_values(by="AVG")["Comb"].iloc[0]
worst= df_orto.sort_values(by="AVG", ascending=False)["Comb"].iloc[0]
print(f'Best Comb: {best} | Worst Comb: {worst}')
plot_comb("Patrones", data["comb"], df)
hopfield= Hopfield(input_dim=data["input_dim"], iter=data["iter"], convergence=data["conv"])
#comb= ["x","e","g","d"]
W= hopfield.calculate_weights(data["comb"], df)

pattern= obtain_letter_matrix(data["letter"], df)
pattern= hopfield.noise_with_k(pattern, data["k_noise"])

estados, energia= hopfield.iterate_state(pattern)
print(energia[-1])
plot_estados(estados)
plt.show()
plt.plot(energia)
plt.xlabel("Iteraciones")
plt.ylabel("Energía")
plt.show()
