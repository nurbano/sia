from src.tools import noise_with_k
from src.tools import to_bin_array, view_all_characters, from_bin_array
from dataset.font3 import Font3
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import pandas as pd

cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="Generate Noise"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--k", default="1", help="Pixel to invert")

args= parser.parse_args(cmd_args)


flattened_data = np.array([to_bin_array(char).flatten() for char in Font3])

flattened_data_noise= np.empty_like(flattened_data)
for i in range(len(flattened_data)):
    flattened_data_noise[i]= noise_with_k(flattened_data[i], 35, int(args.k))


n = 32  # Número total de caracteres
num_cols = 5  # Número de pares por fila
num_rows = n // num_cols + (n % num_cols > 0)  # Calculamos el número de filas necesarias

for i in range(32):
    flattened_data[i].reshape(1, -1)
    ax = plt.subplot(num_rows * 2, num_cols, i + 1 + (i // num_cols) * num_cols)
    plt.imshow(flattened_data_noise[i].reshape(7, 5), cmap="binary")
    print(f'{flattened_data_noise[i].reshape(7, 5):02x}')
    plt.title("Original")
    plt.axis("off")
plt.tight_layout()
plt.show()

print(from_bin_array(flattened_data_noise))
#view_all_characters(Font3)
