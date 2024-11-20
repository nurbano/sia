import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def import_json(file_name):

    f= open(file_name, 'r')
    j=json.load(f)  
    f.close()
    return { atr: j[atr] for atr in j}

# Función para convertir los valores hexadecimales a binarios de 7x5
def to_bin_array(encoded_character):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(7):
        current_row = encoded_character[row]
        for col in range(5):
            bin_array[row][4 - col] = current_row & 1
            current_row >>= 1
    return bin_array

# Visualizar los 32 caracteres en una grilla de 5 columnas y 7 filas
def view_all_characters(fonts):
    num_caracteres = len(fonts)  # Número total de caracteres
    columnas = 5  # Número de columnas para mostrar
    filas = (num_caracteres // columnas) + (num_caracteres % columnas > 0)  # Cálculo de filas necesarias

    plt.figure(figsize=(10, 10))
    for i in range(num_caracteres):
        plt.subplot(filas, columnas, i + 1)
        binary_matrix = to_bin_array(fonts[i])
        sns.heatmap(binary_matrix, cmap="binary", cbar=False, square=True,
                    linewidths=0.2, linecolor='black')
        plt.axis('off')
        plt.title(f'{i}', fontsize=10)

    plt.tight_layout()
    plt.show()

# Definir una función para interpolar entre dos puntos en el espacio latente
def interpolate_latent_points(point1, point2, num_steps=10):
    """
    Interpolación lineal entre dos puntos en el espacio latente.
    """
    ratios = np.linspace(0, 1, num_steps)
    interpolated_points = np.array([(1 - ratio) * point1 + ratio * point2 for ratio in ratios])
    return interpolated_points