import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import math
import os
from scipy.stats import norm

def import_json(file_name):

    f= open(file_name, 'r')
    j=json.load(f)  
    f.close()
    return { atr: j[atr] for atr in j}


def from_bin_array(bin_array):
    encoded_character = []
    for row in range(7):
        current_row_value = 0
        for col in range(5):
            current_row_value |= (bin_array[row][4 - col] << col)
        encoded_character.append(current_row_value)
    return encoded_character

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


def noise_with_k(data, n , noise_k): 
    indexes = np.random.choice(np.arange(n), noise_k, False)
    #print(indexes)
    new_data= np.copy(data)
    for index in indexes:
        if new_data[index]==0:
            new_data[index]=1
        else:
            new_data[index]=0
    return new_data

# Definir una función para interpolar entre dos puntos en el espacio latente
def interpolate_latent_points(point1, point2, num_steps=10):
    """
    Interpolación lineal entre dos puntos en el espacio latente.
    """
    ratios = np.linspace(0, 1, num_steps)
    interpolated_points = np.array([(1 - ratio) * point1 + ratio * point2 for ratio in ratios])
    return interpolated_points

def get_smile_dataset():
    # Descargar y cargar el dataset de caritas felices
    
    if os.path.isfile("./dataset/full_numpy_bitmap_smiley_face.npy"):
        print("File exist")
    else:
        print("Download dataset...")
        url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/smiley%20face.npy"
        response = requests.get(url)
        with open("./dataset/full_numpy_bitmap_smiley_face.npy", "wb") as f:
            f.write(response.content)
    data = np.load("./dataset/full_numpy_bitmap_smiley_face.npy")

    # Normalizar las imágenes (valores entre 0 y 1)
    normalized_data = data / 255.0

    # Asegurarse de que las imágenes están aplanadas
    flattened_data = normalized_data[:100]  # Tomar un subconjunto si es necesario
    # Lista de índices de imágenes a eliminar
    indices_a_eliminar = [12, 22, 24, 26, 41, 43, 63, 65, 85, 88, 99,]  # Reemplaza con los índices que quieras eliminar

    # Crear un nuevo dataset excluyendo los índices seleccionados
    filtered_data = np.delete(flattened_data, indices_a_eliminar, axis=0)
    flattened_data=filtered_data
    return flattened_data

def plot_smile_dataset(flattened_data):
    num_images = len(flattened_data)  # Número total de imágenes en el dataset actual
    max_images = 100  # Máximo número de imágenes a mostrar
    images_to_display = min(num_images, max_images)  # Mostrar solo las disponibles o 100

    # Calcular el número dinámico de filas según las imágenes disponibles
    num_cols = 10  # Número fijo de columnas
    num_rows = math.ceil(images_to_display / num_cols)  # Filas necesarias

    plt.figure(figsize=(15, num_rows * 1.5))  # Ajustar tamaño dinámico según las filas
    for i in range(images_to_display):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(flattened_data[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.suptitle("Caritas Felices Filtradas", fontsize=16)
    plt.show()

def graficar_region_latente(autoencoder, n=15, digit_size=28, rango_latente_x=(2, 6.21), rango_latente_y=(2, 6.8)):
    """
    Genera y visualiza una cuadrícula de imágenes en una región específica del espacio latente.

    Args:
        autoencoder: Modelo autoencoder ya entrenado.
        n: Tamaño de la cuadrícula (n x n).
        digit_size: Dimensión de las imágenes generadas (28x28 píxeles).
        rango_latente_x: Tuple con los límites del eje x en el espacio latente (min_x, max_x).
        rango_latente_y: Tuple con los límites del eje y en el espacio latente (min_y, max_y).
    """
    # Crear una figura vacía de dimensiones apropiadas
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generar coordenadas en el rango especificado
    grid_x = np.linspace(rango_latente_x[0], rango_latente_x[1], n)
    grid_y = np.linspace(rango_latente_y[0], rango_latente_y[1], n)

    # Generar las caritas en el espacio latente
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])  # Muestra en el espacio latente
            decoded_sample = autoencoder.decoder.forward(z_sample)  # Decodificar desde el espacio latente
            decoded_face = decoded_sample[0].reshape(digit_size, digit_size)  # Reshape para la visualización
            # Asignar la carita a la figura
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = decoded_face

    # Visualizar las caritas generadas
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.title(f"Distribución en el espacio latente (X: {rango_latente_x}, Y: {rango_latente_y})")
    plt.axis("off")
    plt.show()


def graficar_region_latente_auto(autoencoder, n=15, digit_size=28, puntos_latentes=None):
    """
    Genera y visualiza una cuadrícula de imágenes en una región específica del espacio latente, automáticamente definida.

    Args:
        autoencoder: Modelo autoencoder ya entrenado.
        n: Tamaño de la cuadrícula (n x n).
        digit_size: Dimensión de las imágenes generadas (28x28 píxeles).
        puntos_latentes: Puntos codificados en el espacio latente (opcional, se calculan si no se proporcionan).
    """
    # Si no se proporcionan puntos latentes, calcularlos automáticamente
    if puntos_latentes is None:
        puntos_latentes = np.array([autoencoder.encoder.forward(face.reshape(1, -1))[0] for face in flattened_data])

    # Calcular rango automático basado en la distribución de los puntos latentes
    min_x, max_x = puntos_latentes[:, 0].min(), puntos_latentes[:, 0].max()
    min_y, max_y = puntos_latentes[:, 1].min(), puntos_latentes[:, 1].max()

    # Crear figura vacía
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generar coordenadas en el rango especificado
    grid_x = np.linspace(min_x, max_x, n)
    grid_y = np.linspace(min_y, max_y, n)

    # Generar las caritas en el espacio latente
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])  # Muestra en el espacio latente
            decoded_sample = autoencoder.decoder.forward(z_sample)  # Decodificar desde el espacio latente
            decoded_face = decoded_sample[0].reshape(digit_size, digit_size)  # Reshape para visualización
            # Asignar la carita a la figura
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = decoded_face

    # Visualizar las caritas generadas
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.title(f"Distribución en el espacio latente (X: [{min_x:.2f}, {max_x:.2f}], Y: [{min_y:.2f}, {max_y:.2f}])")
    plt.axis("off")
    plt.show()


def generate_noise(flattened_data, k):
    flattened_data_noise= np.empty_like(flattened_data)
    for i in range(len(flattened_data)):
        flattened_data_noise[i]= noise_with_k(flattened_data[i], 35, int(k))
    return flattened_data_noise