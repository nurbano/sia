from dataset.font3 import Font3
from src.tools import to_bin_array, view_all_characters, import_json, interpolate_latent_points
import numpy as np
import matplotlib.pyplot as plt

from src.layers import Autoencoder
from src.losses import mean_squared_error, binary_cross_entropy

import sys
import argparse

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
import seaborn as sns


cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="Autoencoder"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")

parser.add_argument("--config_json", default="config.json", help="Path to json config file")

args= parser.parse_args(cmd_args)

data= import_json(args.config_json)
# Hiperparámetros
epochs = data["epochs"]
batch_size = data["batch_size"]
learning_rate = data["learning_rate"]
seed= data["seed"]
np.random.seed(seed)

noised_data= np.load(f'./dataset/{data["dataset"]}.npy')
flattened_data= np.load('./dataset/font3.npy')

# if data["dataset"]=="Font3":
#     flattened_data = np.array([to_bin_array(char).flatten() for char in Font3])

autoencoder = Autoencoder(learning_rate=learning_rate)

# Almacenar los valores de BCE y MSE
bce_history = []
mse_history = []

# Entrenamiento
for epoch in range(epochs):
    # Mezclar los datos
    indices = np.arange(flattened_data.shape[0])
    np.random.shuffle(indices)
    X_shuffled = flattened_data[indices]

    # Dividir en lotes
    # for i in range(0, flattened_data.shape[0], batch_size):
    #     X_batch = X_shuffled[i:i+batch_size]

    #     # Forward pass
    #     reconstructed_output = autoencoder.forward(X_batch)

    #     # Calcular la pérdida
    #     bce_loss = binary_cross_entropy(X_batch, reconstructed_output)
    #     mse_loss = mean_squared_error(X_batch, reconstructed_output)

    #     # Backward pass
    #     autoencoder.backward(X_batch, reconstructed_output)
    
    #reconstructed_output = autoencoder.forward(flattened_data)
    
    reconstructed_output = autoencoder.forward(noised_data)
    bce_loss = binary_cross_entropy(flattened_data, reconstructed_output)
    mse_loss = mean_squared_error(flattened_data, reconstructed_output)
    autoencoder.backward(flattened_data, reconstructed_output)
    # Almacenar las pérdidas al final de cada época
    bce_history.append(bce_loss)
    mse_history.append(mse_loss)

    # Imprimir cada 100 épocas
    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch + 1}/{epochs} - BCE: {bce_loss:.4f} - MSE: {mse_loss:.4f}")



if data["show_train"]== "True":
    plt.figure(figsize=(10, 6))
    plt.plot(bce_history, label='BCE')
    plt.plot(mse_history, label='MSE')
    plt.title('Curvas de Pérdida durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Valor de la Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()

def encode(self, X):
    return self.encoder.forward(X)

# Lo añadimos a la clase Autoencoder (si no se ha hecho ya)
Autoencoder.encode = encode

# Codificar los datos en el espacio latente
encoded_imgs = autoencoder.encode(flattened_data)

# Visualización de la distribución de los caracteres en el espacio latente
character_labels = [
    '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL'
]


if data["show_latent_space"]== "True":
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(character_labels):
        plt.scatter(encoded_imgs[i, 0], encoded_imgs[i, 1])
        plt.text(encoded_imgs[i, 0] + 0.05, encoded_imgs[i, 1] + 0.05, label, fontsize=9)

    plt.title('Distribución de los caracteres en el espacio latente con etiquetas')
    plt.xlabel('Dimensión Latente 1')
    plt.ylabel('Dimensión Latente 2')
    plt.grid(True)
    plt.show()

def reconstruct(self, X):
    return self.forward(X)

# Lo añadimos a la clase Autoencoder (si no se ha hecho ya)
Autoencoder.reconstruct = reconstruct

# Visualización de los caracteres originales y reconstruidos
n = 32  # Número total de caracteres
num_cols = 5  # Número de pares por fila
num_rows = n // num_cols + (n % num_cols > 0)  # Calculamos el número de filas necesarias

if data["show_letters"]=="True":
    plt.figure(figsize=(20, num_rows * 2))
    pixels_dif = []
    for i in range(n):
        # Original
        ax = plt.subplot(num_rows * 2, num_cols, i + 1 + (i // num_cols) * num_cols)
        # original_img = flattened_data[i].reshape(7, 5)
        original_img = noised_data[i].reshape(7, 5)
        plt.imshow(original_img, cmap="binary")
        plt.title("Original")
        plt.axis("off")

        # Reconstruido (debajo del original)
        ax = plt.subplot(num_rows * 2, num_cols, i + 1 + num_cols + (i // num_cols) * num_cols)
        # reconstructed = autoencoder.reconstruct(flattened_data[i].reshape(1, -1))
        reconstructed = autoencoder.reconstruct(noised_data[i].reshape(1, -1))
        decoded_img = (reconstructed > 0.5).astype(int).reshape(7, 5)

        # Calcular la diferencia de píxeles
        true_image= flattened_data[i].reshape(7, 5)
        #dif = np.sum(np.abs(original_img - decoded_img))
        dif = np.sum(np.abs(true_image - decoded_img))
        pixels_dif.append(dif)

        plt.imshow(decoded_img, cmap="binary")
        plt.title("Reconstruido")
        plt.axis("off")

    print("Promedio de diferencias de píxeles:", np.mean(pixels_dif))
    plt.tight_layout()
    plt.show()

if data["evaluate"]=="True":
    evaluated_data= np.load(f'./dataset/{data["ev_dataset"]}.npy')
    plt.figure(figsize=(20, num_rows * 2))
    pixels_dif = []
    for i in range(n):
        # Original
        ax = plt.subplot(num_rows * 2, num_cols, i + 1 + (i // num_cols) * num_cols)
        # original_img = flattened_data[i].reshape(7, 5)
        original_img =  evaluated_data[i].reshape(7, 5)
        plt.imshow(original_img, cmap="binary")
        plt.title("Original")
        plt.axis("off")

        # Reconstruido (debajo del original)
        ax = plt.subplot(num_rows * 2, num_cols, i + 1 + num_cols + (i // num_cols) * num_cols)
        # reconstructed = autoencoder.reconstruct(flattened_data[i].reshape(1, -1))
        reconstructed = autoencoder.reconstruct(evaluated_data[i].reshape(1, -1))
        decoded_img = (reconstructed > 0.5).astype(int).reshape(7, 5)

        # Calcular la diferencia de píxeles
        true_image= flattened_data[i].reshape(7, 5)
        #dif = np.sum(np.abs(original_img - decoded_img))
        dif = np.sum(np.abs(true_image - decoded_img))
        pixels_dif.append(dif)

        plt.imshow(decoded_img, cmap="binary")
        plt.title("Reconstruido")
        plt.axis("off")

    print("Promedio de diferencias de píxeles:", np.mean(pixels_dif))
    plt.tight_layout()
    plt.show()


if data["interpolar"]=="True":
    # Seleccionar los caracteres para la interpolación
    char_start = data["char_start"]
    char_end = data["char_end"]
    index_inicio = character_labels.index(char_start)
    index_final = character_labels.index(char_end)

    latent_point_inicio = encoded_imgs[index_inicio]
    latent_point_final = encoded_imgs[index_final]

    # Generar puntos intermedios entre los dos caracteres
    num_interpolations = 10
    interpolated_points = interpolate_latent_points(latent_point_inicio, latent_point_final, num_interpolations)

    # Decodificar los puntos interpolados
    decoded_interpolations = autoencoder.decoder.forward(interpolated_points)

    # Binarizar las imágenes decodificadas
    threshold = 0.5
    binarized_interpolations = (decoded_interpolations > threshold).astype(int)

    # Visualizar los resultados
    plt.figure(figsize=(20, 4))
    for i, decoded_img in enumerate(binarized_interpolations):
        plt.subplot(1, num_interpolations, i + 1)
        plt.imshow(decoded_img.reshape(7, 5), cmap="binary")
        plt.title(f"Paso {i + 1}")
        plt.axis("off")
    plt.suptitle(f"Interpolación entre '{char_start}' y '{char_end}'")
    plt.show()



if data["show_similitud"]== "True":
    # Lista de caracteres correspondientes al índice en Font3
    
    
    caracteres = character_labels

    # Verificar si algún carácter tiene valores constantes
    constantes = np.apply_along_axis(lambda x: np.std(x) == 0, 1, flattened_data)
    print(f"Caracteres con valores constantes: {np.where(constantes)[0]}")

    # Convertir a float para agregar ruido
    flattened_data = flattened_data.astype(float)

    # Reemplazar filas constantes con un pequeño ruido aleatorio
    flattened_data[constantes] += np.random.rand(flattened_data.shape[1]) * 1e-6

    # Recalcular las correlaciones después de corregir los valores constantes
    similaridades_visuales = 1 - pairwise_distances(flattened_data, metric='correlation')

    # Visualizar la matriz de similitudes visuales
    plt.figure(figsize=(10, 8))
    sns.heatmap(similaridades_visuales, cmap="viridis", square=True, xticklabels=caracteres, yticklabels=caracteres)
    plt.title("Similitudes Visuales (Corregido)")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.show()

    # Calcula las distancias euclidianas en el espacio latente
    distancias_latentes = squareform(pdist(encoded_imgs, metric='euclidean'))

    # Visualizar la matriz de distancias latentes
    plt.figure(figsize=(10, 8))
    sns.heatmap(distancias_latentes, cmap="viridis_r", square=True, xticklabels=caracteres, yticklabels=caracteres)
    plt.title("Distancias en el Espacio Latente")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.show()

    # Correlación entre las matrices de distancias y similitudes visuales
    correlacion_matrices = np.corrcoef(distancias_latentes.flatten(), (1 - similaridades_visuales).flatten())[0, 1]
    print(f"Correlación entre las matrices: {correlacion_matrices:.4f}")