from dataset.font3 import Font3
from src.tools import to_bin_array, view_all_characters, import_json, interpolate_latent_points
from src.tools import get_smile_dataset, plot_smile_dataset, graficar_region_latente
import numpy as np
import matplotlib.pyplot as plt

from src.layers import Autoencoder_vae
from src.losses import mean_squared_error, binary_cross_entropy, loss_function

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
#np.random.seed(seed)
if data["dataset"]=="Font3":
    flattened_data = np.array([to_bin_array(char).flatten() for char in Font3])
    input_size= 35
    hidden_size= 18

if data["dataset"]=="smile":
    flattened_data = get_smile_dataset()
    input_size= 784
    hidden_size= 256
if data["show_letters"]=="True":
    plot_smile_dataset(flattened_data)
    
autoencoder = Autoencoder_vae(input_size=input_size, hidden_size=hidden_size, latent_size=2, learning_rate=learning_rate)

# Almacenar las pérdidas
total_loss_history = []
reconstruction_loss_history = []
kl_loss_history = []
kl_weight=[]
# Entrenamiento
for epoch in range(epochs):
    # Mezclar los datos
    indices = np.arange(flattened_data.shape[0])
    np.random.shuffle(indices)
    X_shuffled = flattened_data[indices]

    # Dividir en lotes
    for i in range(0, flattened_data.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]

        # Forward pass
        reconstructed_output = autoencoder.forward(X_batch)
        if data["kl_variable"]=="True":
            kl_weight_var= epoch/epochs
        else:
            kl_weight_var= 1
        # Calcular la pérdida
        total_loss, reconstruction_loss, kl_loss, kl = loss_function(
            X_batch, reconstructed_output, 
            autoencoder.encoder.z_mean, 
            autoencoder.encoder.z_log_var,  kl_weight_var
        )
        kl_weight.append(kl)
        # Backward pass
        autoencoder.backward(X_batch, reconstructed_output,  kl_weight_var)
       

    # Almacenar las pérdidas al final de cada época
    total_loss_history.append(total_loss)
    reconstruction_loss_history.append(reconstruction_loss)
    kl_loss_history.append(kl_loss)

    # Imprimir cada 100 épocas
    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch + 1}/{epochs} - Pérdida Total: {total_loss:.4f} - Reconstrucción: {reconstruction_loss:.4f} - KL Divergence: {kl_loss:.4f}")

# Graficar las pérdidas
if data["show_train"]=="True":
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history, label='Pérdida Total')
    plt.plot(reconstruction_loss_history, label='Pérdida de Reconstrucción')
    plt.plot(kl_loss_history, label='Divergencia KL')
    plt.title('Curvas de Pérdida durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Valor de la Pérdida')
    plt.legend()
    plt.grid(True)
    plt.show()
    if data["kl_variable"]=="True":
        plt.title('Variación del peso de KL')
        plt.xlabel('Épocas')
        plt.ylabel('Peso de KL')
        plt.plot(kl_weight)
        plt.show()

if data["show_latent_region"]=="True":
    graficar_region_latente(autoencoder, n=20, digit_size=28, rango_latente_x=(-6, 1), rango_latente_y=(-2, 2))

if data["show_latent_space"]=="True":
    # Visualizar los puntos en el espacio latente
    encoded_faces = np.array([autoencoder.encoder.forward(face.reshape(1, -1))[0] for face in flattened_data])

    plt.figure(figsize=(10, 8))
    plt.scatter(encoded_faces[:, 0], encoded_faces[:, 1], alpha=0.7, c='blue', edgecolors='k')
    plt.title('Distribución de las Caritas en el Espacio Latente')
    plt.xlabel('Dimensión Latente 1')
    plt.ylabel('Dimensión Latente 2')
    plt.grid(True)
    plt.show()