from dataset.font3 import Font3
from src.tools import to_bin_array, view_all_characters
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, losses
import matplotlib.pyplot as plt

flattened_data = np.array([to_bin_array(char).flatten() for char in Font3])
print("Dataset aplanado listo para el entrenamiento:", flattened_data.shape)

#view_all_characters(Font3)

# Parámetros del modelo
input_dim = 35        # Número de píxeles de cada imagen (7x5)
latent_dim = 2        # Dimensión del espacio latente (para compresión)
intermediate_dim = 18 # antes 16 Número de unidades en la capa intermedia

# Definición del Codificador
input_img = Input(shape=(input_dim,), name="input_layer")
encoded = Dense(intermediate_dim, activation='relu', name="encoding_layer")(input_img)
latent_space = Dense(latent_dim, activation='linear', name="latent_space")(encoded)

# Definición del Decodificador
decoded = Dense(intermediate_dim, activation='relu', name="decoding_layer")(latent_space)
output_img = Dense(input_dim, activation='sigmoid', name="output_layer")(decoded)

# Creación del modelo Autoencoder completo
autoencoder = Model(input_img, output_img, name="autoencoder_model")

# Creación de modelos Codificador y Decodificador separados para análisis
encoder = Model(input_img, latent_space, name="encoder_model")

# Para el decodificador, definimos una entrada en el espacio latente
latent_input = Input(shape=(latent_dim,), name="latent_input")
decoder_layer1 = autoencoder.get_layer("decoding_layer")(latent_input)
decoder_output = autoencoder.get_layer("output_layer")(decoder_layer1)
decoder = Model(latent_input, decoder_output, name="decoder_model")

# Resumen del modelo completo
#autoencoder.summary()
losses_func= losses.BinaryCrossentropy(
       label_smoothing=0,
       from_logits=True,
    name='binary_crossentropy'
)
opt = optimizers.Adam(learning_rate=0.005)

autoencoder.compile(optimizer=opt, loss=losses_func, metrics=['mse'])


# Resumen para confirmar configuración
#autoencoder.summary()
# Hiperparámetros de entrenamiento
epochs = 1000       # Número de épocas para el entrenamiento
batch_size = 35     # Tamaño del lote

# Entrenamiento del Autoencoder
history = autoencoder.fit(
    flattened_data, flattened_data,   # Entrada y salida son el mismo dataset
    epochs=epochs,
    verbose= 1,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0  # Usamos el 20% de los datos para validación
)

# Visualización de la evolución de la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['mse'], label='MSE')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Evolución de la pérdida durante el entrenamiento')
plt.show()

# Codificar los datos en el espacio latente
encoded_imgs = encoder.predict(flattened_data)
# Visualización de la distribución de los caracteres en el espacio latente
plt.figure(figsize=(8, 6))
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1])
plt.title('Distribución de los caracteres en el espacio latente')
plt.xlabel('Dimensión Latente 1')
plt.ylabel('Dimensión Latente 2')
plt.show()

# Visualización de los 32 caracteres agrupados en filas de 5 pares (original y reconstruido)
n = 32  # Número total de caracteres
num_cols = 5  # Número de pares por fila
num_rows = n // num_cols + (n % num_cols > 0)  # Calculamos el número de filas necesarias

plt.figure(figsize=(20, num_rows * 2))
pixels_dif=[]
for i in range(n):
    # Original
    ax = plt.subplot(num_rows * 2, num_cols, i + 1 + (i // num_cols) * num_cols)
    plt.imshow(to_bin_array(Font3[i]), cmap="binary")
    plt.title("Original")
    plt.axis("off")

    # Reconstruido (debajo del original)
    ax = plt.subplot(num_rows * 2, num_cols, i + 1 + num_cols + (i // num_cols) * num_cols)
    #binarized_predictions = (encoded_imgs > 0.5).astype(int)  # Convert to 0 or 1

    decoded_img = (autoencoder.predict(flattened_data[i].reshape(1, -1)) > 0.5).astype(int)
    #decoded_img = autoencoder.predict(flattened_data[i].reshape(1, -1)) 
    print(autoencoder.predict(flattened_data[i].reshape(1, -1)))
    dif= 35-np.count_nonzero(flattened_data[i].reshape(1, -1)==decoded_img)
    print(dif)

    pixels_dif.append(dif)
    plt.imshow(decoded_img.reshape(7, 5), cmap="binary")
    plt.title("Reconstruido")
    plt.axis("off")

print("Promedio:", sum(pixels_dif)/len(pixels_dif))
plt.tight_layout()
plt.show()
