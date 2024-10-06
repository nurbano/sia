import json
import csv
import matplotlib.pyplot as plt
import numpy as np

#Función para importar parámetros de un json
def import_json(file_name):

    f= open(file_name, 'r')
    j=json.load(f)  
    f.close()
    return { atr: j[atr] for atr in j}


# Función para cargar datos desde un archivo CSV
def cargar_datos(nombre_archivo):
    datos = []
    with open(nombre_archivo, 'r') as archivo:
        lector_csv = csv.reader(archivo)
        next(lector_csv)  # Saltar encabezado
        for fila in lector_csv:
            try:
                datos.append([float(valor) for valor in fila])
            except ValueError as e:
                print(f"Error en la fila: {fila}, Error: {e}")
    return datos

def normalizar(x):
    x_min= x.min()
    x_max= x.max()
    x_norm= (x-x.min())/(x.max()-x.min())
    return x_norm

def k_fold_split(X, y, k):
    X = np.array(X)
    y = np.array(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_sizes = len(X) // k
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_sizes: (i + 1) * fold_sizes]
        train_indices = np.setdiff1d(indices, test_indices)
        folds.append((X[train_indices], y[train_indices], X[test_indices], y[test_indices]))
    return folds

def cross_validation(perceptron_class, X, y, k=5, **kwargs):
    folds = k_fold_split(X, y, k)
    validation_errors = []
    for i, (train_X, train_y, test_X, test_y) in enumerate(folds):
        #print(f"Fold {i+1}/{k}")
        perceptron = perceptron_class(input_size=train_X.shape[1], **kwargs)
        perceptron.train(train_X, train_y)
        predictions = perceptron.predict(test_X)
        mse = np.mean((test_y - predictions) ** 2)
        validation_errors.append(mse)
        print(f"MSE de Validación para el fold {i+1} de {k}: {mse}")
    avg_validation_error = np.mean(validation_errors)
    return avg_validation_error

def plot_errors(errors, title):
    plt.figure()
    plt.plot(errors, marker='o')
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Error Cuadrático Medio')
    plt.grid(True)
    plt.show()

def plot_weight_evolution(weight_history, title):
    plt.figure()
    for i in range(len(weight_history[0])):
        weights = [weights[i] for weights in weight_history]
        plt.plot(weights, label=f'Peso {i+1}')
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Valor del Peso')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions_vs_real(y_true, y_pred, title):
    plt.figure()
    plt.scatter(range(len(y_true)), y_true, label='Valores Reales')
    plt.scatter(range(len(y_pred)), y_pred, label='Predicciones')
    plt.title(title)
    plt.xlabel('Índice de Muestra')
    plt.ylabel('Valor de Salida')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para cargar los datos de los dígitos desde un archivo de texto
def cargar_datos_digitos(filename):
    datos = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        num_images = len(lines) // 7  # Cada imagen tiene 7 líneas
        for i in range(num_images):
            image_data = []
            for j in range(7):  # 7 filas por imagen
                line_index = i * 7 + j
                linea = lines[line_index]
                fila = [int(x) for x in linea.strip().split()]
                image_data.extend(fila)  # Añadir los 5 píxeles de la fila
            if len(image_data) == 35:  # 7 filas * 5 columnas
                datos.append(image_data)
    return np.array(datos)
def cargar_datos_digitos_ruido(filename):
    datos = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        num_images = len(lines) // 7  # Cada imagen tiene 7 líneas
        for i in range(num_images):
            image_data = []
            for j in range(7):  # 7 filas por imagen
                line_index = i * 7 + j
                linea = lines[line_index]
                fila = [float(x) for x in linea.strip().split(",")]
                image_data.extend(fila)  # Añadir los 5 píxeles de la fila
            if len(image_data) == 35:  # 7 filas * 5 columnas
                datos.append(image_data)
    return np.array(datos)


def mostrar_imagen(pixels, titulo=''):
    plt.imshow(np.reshape(pixels, (7,5)), cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()
