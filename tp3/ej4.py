import keras
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from src.layers import MLPDigitos

data= keras.datasets.mnist.load_data(path="./data/mnist.npz")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train.shape, x_test.shape

x_train.max(), x_train.min()

x_train[0]

x_train_n= np.asarray((x_train.reshape(60000,28*28 )-x_train.min())/(x_train.max()-x_train.min()))
x_test_n= np.asarray((x_test.reshape(10000,28*28 )-x_test.min())/(x_test.max()-x_test.min()))

x_train_n.shape, x_train_n.min(), x_train_n.max()

y_train.shape

mlp_clasificacion = MLPDigitos(
    input_size=784,
    hidden_size=100,
    learning_rate=0.001,
    epochs=1000,
    weight_update_method='Adam'  # Opciones: 'GD', 'Momentum', 'Adam'
)

classes= np.unique(y_train)

enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary')
enc.fit(y_train.reshape(-1, 1))
y_train_one_hot= enc.transform(y_train.reshape(-1, 1)).toarray()
y_test_one_hot= enc.transform(y_test.reshape(-1, 1)).toarray()

mlp_clasificacion.train(X=x_train_n, y=y_train_one_hot)



plt.figure(figsize=(10,5))
plt.plot(mlp_clasificacion.errors, label='Error de Clasificación')
plt.title('Evolución del Error durante el Entrenamiento - Clasificación')
plt.xlabel('Épocas')
plt.ylabel('Error')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(mlp_clasificacion.accuracies, label='Precisión Paridad')
plt.title('Evolución del precisión durante el Entrenamiento - Paridad')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Función para mostrar la matriz de confusión
def mostrar_matriz_confusion(y_true, y_pred, classes, title='Matriz de Confusión'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta real')
    plt.show()

def evaluar_digitos_con_matriz(mlp, X, y_true):
    _, output = mlp.forward(X)
    y_pred = np.argmax(output, axis=1)
    y_true_digits = np.argmax(y_true, axis=1)  # Convertir one-hot a clases
    mostrar_matriz_confusion(y_true_digits, y_pred, classes=list(range(10)), title="Matriz de Confusión - Dígitos")

# Función para evaluar la precisión en dígitos
def evaluar_precision_digitos(modelo, X, y_true):
    _, output = modelo.forward(X)
    predictions = np.argmax(output, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    accuracy = np.mean(predictions == true_labels)
    return accuracy

evaluar_digitos_con_matriz(mlp_clasificacion, x_test_n, y_test_one_hot)

precision_digitos_ruido = evaluar_precision_digitos(mlp_clasificacion, x_test_n, y_test_one_hot)

print(f"Precisión en Clasificación de Dígitos: {precision_digitos_ruido*100:.2f}%")