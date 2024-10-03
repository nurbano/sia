import numpy as np
import random
# Definimos la clase Perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100, activation_function=None):
        # Inicializamos pesos y bias
        np.random.seed(seed=20)
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_history = []  # Para registrar la evolución de los pesos
        # Establecemos la función de activación
        self.activation_function = activation_function if activation_function else self.step_function


    # Definimos la función escalón por defecto para clasificación
    def step_function(self, value):
        return 1 if value >= 0 else -1

    # Definimos una función lineal para regresión
    def linear_function(self, value):
        return value

    # Definimos una función sigmoide para un perceptrón no lineal
    def sigmoid_function(self, value):
        return 1 / (1 + np.exp(-value))

    def activation_derivative(self, x):
        if self.activation_function == 'lineal':
            return 1
        elif self.activation_function== 'step':
            return 1
        elif self.activation_function == 'sigmoide':
            #sig = self.activation(x)
            sig = self.sigmoid_function(x)
            return sig * (1 - sig)
        else:
            raise ValueError("Función de activación no soportada")
        
    # Algoritmo de entrenamiento del perceptrón
    def train(self, x, y):
        errors_per_epoch = []  # Para almacenar el error total por época
        
        for epoch in range(self.epochs):
            total_error = 0  # Inicializamos el error total por época
            mse= 0
            for i in range(len(x)):
                # Calculamos la salida del perceptrón
                #weighted_sum = sum(self.weights[j] * x[i][j] for j in range(len(x[0]))) + self.bias
                weighted_sum= np.dot(self.weights,  x[i]) + self.bias
                output = self.activation_function(weighted_sum)

                # Calculamos el error
                error = y[i] - output
                total_error += abs(error)  # Acumulamos el error absoluto
                mse+= error**2             # Acumulo el Error cuadrático MEDIO
                # Actualizamos los pesos y el bias si hay error
                # for j in range(len(self.weights)):
                #     self.weights[j] += self.learning_rate * error * x[i][j]
                #print(f'Error : {error} | x[i] shape: {x[i]}')

                #Calculo del Gradiente
                gradiente= error*self.activation_derivative(weighted_sum)
                self.weights+=self.learning_rate * error * np.asarray(x[i])
                #print(self.weights)
                self.bias += self.learning_rate * error

            # Guardamos el error total para esta época
            errors_per_epoch.append(total_error)

            # Mostrar los pesos, bias y error total para esta época
            print(f"Época {epoch+1}: Pesos: {self.weights}, Bias: {self.bias}, Error total: {total_error}")

        return errors_per_epoch

    # Función para predecir nuevas entradas
    def predict(self, x):
        result = []
        for i in range(len(x)):
            weighted_sum = sum(self.weights[j] * x[i][j] for j in range(len(x[0]))) + self.bias
            output = self.activation_function(weighted_sum)
            result.append(output)
        return result