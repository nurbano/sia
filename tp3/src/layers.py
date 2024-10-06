import numpy as np
import random
import pandas as pd
# Definimos la clase Perceptron
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100, activation=None, iter=1):
        # Inicializamos pesos y bias
        #np.random.seed(seed=20)
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_history = []  # Para registrar la evolución de los pesos
        # Establecemos la función de activación
        self.iter= iter
        self.activation =activation

    def activation_function(self, value):
        if self.activation == 'lineal':
            #función lineal (regresión)
            return value
        elif self.activation== 'step':
            # Función escalón (clasificación binaria)
            return 1 if value >= 0 else -1
        elif self.activation== 'sigmoide':
            #función sigmoide para un perceptrón no lineal
            return 1 / (1 + np.exp(-value))
        
    def activation_derivative(self, x):
        if self.activation == 'lineal':
            return 1
        elif self.activation== 'step':
            return 1
        elif self.activation== 'sigmoide':
            #sig = self.activation(x) 
            #2*beta= 1 por lo tanto beta= 1/2
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError("Función de activación no soportada")
        
    # Algoritmo de entrenamiento del perceptrón
    def train(self, x, y):
        errors_per_epoch = []  # Para almacenar el error total por época
        df= pd.DataFrame({"Iteración": [], "Activación": [], "LR": [], "Época":[], "Pesos": [], "Bias": [], "MSE": []})
        for epoch in range(self.epochs):
            total_error = 0  # Inicializamos el error total por época
            
            for i in range(len(x)):
                # Calculamos la salida del perceptrón
                #weighted_sum = sum(self.weights[j] * x[i][j] for j in range(len(x[0]))) + self.bias
                weighted_sum= np.dot(self.weights,  x[i]) + self.bias #Exitación
                output = self.activation_function(weighted_sum) #Salida neurona

                # Calculamos el error
                error = y[i] - output  #Salida esperada menos la salida obtenida
                total_error += error ** 2  # ECM
                #total_error += abs(error)  # Acumulamos el error absoluto
                #mse+= error**2             # Acumulo el Error cuadrático MEDIO
                
                #Calculo del Gradiente
                gradiente= error*self.activation_derivative(weighted_sum)
               
                #Actualizo los parámetros
                self.weights+=self.learning_rate * gradiente * np.asarray(x[i])
                self.bias += self.learning_rate * gradiente

            # Guardamos el error total para esta época
            self.weight_history.append(self.weights.copy())
            mse = total_error / len(x) #El error depende de todos
            errors_per_epoch.append(mse)
            # Mostrar los pesos, bias y error total para esta época
            #print(f"Época {epoch+1}: Pesos: {self.weights}, Bias: {self.bias}, MSE: {mse}")
            #print(epoch+1)
            df_aux= pd.DataFrame({"Iteración": self.iter, "Activación": self.activation, "LR": self.learning_rate,"Época": epoch+1, "Pesos": [self.weights], "Bias": self.bias, "MSE": mse})
            #print(df_aux)
            df= pd.concat([df,df_aux]).reset_index(drop=True)
        return errors_per_epoch, df

    # Función para predecir nuevas entradas
    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        outputs = self.activation_function(weighted_sum)
        return outputs
    
class MLPParidad:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, epochs=1000, weight_update_method='GD'):
        # Inicializar pesos y biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, 1))
        self.bias_output = np.random.uniform(-1, 1, (1, 1))

        # Parámetros de entrenamiento
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_update_method = weight_update_method  # 'GD', 'Momentum', 'Adam'

        # Variables para Momentum
        self.velocity_w_ih = np.zeros_like(self.weights_input_hidden)
        self.velocity_b_h = np.zeros_like(self.bias_hidden)
        self.velocity_w_ho = np.zeros_like(self.weights_hidden_output)
        self.velocity_b_o = np.zeros_like(self.bias_output)
        self.momentum_gamma = 0.9

        # Variables para Adam
        self.m_w_ih = np.zeros_like(self.weights_input_hidden)
        self.v_w_ih = np.zeros_like(self.weights_input_hidden)
        self.m_b_h = np.zeros_like(self.bias_hidden)
        self.v_b_h = np.zeros_like(self.bias_hidden)
        self.m_w_ho = np.zeros_like(self.weights_hidden_output)
        self.v_w_ho = np.zeros_like(self.weights_hidden_output)
        self.m_b_o = np.zeros_like(self.bias_output)
        self.v_b_o = np.zeros_like(self.bias_output)
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_t = 0

        # Agregar listas para almacenar los pesos y biases en cada época
        self.weights_input_hidden_history = []
        self.bias_hidden_history = []
        self.weights_hidden_output_history = []
        self.bias_output_history = []

        # Para almacenar el error en cada época
        self.errors = []
        self.accuracies= []

    # Función sigmoide y su derivada
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    
    # Propagación hacia adelante
    def forward(self, X):
        # Capa oculta
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        # Capa de salida
        output = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output) + self.bias_output)
        
        return hidden_output, output

    # Retropropagación y actualización de pesos
    def train(self, X, y):
        for epoch in range(self.epochs):
            # Propagación hacia adelante
            hidden_output, output = self.forward(X)
            
            # Cálculo del error
            error = y.reshape(-1,1) - output

            # Calcular el error cuadrático medio
            mse = np.mean(np.square(error))
            self.errors.append(mse)
            # Calcular la precisión (accuracy) como la cantidad de predicciones correctas sobre el total

            predictions = (output >= 0.5).astype(int)
            correct_predictions = np.sum(predictions == y.reshape(-1,1))
            accuracy = correct_predictions / len(y)
            self.accuracies.append(accuracy)
            # Retropropagación
            delta_output = error * self.sigmoid_derivative(output)
            delta_hidden = delta_output.dot(self.weights_hidden_output.T) * self.sigmoid_derivative(hidden_output)

            # Actualización de pesos según el método seleccionado
            if self.weight_update_method == 'GD':
                self.update_weights_gd(X, hidden_output, delta_hidden, delta_output)
            elif self.weight_update_method == 'Momentum':
                self.update_weights_momentum(X, hidden_output, delta_hidden, delta_output)
            elif self.weight_update_method == 'Adam':
                self.update_weights_adam(X, hidden_output, delta_hidden, delta_output)

            # Guardar los pesos y biases actuales
            self.weights_input_hidden_history.append(self.weights_input_hidden.copy())
            self.bias_hidden_history.append(self.bias_hidden.copy())
            self.weights_hidden_output_history.append(self.weights_hidden_output.copy())
            self.bias_output_history.append(self.bias_output.copy())

            # Opcional: imprimir el error cada cierto número de épocas
            if (epoch+1) % 100 == 0:
                print(f"Paridad - Época {epoch+1}/{self.epochs} - Error: {mse:.4f} - Acc. {accuracy:.4f}")

    # Métodos de actualización de pesos (GD, Momentum, Adam)
    def update_weights_gd(self, X, hidden_output, delta_hidden, delta_output):
        # Actualizar pesos y biases de la capa de salida
        self.weights_hidden_output += self.learning_rate * hidden_output.T.dot(delta_output)
        self.bias_output += self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)

        # Actualizar pesos y biases de la capa oculta
        self.weights_input_hidden += self.learning_rate * X.T.dot(delta_hidden)
        self.bias_hidden += self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    def update_weights_momentum(self, X, hidden_output, delta_hidden, delta_output):
        # Actualizar pesos y biases de la capa de salida
        self.velocity_w_ho = self.momentum_gamma * self.velocity_w_ho + self.learning_rate * hidden_output.T.dot(delta_output)
        self.weights_hidden_output += self.velocity_w_ho
        self.velocity_b_o = self.momentum_gamma * self.velocity_b_o + self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)
        self.bias_output += self.velocity_b_o

        # Actualizar pesos y biases de la capa oculta
        self.velocity_w_ih = self.momentum_gamma * self.velocity_w_ih + self.learning_rate * X.T.dot(delta_hidden)
        self.weights_input_hidden += self.velocity_w_ih
        self.velocity_b_h = self.momentum_gamma * self.velocity_b_h + self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)
        self.bias_hidden += self.velocity_b_h

    def update_weights_adam(self, X, hidden_output, delta_hidden, delta_output):
        self.adam_t += 1

        # Parámetros de corrección de sesgo
        lr_t = self.learning_rate * (np.sqrt(1 - self.adam_beta2 ** self.adam_t) / (1 - self.adam_beta1 ** self.adam_t))

        # Gradientes para la capa de salida
        g_w_ho = hidden_output.T.dot(delta_output)
        g_b_o = np.sum(delta_output, axis=0, keepdims=True)

        # Actualización de momentos
        self.m_w_ho = self.adam_beta1 * self.m_w_ho + (1 - self.adam_beta1) * g_w_ho
        self.v_w_ho = self.adam_beta2 * self.v_w_ho + (1 - self.adam_beta2) * (g_w_ho ** 2)
        self.m_b_o = self.adam_beta1 * self.m_b_o + (1 - self.adam_beta1) * g_b_o
        self.v_b_o = self.adam_beta2 * self.v_b_o + (1 - self.adam_beta2) * (g_b_o ** 2)

        # Actualización de pesos y biases de la capa de salida
        m_hat_w_ho = self.m_w_ho / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_w_ho = self.v_w_ho / (1 - self.adam_beta2 ** self.adam_t)
        self.weights_hidden_output += lr_t * m_hat_w_ho / (np.sqrt(v_hat_w_ho) + self.adam_epsilon)

        m_hat_b_o = self.m_b_o / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_b_o = self.v_b_o / (1 - self.adam_beta2 ** self.adam_t)
        self.bias_output += lr_t * m_hat_b_o / (np.sqrt(v_hat_b_o) + self.adam_epsilon)

        # Gradientes para la capa oculta
        g_w_ih = X.T.dot(delta_hidden)
        g_b_h = np.sum(delta_hidden, axis=0, keepdims=True)

        # Actualización de momentos
        self.m_w_ih = self.adam_beta1 * self.m_w_ih + (1 - self.adam_beta1) * g_w_ih
        self.v_w_ih = self.adam_beta2 * self.v_w_ih + (1 - self.adam_beta2) * (g_w_ih ** 2)
        self.m_b_h = self.adam_beta1 * self.m_b_h + (1 - self.adam_beta1) * g_b_h
        self.v_b_h = self.adam_beta2 * self.v_b_h + (1 - self.adam_beta2) * (g_b_h ** 2)

        # Actualización de pesos y biases de la capa oculta
        m_hat_w_ih = self.m_w_ih / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_w_ih = self.v_w_ih / (1 - self.adam_beta2 ** self.adam_t)
        self.weights_input_hidden += lr_t * m_hat_w_ih / (np.sqrt(v_hat_w_ih) + self.adam_epsilon)

        m_hat_b_h = self.m_b_h / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_b_h = self.v_b_h / (1 - self.adam_beta2 ** self.adam_t)
        self.bias_hidden += lr_t * m_hat_b_h / (np.sqrt(v_hat_b_h) + self.adam_epsilon)

    # Método para cargar pesos de una época específica
    def load_weights(self, epoch):
        self.weights_input_hidden = self.weights_input_hidden_history[epoch].copy()
        self.bias_hidden = self.bias_hidden_history[epoch].copy()
        self.weights_hidden_output = self.weights_hidden_output_history[epoch].copy()
        self.bias_output = self.bias_output_history[epoch].copy()

class MLPDigitos:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, epochs=1000, weight_update_method='GD'):
        # Inicializar pesos y biases
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, 10))  # 10 salidas para los 10 dígitos
        self.bias_output = np.random.uniform(-1, 1, (1, 10))  # 10 salidas

        # Parámetros de entrenamiento
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_update_method = weight_update_method  # 'GD', 'Momentum', 'Adam'

        # Variables para Momentum
        self.velocity_w_ih = np.zeros_like(self.weights_input_hidden)
        self.velocity_b_h = np.zeros_like(self.bias_hidden)
        self.velocity_w_ho = np.zeros_like(self.weights_hidden_output)
        self.velocity_b_o = np.zeros_like(self.bias_output)
        self.momentum_gamma = 0.9

        # Variables para Adam
        self.m_w_ih = np.zeros_like(self.weights_input_hidden)
        self.v_w_ih = np.zeros_like(self.weights_input_hidden)
        self.m_b_h = np.zeros_like(self.bias_hidden)
        self.v_b_h = np.zeros_like(self.bias_hidden)
        self.m_w_ho = np.zeros_like(self.weights_hidden_output)
        self.v_w_ho = np.zeros_like(self.weights_hidden_output)
        self.m_b_o = np.zeros_like(self.bias_output)
        self.v_b_o = np.zeros_like(self.bias_output)
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_t = 0

        # Agregar listas para almacenar los pesos y biases en cada época
        self.weights_input_hidden_history = []
        self.bias_hidden_history = []
        self.weights_hidden_output_history = []
        self.bias_output_history = []

        # Para almacenar el error en cada época
        self.errors = []
        self.accuracies= []

    # Función sigmoide
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivada de la sigmoide
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Función Softmax
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Para estabilidad numérica
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Propagación hacia adelante
    def forward(self, X):
        # Capa oculta con sigmoide
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        # Capa de salida con Softmax
        output = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output_softmax = self.softmax(output)
        return hidden_output, output_softmax

    # Retropropagación y actualización de pesos
    def train(self, X, y):
        for epoch in range(self.epochs):
            # Propagación hacia adelante
            hidden_output, output_softmax = self.forward(X)

            # Cálculo del error (para Softmax + Cross-Entropy, el error es y - y_hat)
            error = y - output_softmax

            # Calcular el error cuadrático medio (MSE) solo para registro
            mse = np.mean(np.square(error))
            self.errors.append(mse)
               # Calcular la precisión (accuracy)

            # Convertir las salidas softmax a predicciones de clase (índice de la mayor probabilidad)
            predicted_classes = np.argmax(output_softmax, axis=1)
            #print(predicted_classes)
            # Convertir las etiquetas one-hot (y) a índices de clases verdaderas
            
            true_classes = np.argmax(y, axis=1)
            
            # Comparar las predicciones con las clases verdaderas y calcular la precisión
            accuracy = np.mean(predicted_classes == true_classes)
            self.accuracies.append(accuracy)
            # Retropropagación (error es suficiente para Softmax con Cross-Entropy)
            delta_output = error  # No necesitamos multiplicar por la derivada
            delta_hidden = delta_output.dot(self.weights_hidden_output.T) * self.sigmoid_derivative(hidden_output)

            # Actualización de pesos según el método seleccionado
            if self.weight_update_method == 'GD':
                self.update_weights_gd(X, hidden_output, delta_hidden, delta_output)
            elif self.weight_update_method == 'Momentum':
                self.update_weights_momentum(X, hidden_output, delta_hidden, delta_output)
            elif self.weight_update_method == 'Adam':
                self.update_weights_adam(X, hidden_output, delta_hidden, delta_output)

            # Guardar los pesos y biases actuales
            self.weights_input_hidden_history.append(self.weights_input_hidden.copy())
            self.bias_hidden_history.append(self.bias_hidden.copy())
            self.weights_hidden_output_history.append(self.weights_hidden_output.copy())
            self.bias_output_history.append(self.bias_output.copy())

            # Opcional: imprimir el error cada cierto número de épocas
            if (epoch+1) % 100 == 0:
                print(f"Dígitos - Época {epoch+1}/{self.epochs} - Error: {mse:.4f} -  Acc: {accuracy:.4f}")

    # Métodos de actualización de pesos (GD, Momentum, Adam)
    def update_weights_gd(self, X, hidden_output, delta_hidden, delta_output):
        # Actualizar pesos y biases de la capa de salida
        self.weights_hidden_output += self.learning_rate * hidden_output.T.dot(delta_output)
        self.bias_output += self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)

        # Actualizar pesos y biases de la capa oculta
        self.weights_input_hidden += self.learning_rate * X.T.dot(delta_hidden)
        self.bias_hidden += self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    def update_weights_momentum(self, X, hidden_output, delta_hidden, delta_output):
        # Actualizar pesos y biases de la capa de salida
        self.velocity_w_ho = self.momentum_gamma * self.velocity_w_ho + self.learning_rate * hidden_output.T.dot(delta_output)
        self.weights_hidden_output += self.velocity_w_ho
        self.velocity_b_o = self.momentum_gamma * self.velocity_b_o + self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)
        self.bias_output += self.velocity_b_o

        # Actualizar pesos y biases de la capa oculta
        self.velocity_w_ih = self.momentum_gamma * self.velocity_w_ih + self.learning_rate * X.T.dot(delta_hidden)
        self.weights_input_hidden += self.velocity_w_ih
        self.velocity_b_h = self.momentum_gamma * self.velocity_b_h + self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)
        self.bias_hidden += self.velocity_b_h

    def update_weights_adam(self, X, hidden_output, delta_hidden, delta_output):
        self.adam_t += 1

        # Parámetros de corrección de sesgo
        lr_t = self.learning_rate * (np.sqrt(1 - self.adam_beta2 ** self.adam_t) / (1 - self.adam_beta1 ** self.adam_t))

        # Gradientes para la capa de salida
        g_w_ho = hidden_output.T.dot(delta_output)
        g_b_o = np.sum(delta_output, axis=0, keepdims=True)

        # Actualización de momentos
        self.m_w_ho = self.adam_beta1 * self.m_w_ho + (1 - self.adam_beta1) * g_w_ho
        self.v_w_ho = self.adam_beta2 * self.v_w_ho + (1 - self.adam_beta2) * (g_w_ho ** 2)
        self.m_b_o = self.adam_beta1 * self.m_b_o + (1 - self.adam_beta1) * g_b_o
        self.v_b_o = self.adam_beta2 * self.v_b_o + (1 - self.adam_beta2) * (g_b_o ** 2)

        # Actualización de pesos y biases de la capa de salida
        m_hat_w_ho = self.m_w_ho / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_w_ho = self.v_w_ho / (1 - self.adam_beta2 ** self.adam_t)
        self.weights_hidden_output += lr_t * m_hat_w_ho / (np.sqrt(v_hat_w_ho) + self.adam_epsilon)

        m_hat_b_o = self.m_b_o / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_b_o = self.v_b_o / (1 - self.adam_beta2 ** self.adam_t)
        self.bias_output += lr_t * m_hat_b_o / (np.sqrt(v_hat_b_o) + self.adam_epsilon)

        # Gradientes para la capa oculta
        g_w_ih = X.T.dot(delta_hidden)
        g_b_h = np.sum(delta_hidden, axis=0, keepdims=True)

        # Actualización de momentos
        self.m_w_ih = self.adam_beta1 * self.m_w_ih + (1 - self.adam_beta1) * g_w_ih
        self.v_w_ih = self.adam_beta2 * self.v_w_ih + (1 - self.adam_beta2) * (g_w_ih ** 2)
        self.m_b_h = self.adam_beta1 * self.m_b_h + (1 - self.adam_beta1) * g_b_h
        self.v_b_h = self.adam_beta2 * self.v_b_h + (1 - self.adam_beta2) * (g_b_h ** 2)

        # Actualización de pesos y biases de la capa oculta
        m_hat_w_ih = self.m_w_ih / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_w_ih = self.v_w_ih / (1 - self.adam_beta2 ** self.adam_t)
        self.weights_input_hidden += lr_t * m_hat_w_ih / (np.sqrt(v_hat_w_ih) + self.adam_epsilon)

        m_hat_b_h = self.m_b_h / (1 - self.adam_beta1 ** self.adam_t)
        v_hat_b_h = self.v_b_h / (1 - self.adam_beta2 ** self.adam_t)
        self.bias_hidden += lr_t * m_hat_b_h / (np.sqrt(v_hat_b_h) + self.adam_epsilon)

    # Método para cargar pesos de una época específica
    def load_weights(self, epoch):
        self.weights_input_hidden = self.weights_input_hidden_history[epoch].copy()
        self.bias_hidden = self.bias_hidden_history[epoch].copy()
        self.weights_hidden_output = self.weights_hidden_output_history[epoch].copy()
        self.bias_output = self.bias_output_history[epoch].copy()