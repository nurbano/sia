import numpy as np


class Encoder:
    def __init__(self, input_size=35, hidden_size=18, latent_size=2):
        # Inicializar pesos y biases con Xavier Initialization
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.weights_input_hidden = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))

        limit = np.sqrt(6 / (hidden_size + latent_size))
        self.weights_hidden_latent = np.random.uniform(-limit, limit, (hidden_size, latent_size))
        self.bias_latent = np.zeros((1, latent_size))

        # Variables para Adam
        # Para weights_input_hidden
        self.m_w_ih = np.zeros_like(self.weights_input_hidden)
        self.v_w_ih = np.zeros_like(self.weights_input_hidden)
        self.m_b_h = np.zeros_like(self.bias_hidden)
        self.v_b_h = np.zeros_like(self.bias_hidden)

        # Para weights_hidden_latent
        self.m_w_hl = np.zeros_like(self.weights_hidden_latent)
        self.v_w_hl = np.zeros_like(self.weights_hidden_latent)
        self.m_b_l = np.zeros_like(self.bias_latent)
        self.v_b_l = np.zeros_like(self.bias_latent)

        # Hiperparámetros de Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-7

    def relu(self, x):
        return np.where(x > 0, x, 0)
    
    def relu_derivative(self,x):
        return np.where(x > 0, 1, 0)
    
    # Función Leaky ReLU
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    # Derivada de Leaky ReLU
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    # Propagación hacia adelante
    def forward(self, X):
        # Entrada a capa oculta
        self.X = X  # Almacenar para backward
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        #self.hidden_output = self.leaky_relu(self.hidden_input)
        # Capa oculta a espacio latente
        self.latent_input = np.dot(self.hidden_output, self.weights_hidden_latent) + self.bias_latent
        # Activación lineal en la capa latente
        self.latent_output = self.latent_input
        return self.latent_output

    # Retropropagación
    def backward(self, grad_latent):
        # Gradiente en la capa oculta a latente
        grad_weights_hidden_latent = np.dot(self.hidden_output.T, grad_latent)
        grad_bias_latent = np.sum(grad_latent, axis=0, keepdims=True)

        # Gradiente en la capa oculta
        # grad_hidden = np.dot(grad_latent, self.weights_hidden_latent.T) * self.leaky_relu_derivative(self.hidden_input)
        grad_hidden = np.dot(grad_latent, self.weights_hidden_latent.T) * self.relu_derivative(self.hidden_input)

        # Gradiente en la capa de entrada a oculta
        grad_weights_input_hidden = np.dot(self.X.T, grad_hidden)
        grad_bias_hidden = np.sum(grad_hidden, axis=0, keepdims=True)

        # Almacenar gradientes para la actualización de pesos
        self.grad_weights_input_hidden = grad_weights_input_hidden
        self.grad_bias_hidden = grad_bias_hidden
        self.grad_weights_hidden_latent = grad_weights_hidden_latent
        self.grad_bias_latent = grad_bias_latent

    # Actualizar pesos y biases usando Adam
    def update_parameters(self, learning_rate, t):
        # Actualización para weights_input_hidden
        self.m_w_ih = self.beta1 * self.m_w_ih + (1 - self.beta1) * self.grad_weights_input_hidden
        self.v_w_ih = self.beta2 * self.v_w_ih + (1 - self.beta2) * (self.grad_weights_input_hidden ** 2)
        m_hat_w_ih = self.m_w_ih / (1 - self.beta1 ** t)
        v_hat_w_ih = self.v_w_ih / (1 - self.beta2 ** t)
        self.weights_input_hidden -= learning_rate * m_hat_w_ih / (np.sqrt(v_hat_w_ih) + self.epsilon)

        # Actualización para bias_hidden
        self.m_b_h = self.beta1 * self.m_b_h + (1 - self.beta1) * self.grad_bias_hidden
        self.v_b_h = self.beta2 * self.v_b_h + (1 - self.beta2) * (self.grad_bias_hidden ** 2)
        m_hat_b_h = self.m_b_h / (1 - self.beta1 ** t)
        v_hat_b_h = self.v_b_h / (1 - self.beta2 ** t)
        self.bias_hidden -= learning_rate * m_hat_b_h / (np.sqrt(v_hat_b_h) + self.epsilon)

        # Actualización para weights_hidden_latent
        self.m_w_hl = self.beta1 * self.m_w_hl + (1 - self.beta1) * self.grad_weights_hidden_latent
        self.v_w_hl = self.beta2 * self.v_w_hl + (1 - self.beta2) * (self.grad_weights_hidden_latent ** 2)
        m_hat_w_hl = self.m_w_hl / (1 - self.beta1 ** t)
        v_hat_w_hl = self.v_w_hl / (1 - self.beta2 ** t)
        self.weights_hidden_latent -= learning_rate * m_hat_w_hl / (np.sqrt(v_hat_w_hl) + self.epsilon)

        # Actualización para bias_latent
        self.m_b_l = self.beta1 * self.m_b_l + (1 - self.beta1) * self.grad_bias_latent
        self.v_b_l = self.beta2 * self.v_b_l + (1 - self.beta2) * (self.grad_bias_latent ** 2)
        m_hat_b_l = self.m_b_l / (1 - self.beta1 ** t)
        v_hat_b_l = self.v_b_l / (1 - self.beta2 ** t)
        self.bias_latent -= learning_rate * m_hat_b_l / (np.sqrt(v_hat_b_l) + self.epsilon)

class Decoder:
    def __init__(self, latent_size=2, hidden_size=18, output_size=35):
        # Inicializar pesos y biases con Xavier Initialization
        limit = np.sqrt(6 / (latent_size + hidden_size))
        self.weights_latent_hidden = np.random.uniform(-limit, limit, (latent_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))

        limit = np.sqrt(6 / (hidden_size + output_size))
        self.weights_hidden_output = np.random.uniform(-limit, limit, (hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))

        # Variables para Adam
        # Para weights_latent_hidden
        self.m_w_lh = np.zeros_like(self.weights_latent_hidden)
        self.v_w_lh = np.zeros_like(self.weights_latent_hidden)
        self.m_b_h = np.zeros_like(self.bias_hidden)
        self.v_b_h = np.zeros_like(self.bias_hidden)

        # Para weights_hidden_output
        self.m_w_ho = np.zeros_like(self.weights_hidden_output)
        self.v_w_ho = np.zeros_like(self.weights_hidden_output)
        self.m_b_o = np.zeros_like(self.bias_output)
        self.v_b_o = np.zeros_like(self.bias_output)

        # Hiperparámetros de Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-7

    def relu(self, x):
        return np.where(x > 0, x, 0)
    
    def relu_derivative(self,x):
        return np.where(x > 0, 1, 0)
    
    # Función Leaky ReLUc
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    # Derivada de Leaky ReLU
    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    # Función Sigmoid con manejo de estabilidad numérica
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    # Derivada de Sigmoid
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    # Propagación hacia adelante
    def forward(self, Z):
        # Entrada latente a capa oculta
        self.Z = Z  # Almacenar para backward
        self.hidden_input = np.dot(Z, self.weights_latent_hidden) + self.bias_hidden
        #self.hidden_output = self.leaky_relu(self.hidden_input)
        self.hidden_output = self.relu(self.hidden_input)
        # Capa oculta a salida
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        return self.output

    # Retropropagación
    def backward(self, grad_output):
        # Gradiente en la capa de salida
        grad_output_input = grad_output * self.sigmoid_derivative(self.output_input)

        # Gradiente en pesos y biases de la capa oculta a salida
        grad_weights_hidden_output = np.dot(self.hidden_output.T, grad_output_input)
        grad_bias_output = np.sum(grad_output_input, axis=0, keepdims=True)

        # Gradiente en la capa oculta
        #grad_hidden = np.dot(grad_output_input, self.weights_hidden_output.T) * self.leaky_relu_derivative(self.hidden_input)
        grad_hidden = np.dot(grad_output_input, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_input)
        # Gradiente en pesos y biases de la capa latente a oculta
        grad_weights_latent_hidden = np.dot(self.Z.T, grad_hidden)
        grad_bias_hidden = np.sum(grad_hidden, axis=0, keepdims=True)

        # Almacenar gradientes para la actualización de pesos
        self.grad_weights_latent_hidden = grad_weights_latent_hidden
        self.grad_weights_latent_hidden = np.clip(self.grad_weights_latent_hidden, -1, 1)

        self.grad_bias_hidden = grad_bias_hidden
        self.grad_weights_hidden_output = grad_weights_hidden_output
        self.grad_weights_hidden_output = np.clip(self.grad_weights_hidden_output, -1, 1)
        self.grad_bias_output = grad_bias_output

        # Gradiente con respecto a la entrada latente (para el Encoder)
        self.grad_latent_input = np.dot(grad_hidden, self.weights_latent_hidden.T)
    def update_parameters(self, learning_rate, t):
        if t < 1:
            raise ValueError("El valor de t debe ser mayor o igual a 1 para evitar divisiones por cero.")

        beta1_t = 1 - self.beta1 ** t
        beta2_t = 1 - self.beta2 ** t

        # Lista de parámetros
        params_and_grads = [
            (self.weights_latent_hidden, self.grad_weights_latent_hidden, self.m_w_lh, self.v_w_lh),
            (self.bias_hidden, self.grad_bias_hidden, self.m_b_h, self.v_b_h),
            (self.weights_hidden_output, self.grad_weights_hidden_output, self.m_w_ho, self.v_w_ho),
            (self.bias_output, self.grad_bias_output, self.m_b_o, self.v_b_o),
        ]

        # Actualización genérica
        for param, grad, m, v in params_and_grads:
            # Clipping de gradientes para evitar explosión
            grad = np.clip(grad, -1, 1)

            # Actualización de momentos
            m[:] = self.beta1 * m + (1 - self.beta1) * grad
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # Corrección de sesgo
            m_hat = m / (beta1_t + self.epsilon)
            v_hat = v / (beta2_t + self.epsilon)

            # Actualización del parámetro
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    # Actualizar pesos y biases usando Adam
    # def update_parameters(self, learning_rate, t):
    #     # Actualización para weights_latent_hidden
    #     self.m_w_lh = self.beta1 * self.m_w_lh + (1 - self.beta1) * self.grad_weights_latent_hidden
    #     self.v_w_lh = self.beta2 * self.v_w_lh + (1 - self.beta2) * (self.grad_weights_latent_hidden ** 2)
    #     m_hat_w_lh = self.m_w_lh / (1 - self.beta1 ** t)
    #     v_hat_w_lh = self.v_w_lh / (1 - self.beta2 ** t)
    #     self.weights_latent_hidden -= learning_rate * m_hat_w_lh / (np.sqrt(v_hat_w_lh) + self.epsilon)

    #     # Actualización para bias_hidden
    #     self.m_b_h = self.beta1 * self.m_b_h + (1 - self.beta1) * self.grad_bias_hidden
    #     self.v_b_h = self.beta2 * self.v_b_h + (1 - self.beta2) * (self.grad_bias_hidden ** 2)
    #     m_hat_b_h = self.m_b_h / (1 - self.beta1 ** t)
    #     v_hat_b_h = self.v_b_h / (1 - self.beta2 ** t)
    #     self.bias_hidden -= learning_rate * m_hat_b_h / (np.sqrt(v_hat_b_h) + self.epsilon)

    #     # Actualización para weights_hidden_output
    #     self.m_w_ho = self.beta1 * self.m_w_ho + (1 - self.beta1) * self.grad_weights_hidden_output
    #     self.v_w_ho = self.beta2 * self.v_w_ho + (1 - self.beta2) * (self.grad_weights_hidden_output ** 2)
    #     m_hat_w_ho = self.m_w_ho / (1 - self.beta1 ** t)
    #     v_hat_w_ho = self.v_w_ho / (1 - self.beta2 ** t)
    #     self.weights_hidden_output -= learning_rate * m_hat_w_ho / (np.sqrt(v_hat_w_ho) + self.epsilon)

    #     # Actualización para bias_output
    #     self.m_b_o = self.beta1 * self.m_b_o + (1 - self.beta1) * self.grad_bias_output
    #     self.v_b_o = self.beta2 * self.v_b_o + (1 - self.beta2) * (self.grad_bias_output ** 2)
    #     m_hat_b_o = self.m_b_o / (1 - self.beta1 ** t)
    #     v_hat_b_o = self.v_b_o / (1 - self.beta2 ** t)
    #     self.bias_output -= learning_rate * m_hat_b_o / (np.sqrt(v_hat_b_o) + self.epsilon)

class Autoencoder:
    def __init__(self, input_size=35, hidden_size=18, latent_size=2, learning_rate=0.001):
        self.encoder = Encoder(input_size, hidden_size, latent_size)
        self.decoder = Decoder(latent_size, hidden_size, input_size)
        self.learning_rate = learning_rate
        self.t = 0  # Contador de pasos global para Adam
        self.epsilon= 1e-8
    def binary_cross_entropy_loss(self, y_true, y_pred):
        # Asegúrate de que y_pred esté en el rango (epsilon, 1-epsilon)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def binary_cross_entropy_derivative(self, y_true, y_pred):
        # Gradiente de la pérdida BCE con respecto a las predicciones
        
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))
    # Propagación hacia adelante
    def forward(self, X):
        self.latent_output = self.encoder.forward(X)
        self.reconstructed_output = self.decoder.forward(self.latent_output)
        return self.reconstructed_output

    # Retropropagación
    def backward(self, X, reconstructed_output):
        # Gradiente de la pérdida con respecto a la salida reconstruida
        #grad_output = reconstructed_output - X  # Para MSE
        grad_output= self.binary_cross_entropy_derivative(X, reconstructed_output)
        # reconstructed_output = np.clip(reconstructed_output, 1e-8, 1 - 1e-8)
        # grad_output = -(X / reconstructed_output - (1 - X) / (1 - reconstructed_output))
        # Retropropagación a través del decoder
        self.decoder.backward(grad_output)

        # Gradiente en el espacio latente
        grad_latent = self.decoder.grad_latent_input

        # Retropropagación a través del encoder
        self.encoder.backward(grad_latent)

        # Incrementar el contador de pasos
        self.t += 1

        # Actualizar parámetros
        self.encoder.update_parameters(self.learning_rate, self.t)
        self.decoder.update_parameters(self.learning_rate, self.t)


class Encoder_vae:
    def __init__(self, input_size=784, hidden_size=256, latent_size=2):
        # Inicializar pesos y biases con Xavier Initialization
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.weights_input_hidden = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))

        limit = np.sqrt(6 / (hidden_size + latent_size))
        self.weights_hidden_mean = np.random.uniform(-limit, limit, (hidden_size, latent_size))
        self.bias_mean = np.zeros((1, latent_size))

        self.weights_hidden_log_var = np.random.uniform(-limit, limit, (hidden_size, latent_size))
        self.bias_log_var = np.zeros((1, latent_size))

        # Variables para Adam optimizer (con prefijo enc_)
        # Para weights_input_hidden
        self.enc_m_w_ih = np.zeros_like(self.weights_input_hidden)
        self.enc_v_w_ih = np.zeros_like(self.weights_input_hidden)
        self.enc_m_b_h = np.zeros_like(self.bias_hidden)
        self.enc_v_b_h = np.zeros_like(self.bias_hidden)

        # Para weights_hidden_mean
        self.enc_m_w_hm = np.zeros_like(self.weights_hidden_mean)
        self.enc_v_w_hm = np.zeros_like(self.weights_hidden_mean)
        self.enc_m_b_m = np.zeros_like(self.bias_mean)
        self.enc_v_b_m = np.zeros_like(self.bias_mean)

        # Para weights_hidden_log_var
        self.enc_m_w_hv = np.zeros_like(self.weights_hidden_log_var)
        self.enc_v_w_hv = np.zeros_like(self.weights_hidden_log_var)
        self.enc_m_b_v = np.zeros_like(self.bias_log_var)
        self.enc_v_b_v = np.zeros_like(self.bias_log_var)

        # Hiperparámetros de Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8  # Escalar para estabilidad numérica en Adam

    # Funciones de activación y sus derivadas
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    # Propagación hacia adelante
    def forward(self, X):
        self.X = X
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.leaky_relu(self.hidden_input)
        self.z_mean = np.dot(self.hidden_output, self.weights_hidden_mean) + self.bias_mean
        self.z_log_var = np.dot(self.hidden_output, self.weights_hidden_log_var) + self.bias_log_var
        self.epsilon_sampling = np.random.normal(size=self.z_mean.shape)
        self.z = self.z_mean + np.exp(0.5 * self.z_log_var) * self.epsilon_sampling
        return self.z

    # Retropropagación
    def backward(self, grad_latent, epoch_frac):
        # Derivadas parciales
        dz_dz_mean = 1
        dz_dz_log_var = 0.5 * np.exp(0.5 * self.z_log_var) * self.epsilon_sampling

        # Gradiente del término de reconstrucción
        grad_reconstruction_z_mean = grad_latent * dz_dz_mean
        grad_reconstruction_z_log_var = grad_latent * dz_dz_log_var

        # Gradiente del término KL
        grad_kl_z_mean = self.z_mean
        grad_kl_z_log_var = 0.5 * (np.exp(self.z_log_var) - 1)*min(1.0, epoch_frac)

        # Gradiente total
        grad_z_mean = grad_reconstruction_z_mean + grad_kl_z_mean
        grad_z_log_var = grad_reconstruction_z_log_var + grad_kl_z_log_var

        # Gradientes de los pesos y biases
        grad_weights_hidden_mean = np.dot(self.hidden_output.T, grad_z_mean)
        grad_bias_mean = np.sum(grad_z_mean, axis=0, keepdims=True)

        grad_weights_hidden_log_var = np.dot(self.hidden_output.T, grad_z_log_var)
        grad_bias_log_var = np.sum(grad_z_log_var, axis=0, keepdims=True)

        # Gradiente a través de la capa oculta
        grad_hidden_mean = np.dot(grad_z_mean, self.weights_hidden_mean.T)
        grad_hidden_log_var = np.dot(grad_z_log_var, self.weights_hidden_log_var.T)
        grad_hidden = (grad_hidden_mean + grad_hidden_log_var) * self.leaky_relu_derivative(self.hidden_input)

        # Gradientes de los pesos y biases de la capa de entrada a oculta
        grad_weights_input_hidden = np.dot(self.X.T, grad_hidden)
        grad_bias_hidden = np.sum(grad_hidden, axis=0, keepdims=True)

        # Almacenar gradientes para la actualización
        self.grad_weights_input_hidden = grad_weights_input_hidden
        self.grad_bias_hidden = grad_bias_hidden
        self.grad_weights_hidden_mean = grad_weights_hidden_mean
        self.grad_bias_mean = grad_bias_mean
        self.grad_weights_hidden_log_var = grad_weights_hidden_log_var
        self.grad_bias_log_var = grad_bias_log_var

    # Actualización de parámetros usando Adam
    def update_parameters(self, learning_rate, t):
        # Actualización para weights_input_hidden
        self.enc_m_w_ih = self.beta1 * self.enc_m_w_ih + (1 - self.beta1) * self.grad_weights_input_hidden
        self.enc_v_w_ih = self.beta2 * self.enc_v_w_ih + (1 - self.beta2) * (self.grad_weights_input_hidden ** 2)
        m_hat_w_ih = self.enc_m_w_ih / (1 - self.beta1 ** t)
        v_hat_w_ih = self.enc_v_w_ih / (1 - self.beta2 ** t)
        self.weights_input_hidden -= learning_rate * m_hat_w_ih / (np.sqrt(v_hat_w_ih) + self.epsilon)

        # Actualización para bias_hidden
        self.enc_m_b_h = self.beta1 * self.enc_m_b_h + (1 - self.beta1) * self.grad_bias_hidden
        self.enc_v_b_h = self.beta2 * self.enc_v_b_h + (1 - self.beta2) * (self.grad_bias_hidden ** 2)
        m_hat_b_h = self.enc_m_b_h / (1 - self.beta1 ** t)
        v_hat_b_h = self.enc_v_b_h / (1 - self.beta2 ** t)
        self.bias_hidden -= learning_rate * m_hat_b_h / (np.sqrt(v_hat_b_h) + self.epsilon)

        # Actualización para weights_hidden_mean
        self.enc_m_w_hm = self.beta1 * self.enc_m_w_hm + (1 - self.beta1) * self.grad_weights_hidden_mean
        self.enc_v_w_hm = self.beta2 * self.enc_v_w_hm + (1 - self.beta2) * (self.grad_weights_hidden_mean ** 2)
        m_hat_w_hm = self.enc_m_w_hm / (1 - self.beta1 ** t)
        v_hat_w_hm = self.enc_v_w_hm / (1 - self.beta2 ** t)
        self.weights_hidden_mean -= learning_rate * m_hat_w_hm / (np.sqrt(v_hat_w_hm) + self.epsilon)

        # Actualización para bias_mean
        self.enc_m_b_m = self.beta1 * self.enc_m_b_m + (1 - self.beta1) * self.grad_bias_mean
        self.enc_v_b_m = self.beta2 * self.enc_v_b_m + (1 - self.beta2) * (self.grad_bias_mean ** 2)
        m_hat_b_m = self.enc_m_b_m / (1 - self.beta1 ** t)
        v_hat_b_m = self.enc_v_b_m / (1 - self.beta2 ** t)
        self.bias_mean -= learning_rate * m_hat_b_m / (np.sqrt(v_hat_b_m) + self.epsilon)

        # Actualización para weights_hidden_log_var
        self.enc_m_w_hv = self.beta1 * self.enc_m_w_hv + (1 - self.beta1) * self.grad_weights_hidden_log_var
        self.enc_v_w_hv = self.beta2 * self.enc_v_w_hv + (1 - self.beta2) * (self.grad_weights_hidden_log_var ** 2)
        m_hat_w_hv = self.enc_m_w_hv / (1 - self.beta1 ** t)
        v_hat_w_hv = self.enc_v_w_hv / (1 - self.beta2 ** t)
        self.weights_hidden_log_var -= learning_rate * m_hat_w_hv / (np.sqrt(v_hat_w_hv) + self.epsilon)

        # Actualización para bias_log_var
        self.enc_m_b_v = self.beta1 * self.enc_m_b_v + (1 - self.beta1) * self.grad_bias_log_var
        self.enc_v_b_v = self.beta2 * self.enc_v_b_v + (1 - self.beta2) * (self.grad_bias_log_var ** 2)
        m_hat_b_v = self.enc_m_b_v / (1 - self.beta1 ** t)
        v_hat_b_v = self.enc_v_b_v / (1 - self.beta2 ** t)
        self.bias_log_var -= learning_rate * m_hat_b_v / (np.sqrt(v_hat_b_v) + self.epsilon)

class Decoder_vae:
    def __init__(self, latent_size=2, hidden_size=256, output_size=784):
        # Inicializar pesos y biases con Xavier Initialization
        limit = np.sqrt(6 / (latent_size + hidden_size))
        self.weights_latent_hidden = np.random.uniform(-limit, limit, (latent_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))

        limit = np.sqrt(6 / (hidden_size + output_size))
        self.weights_hidden_output = np.random.uniform(-limit, limit, (hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))

        # Variables para Adam optimizer (con prefijo dec_)
        # Para weights_latent_hidden
        self.dec_m_w_lh = np.zeros_like(self.weights_latent_hidden)
        self.dec_v_w_lh = np.zeros_like(self.weights_latent_hidden)
        self.dec_m_b_h = np.zeros_like(self.bias_hidden)
        self.dec_v_b_h = np.zeros_like(self.bias_hidden)

        # Para weights_hidden_output
        self.dec_m_w_ho = np.zeros_like(self.weights_hidden_output)
        self.dec_v_w_ho = np.zeros_like(self.weights_hidden_output)
        self.dec_m_b_o = np.zeros_like(self.bias_output)
        self.dec_v_b_o = np.zeros_like(self.bias_output)

        # Hiperparámetros de Adam
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    # Funciones de activación y sus derivadas
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    # Propagación hacia adelante
    def forward(self, Z):
        self.Z = Z
        self.hidden_input = np.dot(Z, self.weights_latent_hidden) + self.bias_hidden
        self.hidden_output = self.leaky_relu(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)
        return self.output

    # Retropropagación
    def backward(self, grad_output):
        grad_output_input = grad_output * self.sigmoid_derivative(self.output_input)
        grad_weights_hidden_output = np.dot(self.hidden_output.T, grad_output_input)
        grad_bias_output = np.sum(grad_output_input, axis=0, keepdims=True)

        grad_hidden = np.dot(grad_output_input, self.weights_hidden_output.T) * self.leaky_relu_derivative(self.hidden_input)
        grad_weights_latent_hidden = np.dot(self.Z.T, grad_hidden)
        grad_bias_hidden = np.sum(grad_hidden, axis=0, keepdims=True)

        # Almacenar gradientes para la actualización
        self.grad_weights_latent_hidden = grad_weights_latent_hidden
        self.grad_bias_hidden = grad_bias_hidden
        self.grad_weights_hidden_output = grad_weights_hidden_output
        self.grad_bias_output = grad_bias_output

        # Gradiente con respecto a Z
        self.grad_latent_input = np.dot(grad_hidden, self.weights_latent_hidden.T)

    # Actualización de parámetros usando Adam (con prefijo dec_)
    def update_parameters(self, learning_rate, t):
        # Actualización para weights_latent_hidden
        self.dec_m_w_lh = self.beta1 * self.dec_m_w_lh + (1 - self.beta1) * self.grad_weights_latent_hidden
        self.dec_v_w_lh = self.beta2 * self.dec_v_w_lh + (1 - self.beta2) * (self.grad_weights_latent_hidden ** 2)
        m_hat_w_lh = self.dec_m_w_lh / (1 - self.beta1 ** t)
        v_hat_w_lh = self.dec_v_w_lh / (1 - self.beta2 ** t)
        self.weights_latent_hidden -= learning_rate * m_hat_w_lh / (np.sqrt(v_hat_w_lh) + self.epsilon)

        # Actualización para bias_hidden
        self.dec_m_b_h = self.beta1 * self.dec_m_b_h + (1 - self.beta1) * self.grad_bias_hidden
        self.dec_v_b_h = self.beta2 * self.dec_v_b_h + (1 - self.beta2) * (self.grad_bias_hidden ** 2)
        m_hat_b_h = self.dec_m_b_h / (1 - self.beta1 ** t)
        v_hat_b_h = self.dec_v_b_h / (1 - self.beta2 ** t)
        self.bias_hidden -= learning_rate * m_hat_b_h / (np.sqrt(v_hat_b_h) + self.epsilon)

        # Actualización para weights_hidden_output
        self.dec_m_w_ho = self.beta1 * self.dec_m_w_ho + (1 - self.beta1) * self.grad_weights_hidden_output
        self.dec_v_w_ho = self.beta2 * self.dec_v_w_ho + (1 - self.beta2) * (self.grad_weights_hidden_output ** 2)
        m_hat_w_ho = self.dec_m_w_ho / (1 - self.beta1 ** t)
        v_hat_w_ho = self.dec_v_w_ho / (1 - self.beta2 ** t)
        self.weights_hidden_output -= learning_rate * m_hat_w_ho / (np.sqrt(v_hat_w_ho) + self.epsilon)

        # Actualización para bias_output
        self.dec_m_b_o = self.beta1 * self.dec_m_b_o + (1 - self.beta1) * self.grad_bias_output
        self.dec_v_b_o = self.beta2 * self.dec_v_b_o + (1 - self.beta2) * (self.grad_bias_output ** 2)
        m_hat_b_o = self.dec_m_b_o / (1 - self.beta1 ** t)
        v_hat_b_o = self.dec_v_b_o / (1 - self.beta2 ** t)
        self.bias_output -= learning_rate * m_hat_b_o / (np.sqrt(v_hat_b_o) + self.epsilon)



class Autoencoder_vae:
    def __init__(self, input_size=28*28, hidden_size=256, latent_size=2, learning_rate=0.001):
        self.encoder = Encoder_vae(input_size, hidden_size, latent_size)
        self.decoder = Decoder_vae(latent_size, hidden_size, input_size)
        self.learning_rate = learning_rate
        self.t = 0  # Contador de pasos global para Adam

    # Propagación hacia adelante
    def forward(self, X):
        self.z = self.encoder.forward(X)
        self.reconstructed_output = self.decoder.forward(self.z)
        return self.reconstructed_output

    # Retropropagación
    def backward(self, X, reconstructed_output, epoch_frac):
        # Gradiente de la pérdida de reconstrucción con respecto a la salida
        epsilon = 1e-8
        reconstructed_output = np.clip(reconstructed_output, epsilon, 1 - epsilon)
        grad_reconstruction = -(X / reconstructed_output - (1 - X) / (1 - reconstructed_output))

        # Retropropagación a través del decoder
        self.decoder.backward(grad_reconstruction)

        # Gradiente con respecto a z (salida del encoder)
        grad_z = self.decoder.grad_latent_input

        # Retropropagación a través del encoder
        self.encoder.backward(grad_z, epoch_frac)

        # Incrementar el contador de pasos
        self.t += 1

        # Actualizar parámetros
        self.encoder.update_parameters(self.learning_rate, self.t)
        self.decoder.update_parameters(self.learning_rate, self.t)

