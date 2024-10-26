import numpy as np
import matplotlib.pyplot as plt

class Kohonen:
    def __init__(self, K, input_dim, learning_rate, radius, iterations, normalization_type="z-score"):
        self.K = K  # Tamaño del mapa (K x K)
        self.input_dim = input_dim  # Dimensión de los datos de entrada
        self.learning_rate = learning_rate  # Tasa de aprendizaje inicial
        self.initial_learning_rate = learning_rate  # Guardar tasa de aprendizaje inicial
        self.radius = radius  # Radio inicial del vecindario
        self.initial_radius = radius  # Guardar radio inicial
        self.iterations = iterations  # Número total de iteraciones
        self.normalization_type = normalization_type  # Tipo de normalización
        self.map_size = K * K  # Tamaño total del mapa
        self.weights = self.initialize_weights()  # Matriz de pesos
        self.clusters = {}  # Diccionario para almacenar los países por neurona ganadora
        self.quantization_errors = []  # Lista para almacenar el Quantization Error

    def initialize_weights(self):
        # Inicializa los pesos con valores aleatorios pequeños
        weights = np.random.rand(self.K, self.K, self.input_dim)
        return weights

    def normalize_data(self, data):
        # Normaliza los datos usando Z-Score
        data_normalized = data.copy()
        for column in data.columns[1:]:
            mean = data[column].mean()
            std = data[column].std()
            data_normalized[column] = (data[column] - mean) / std
        return data_normalized

    def find_winner_neuron(self, input_vector):
        # Encuentra la neurona ganadora usando distancia euclidiana
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        winner_index = np.unravel_index(np.argmin(distances), distances.shape)
        return winner_index

    def update_weights(self, input_vector, winner_neuron, current_iteration):
        # Actualiza los pesos de la neurona ganadora y sus vecinas
        learning_rate = self.adjust_learning_rate(current_iteration)
        radius = self.adjust_radius(current_iteration)

        for i in range(self.K):
            for j in range(self.K):
                neuron_position = np.array([i, j])
                winner_position = np.array(winner_neuron)
                distance = np.linalg.norm(neuron_position - winner_position)

                if distance <= radius:
                    influence = np.exp(-distance**2 / (2 * (radius**2)))
                    self.weights[i, j] += influence * learning_rate * (input_vector - self.weights[i, j])

    def adjust_learning_rate(self, current_iteration):
        # Ajusta la tasa de aprendizaje
        time_constant = self.iterations / np.log(self.initial_radius)
        learning_rate = self.initial_learning_rate * np.exp(-current_iteration / self.iterations)
        return learning_rate


    def adjust_radius(self, current_iteration):
        # Ajusta el radio del vecindario
        time_constant = self.iterations / np.log(self.initial_radius)
        radius = self.initial_radius * np.exp(-current_iteration / time_constant)
        return radius

    def calculate_quantization_error(self, data):
        # Calcula el Quantization Error
        error = 0
        for index, row in data.iterrows():
            input_vector = row[1:].values.astype(float)
            winner_neuron = self.find_winner_neuron(input_vector)
            error += np.linalg.norm(input_vector - self.weights[winner_neuron])
        quantization_error = error / len(data)
        self.quantization_errors.append(quantization_error)

    def train(self, data):
        # Ejecuta el entrenamiento de la red
        data_array = data.iloc[:, 1:].values.astype(float)
        for iteration in range(self.iterations):
            # Seleccionar una muestra aleatoria
            random_index = np.random.randint(0, len(data_array))
            input_vector = data_array[random_index]

            # Encontrar la neurona ganadora
            winner_neuron = self.find_winner_neuron(input_vector)

            # Actualizar los pesos
            self.update_weights(input_vector, winner_neuron, iteration)

            # Cada cierto número de iteraciones, calcular el Quantization Error
            if (iteration + 1) % 100 == 0:
                self.calculate_quantization_error(data)

        # Después del entrenamiento, asignar cada país a su neurona ganadora
        for index, row in data.iterrows():
            input_vector = row[1:].values.astype(float)
            winner_neuron = self.find_winner_neuron(input_vector)
            neuron_key = str(winner_neuron)
            if neuron_key in self.clusters:
                self.clusters[neuron_key].append(row['Country'])
            else:
                self.clusters[neuron_key] = [row['Country']]

    def plot_quantization_error(self):
        # Visualiza el Quantization Error a lo largo del entrenamiento
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(100, self.iterations + 1, 100), self.quantization_errors)
        plt.title('Quantization Error durante el Entrenamiento')
        plt.xlabel('Iteraciones')
        plt.ylabel('Quantization Error')
        plt.grid(True)
        plt.show()

    def plot_u_matrix(self):
        # Visualiza la U-Matrix
        import matplotlib.pyplot as plt
        u_matrix = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                neighbors = []
                if i > 0:
                    neighbors.append(self.weights[i - 1, j])
                if i < self.K - 1:
                    neighbors.append(self.weights[i + 1, j])
                if j > 0:
                    neighbors.append(self.weights[i, j - 1])
                if j < self.K - 1:
                    neighbors.append(self.weights[i, j + 1])
                distances = [np.linalg.norm(self.weights[i, j] - neighbor) for neighbor in neighbors]
                u_matrix[i, j] = np.mean(distances)

        plt.figure(figsize=(8, 8))
        plt.imshow(u_matrix, cmap='gray')
        plt.title('U-Matrix')
        plt.colorbar()
        plt.show()

    def plot_heatmap(self):
        # Visualiza un mapa de calor de frecuencias
        import seaborn as sns
        import matplotlib.pyplot as plt
        frequency_map = np.zeros((self.K, self.K))
        for neuron_key in self.clusters.keys():
            neuron_pos = eval(neuron_key)
            frequency_map[neuron_pos] = len(self.clusters[neuron_key])

        plt.figure(figsize=(8, 8))
        sns.heatmap(frequency_map, annot=True, fmt=".0f", cmap='viridis')
        plt.title('Mapa de Calor de Frecuencias')
        plt.show()

    def plot_2d_representation(self, data):
        # Visualiza una representación 2D de los datos y las neuronas ganadoras
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # Reducir la dimensionalidad de los datos y los pesos
        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(data.iloc[:, 1:].values.astype(float))
        weights_reshaped = self.weights.reshape(self.K * self.K, self.input_dim)
        weights_reduced = pca.transform(weights_reshaped)

        # Plotear los datos y las neuronas
        plt.figure(figsize=(10, 6))
        plt.scatter(data_reduced[:, 0], data_reduced[:, 1], label='Datos', alpha=0.5)
        plt.scatter(weights_reduced[:, 0], weights_reduced[:, 1], label='Neuronas', marker='x', color='red')
        plt.title('Representación 2D de Datos y Neuronas')
        plt.legend()
        plt.show()

    def show_clustered_countries(self):
        # Muestra los países agrupados por neurona ganadora
        print("Países agrupados por neurona ganadora:")
        for neuron_key, countries in self.clusters.items():
            print(f"Neurona {neuron_key}: {', '.join(countries)}")

class OjaNetwork:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Inicializar los pesos aleatoriamente entre 0 y 1
        self.weights = np.random.uniform(0, 1, input_size)
        self.weight_history = []

    def train(self, X):
        for epoch in range(self.epochs):
            for x in X:
                y = np.dot(self.weights, x)
                # Actualizar los pesos según la Regla de Oja
                self.weights += self.learning_rate * y * (x - y * self.weights)
            # Guardar la magnitud de los pesos para análisis posterior
            self.weight_history.append(np.linalg.norm(self.weights))

    def get_weights(self):
        return self.weights

    def plot_training(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(self.epochs), self.weight_history)
        plt.xlabel('Épocas')
        plt.ylabel('Magnitud de los Pesos')
        plt.title('Evolución de la Magnitud de los Pesos durante el Entrenamiento')
        plt.grid(True)
        print(self.weight_history[-1])
        plt.show()


class Hopfield:
    def __init__(self, input_dim, iter=10, convergence=2, seed=42):
        self.input_dim= input_dim
        self.neurons= input_dim
        self.iter= iter
        self.conv= convergence
        self.K= np.zeros(self.neurons,)
        np.random.seed(seed)
        
    def noise_with_k(self, pattern, noise_k): 
        indexes = np.random.choice(np.arange(self.input_dim), noise_k, False)
        print(indexes)
        for index in indexes:
            if pattern[index]==-1:
                pattern[index]=1
            else:
                pattern[index]=-1
        return pattern
     
    def calculate_weights(self, comb, dataset):
        for comb_letter in comb:
            for i, row in dataset.iterrows():
                letter = row['letter']
                if comb_letter==letter:
                    matrix = np.array(row[1:].astype(int))
                    self.K= np.column_stack((self.K,matrix))
        self.K= self.K[:,1:]
        self.W= (np.dot( self.K, self.K.T))/self.input_dim
        np.fill_diagonal(self.W, 0)
        return self.W
    def iterate_state(self, intial_state):
        ESTADOS=[]
        ENERGY=[]
        S_actual= intial_state
        acu= 0
        ESTADOS.append(S_actual) 
        for i in range(self.iter):
            
            S_nuevo= np.sign(np.dot(self.W, S_actual))
            energy= -0.5 * np.dot(S_actual, np.dot(self.W, S_actual))
            ENERGY.append(energy)
            if(np.array_equal(S_nuevo, S_actual)):
                acu+=1
                print("iter: ", i)
                print("iguales")
            if acu>self.conv:
                break
           
            S_actual= S_nuevo
            ESTADOS.append(S_nuevo)
        return [ESTADOS, ENERGY] 


