import numpy as np

def seleccionar_padres(poblacion, fitness, A, porcentaje_padres):
    num_padres = int(porcentaje_padres * len(poblacion))  # Número de padres a seleccionar
    num_metodo1 = int(A * num_padres)
    num_metodo2 = num_padres - num_metodo1

    # Método 1: Selección por ruleta
    padres_ruleta = seleccionar_padres_ruleta(poblacion, fitness, num_metodo1)

    # Método 2: Selección por torneo
    padres_torneo = seleccionar_padres_torneo(poblacion, fitness, num_metodo2)

    # Unir los padres seleccionados por ambos métodos
    padres = np.concatenate((padres_ruleta, padres_torneo))

    return padres

def seleccionar_padres_ruleta(poblacion, fitness, num_padres):
    total_fitness = np.sum(fitness)
    probs = fitness / total_fitness
    # print("Padres Ruleta")
    # print(len(probs))
    # print(len(poblacion))
    # print(num_padres)
    seleccionados = np.random.choice(len(poblacion), size=num_padres, p=probs)
    return np.array([poblacion[i] for i in seleccionados])

# Función para seleccionar padres mediante torneo
def seleccionar_padres_torneo(poblacion, fitness, num_padres, K=3):
    seleccionados = []
    for _ in range(num_padres):
        torneo = np.random.choice(len(poblacion), size=K)
        ganador = torneo[np.argmax(fitness[torneo])]
        seleccionados.append(poblacion[ganador])
    return np.array(seleccionados)

def realizar_reemplazo(poblacion_anterior, fitness_anterior, padres, fitness_padres, B, N):
    num_metodo3 = int(B * N)
    num_metodo4 = N - num_metodo3

    # Método 3: Reemplazo por elitismo
    reemplazo_elitismo = seleccionar_mejores(poblacion_anterior, fitness_anterior, num_metodo3)

    # Método 4: Reemplazo por ruleta
    reemplazo_ruleta = seleccionar_padres_ruleta(padres, fitness_padres, num_metodo3)

    # Unir las partes de la nueva población
    nueva_poblacion = np.concatenate((reemplazo_elitismo, reemplazo_ruleta))

    return nueva_poblacion

# Función auxiliar para seleccionar los mejores individuos (elitismo)
def seleccionar_mejores(poblacion, fitness, num_mejores):
    indices_mejores = np.argsort(fitness)[-num_mejores:]
    return np.array([poblacion[i] for i in indices_mejores])
