import numpy as np
import random
import math

# def seleccionar_padres(poblacion, fitness, A, porcentaje_padres):
#     num_padres = int(porcentaje_padres * len(poblacion))  # Número de padres a seleccionar
#     num_metodo1 = int(A * num_padres)
#     num_metodo2 = num_padres - num_metodo1

#     # Método 1: Selección por ruleta
#     padres_ruleta = seleccionar_padres_ruleta(poblacion, fitness, num_metodo1)

#     # Método 2: Selección por torneo
#     padres_torneo = seleccionar_padres_torneo(poblacion, fitness, num_metodo2)

#     # Unir los padres seleccionados por ambos métodos
#     padres = np.concatenate((padres_ruleta, padres_torneo))

#     return padres
def seleccionar_padres(poblacion, funcion_aptitud, fitness_relativo, clase_personaje, total_puntos, K, A):
    """Selecciona padres utilizando dos métodos según el valor de A."""

    # Método 1 para la selección de padres
    num_padres_m1 = int(A * K)
    #print(clase_personaje)
    padres_m1 = seleccionar_metodo(poblacion, 'elite', funcion_aptitud, fitness_relativo, clase_personaje,num_padres_m1)

    # Método 2 para la selección de padres (el resto de los padres)
    num_padres_m2 = K - num_padres_m1
    padres_m2 = seleccionar_metodo(poblacion, 'ruleta', funcion_aptitud, fitness_relativo, clase_personaje, num_padres_m2)

    # Combinar los padres seleccionados por ambos métodos
    padres = padres_m1 + padres_m2

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

def seleccion_elite(poblacion, funcion_aptitud, clase_personaje, K):
    """Selecciona los mejores individuos (elitismo)."""
    poblacion_ordenada = sorted(poblacion, key=lambda ind: funcion_aptitud(clase_personaje, ind), reverse=True)
    seleccionados = poblacion_ordenada[:K]  # Seleccionar los mejores K individuos
    return seleccionados

def seleccion_por_ruleta(poblacion, fitness_relativo, K):
    """Selecciona individuos utilizando el método de ruleta tradicional con números aleatorios."""

    # Calcular el fitness acumulado
    fitness_acumulado = [sum(fitness_relativo[:i+1]) for i in range(len(fitness_relativo))]

    # Generar K números aleatorios en el rango [0, 1]
    r_values = [random.uniform(0, 1) for _ in range(K)]

    # Seleccionar individuos con base en los números aleatorios y el fitness acumulado
    seleccionados = []
    for r in r_values:
        for i, q_i in enumerate(fitness_acumulado):
            if r <= q_i:
                seleccionados.append(poblacion[i])
                break

    return seleccionados

def seleccion_universal(poblacion, fitness_relativo, K):
    """Selecciona individuos utilizando el método Universal (equidistante en aptitud relativa)."""
    acumulado_fitness = [sum(fitness_relativo[:i+1]) for i in range(len(fitness_relativo))]
    r_0 = random.uniform(0, 1/K)
    seleccionados = []
    for j in range(K):
        r_j = r_0 + j / K
        for i, qi in enumerate(acumulado_fitness):
            if r_j <= qi:
                seleccionados.append(poblacion[i])
                break
    return seleccionados

def seleccion_por_ranking(poblacion, funcion_aptitud, clase_personaje, K):
    """Selección por ranking con ruleta."""
    poblacion_ordenada = sorted(poblacion, key=lambda ind: funcion_aptitud(clase_personaje, ind), reverse=True)
    fitness_ranking = [(len(poblacion) - rank) / len(poblacion) for rank in range(len(poblacion_ordenada))]
    seleccionados = seleccion_por_ruleta(poblacion_ordenada, fitness_ranking, K)
    return seleccionados

def seleccion_boltzmann(poblacion, funcion_aptitud, clase_personaje, K, T0, Tc, t, k=0.01):
    """Selecciona individuos utilizando el método Boltzmann con enfriamiento."""
    Tt = Tc + (T0 - Tc) * math.exp(-k * t)  # Enfriamiento exponencial
    fitness_boltzmann = [math.exp(funcion_aptitud(clase_personaje, ind) / Tt) for ind in poblacion]
    suma_fitness_boltzmann = sum(fitness_boltzmann)
    pseudo_aptitud = [f / suma_fitness_boltzmann for f in fitness_boltzmann]
    seleccionados = seleccion_por_ruleta(poblacion, pseudo_aptitud, K)
    return seleccionados

def seleccion_torneo_deterministico(poblacion, funcion_aptitud, clase_personaje, M, K):
    """Selecciona individuos utilizando el método de torneo determinístico."""
    seleccionados = []
    for _ in range(K):
        torneo = random.sample(poblacion, M)
        mejor = max(torneo, key=lambda ind: funcion_aptitud(clase_personaje, ind))
        seleccionados.append(mejor)
    return seleccionados

def seleccion_torneo_probabilistico(poblacion, funcion_aptitud, clase_personaje, threshold, K):
    """Selecciona individuos utilizando el método de torneo probabilístico."""
    seleccionados = []
    for _ in range(K):
        ind1, ind2 = random.sample(poblacion, 2)
        r = random.uniform(0, 1)
        if r < threshold:
            mejor = ind1 if funcion_aptitud(clase_personaje, ind1) > funcion_aptitud(clase_personaje, ind2) else ind2
        else:
            peor = ind1 if funcion_aptitud(clase_personaje, ind1) < funcion_aptitud(clase_personaje, ind2) else ind2
        seleccionados.append(mejor if r < threshold else peor)
    return seleccionados

def seleccionar_metodo(poblacion, metodo, funcion_aptitud, fitness_relativo, clase_personaje, K, **kwargs):
    """
    Selecciona individuos utilizando el método de selección especificado.

    Args:
    - poblacion: la población de cromosomas.
    - metodo: el método de selección ('elite', 'ruleta', 'universal', 'ranking', 'boltzmann', 'torneo_deterministico', 'torneo_probabilistico').
    - funcion_aptitud: la función para calcular la aptitud.
    - clase_personaje: la clase del personaje.
    - K: el número de individuos a seleccionar.
    - kwargs: argumentos adicionales específicos para algunos métodos (T0, Tc, t, M, threshold).

    Returns:
    - Lista de individuos seleccionados.
    """
    if metodo == 'elite':
        return seleccion_elite(poblacion, funcion_aptitud, clase_personaje, K)
    elif metodo == 'ruleta':
        #fitness_relativo = calcular_fitness_relativo(poblacion, funcion_aptitud, clase_personaje)
        return seleccion_por_ruleta(poblacion, fitness_relativo, K)
    elif metodo == 'universal':
        #fitness_relativo = calcular_fitness_relativo(poblacion, funcion_aptitud, clase_personaje)
        return seleccion_universal(poblacion, fitness_relativo, K)
    elif metodo == 'ranking':
        return seleccion_por_ranking(poblacion, funcion_aptitud, clase_personaje, K)
    elif metodo == 'boltzmann':
        return seleccion_boltzmann(poblacion, funcion_aptitud, clase_personaje, K, kwargs['T0'], kwargs['Tc'], kwargs['t'], kwargs.get('k', 0.01))
    elif metodo == 'torneo_deterministico':
        return seleccion_torneo_deterministico(poblacion, funcion_aptitud, clase_personaje, kwargs['M'], K)
    elif metodo == 'torneo_probabilistico':
        return seleccion_torneo_probabilistico(poblacion, funcion_aptitud, clase_personaje, kwargs['threshold'], K)
    else:
        raise ValueError("Método de selección no válido")
    
def mostrar_seleccionados(metodo, seleccionados, funcion_aptitud, clase_personaje):
    print(f"Seleccionados {metodo}:")
    for i, cromosoma in enumerate(seleccionados):
        aptitud = funcion_aptitud(clase_personaje, cromosoma)
        print(f"Individuo {i+1}: {cromosoma}, Aptitud: {aptitud:.4f}")
    print("\n")  # Para separar cada método con un salto de línea

#Todo Posibilidad de elegir método
def seleccionar_nueva_generacion(poblacion, hijos, funcion_aptitud, fitness_relativo, metodo_seleccion_1, metodo_seleccion_2, clase_personaje, total_puntos, N, B, **kwargs):
    atributos = ["fuerza", "destreza", "inteligencia", "vigor", "constitución"]

    num_reemplazo_m1 = int(B * N)
    seleccionados_m1 = seleccionar_metodo(poblacion + hijos, metodo_seleccion_1, funcion_aptitud, fitness_relativo,clase_personaje, num_reemplazo_m1, **kwargs )

    num_reemplazo_m2 = N - num_reemplazo_m1
    seleccionados_m2 = seleccionar_metodo(poblacion + hijos, metodo_seleccion_2, funcion_aptitud, fitness_relativo,clase_personaje, num_reemplazo_m2, **kwargs)

    nueva_generacion = seleccionados_m1 + seleccionados_m2
    #print(len(nueva_generacion))
    
    for cromosoma in nueva_generacion:
        #print(cromosoma, atributos, total_puntos)
        cromosoma = ajustar_suma_final(cromosoma, atributos, total_puntos)
        
        for attr in atributos:
            cromosoma[attr] = round(cromosoma[attr])  # Asegurar que los valores sean enteros

    return nueva_generacion

def ajustar_suma_final(cromosoma, atributos_sum, total_puntos):
    # Obtener los valores ajustados sin redondeo
    suma_actual = sum(cromosoma[attr] for attr in atributos_sum)
    #print("Suma Actual",suma_actual)
    factor_ajuste = total_puntos / suma_actual
    #print("Factor de ajuste", factor_ajuste)
    for attr in atributos_sum:
        cromosoma[attr] *= factor_ajuste
    #print(cromosoma)
    # Aplicar floor a los atributos y calcular la suma parcial
    suma_parcial = 0
    partes_enteras = {}
    partes_decimales = {}
    for attr in atributos_sum:
        partes_enteras[attr] = int(cromosoma[attr])  # floor
        partes_decimales[attr] = cromosoma[attr] - partes_enteras[attr]
        suma_parcial += partes_enteras[attr]
    #print("Suma parcial: ", suma_parcial)
    # Calcular la diferencia y ordenar los atributos por su parte decimal descendente
    diferencia = int(total_puntos - suma_parcial)
    atributos_ordenados = sorted(partes_decimales.items(), key=lambda x: x[1], reverse=True)
     # Distribuir los puntos restantes
    for i in range(diferencia):
        attr = atributos_ordenados[i % len(atributos_ordenados)][0]
        partes_enteras[attr] += 1

    # Actualizar el cromosoma con los valores finales
    for attr in atributos_sum:
        cromosoma[attr] = partes_enteras[attr]

    return cromosoma