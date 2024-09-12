import numpy as np

def realizar_crossover(padres, total_puntos):
    num_pares = len(padres) // 2
    descendientes = []

    for i in range(num_pares):
        padre1 = padres[i]
        padre2 = padres[num_pares + i]

        # Realizar crossover aritmético y obtener dos hijos
        hijo1, hijo2 = crossover_aritmetico(padre1, padre2, total_puntos)

        descendientes.append(hijo1)
        descendientes.append(hijo2)

    return descendientes

def crossover_aritmetico(padre1, padre2, total_puntos):
    alpha = np.random.rand()  # Ponderación aleatoria entre 0 y 1
    hijo1 = {}
    hijo2 = {}

    # Realizar el crossover para cada atributo (excepto la altura)
    atributos = ["fuerza", "destreza", "inteligencia", "vigor", "constitución"]

    for atributo in atributos:
        hijo1[atributo] = round(alpha * padre1[atributo] + (1 - alpha) * padre2[atributo])
        hijo2[atributo] = round((1 - alpha) * padre1[atributo] + alpha * padre2[atributo])

    # Asignar la altura sin cambiarla
    hijo1["altura"] = padre1["altura"]
    hijo2["altura"] = padre2["altura"]

    # Reparar los hijos para que la suma de los atributos sea igual a total_puntos
    hijo1 = reparar_cromosoma(hijo1, total_puntos)
    hijo2 = reparar_cromosoma(hijo2, total_puntos)

    return hijo1, hijo2

def reparar_cromosoma(cromosoma, total_puntos):
    atributos = ["fuerza", "destreza", "inteligencia", "vigor", "constitución"]
    suma_atributos = sum([cromosoma[atributo] for atributo in atributos])

    # Si la suma es diferente a total_puntos, ajustamos proporcionalmente
    if suma_atributos != total_puntos:
        factor_ajuste = total_puntos / suma_atributos
        for atributo in atributos:
            cromosoma[atributo] = round(cromosoma[atributo] * factor_ajuste)

    # Si después de ajustar, la suma sigue sin coincidir (por redondeo), ajustar manualmente
    diferencia = total_puntos - sum([cromosoma[atributo] for atributo in atributos])

    if diferencia != 0:
        atributo_ajustado = np.random.choice(atributos)
        cromosoma[atributo_ajustado] += diferencia

    return cromosoma