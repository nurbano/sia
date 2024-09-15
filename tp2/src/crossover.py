import numpy as np
import random

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

def crossover_aritmetico_reparacion(padre1, padre2, total_puntos):
    alpha = random.uniform(0, 1)
    hijo1 = {}
    hijo2 = {}
    atributos_sum = ["fuerza", "destreza", "inteligencia", "vigor", "constitución"]

    for attr in atributos_sum:
        hijo1[attr] = alpha * padre1[attr] + (1 - alpha) * padre2[attr]
        hijo2[attr] = (1 - alpha) * padre1[attr] + alpha * padre2[attr]

    hijo1 = ajustar_suma_final(hijo1, atributos_sum, total_puntos)
    hijo2 = ajustar_suma_final(hijo2, atributos_sum, total_puntos)

    for attr in atributos_sum:
        hijo1[attr] = max(0, round(hijo1[attr]))  # Usar round para asegurar enteros
        hijo2[attr] = max(0, round(hijo2[attr]))  # Usar round para asegurar enteros
    #Las alturas de cruzan
    hijo1["altura"] = padre2["altura"]
    hijo2["altura"] = padre1["altura"]

    return hijo1, hijo2

def ajustar_suma_final(cromosoma, atributos_sum, total_puntos):
    # Obtener los valores ajustados sin redondeo
    suma_actual = sum(cromosoma[attr] for attr in atributos_sum)
    factor_ajuste = total_puntos / suma_actual
    for attr in atributos_sum:
        cromosoma[attr] *= factor_ajuste

    # Aplicar floor a los atributos y calcular la suma parcial
    suma_parcial = 0
    partes_enteras = {}
    partes_decimales = {}
    for attr in atributos_sum:
        partes_enteras[attr] = int(cromosoma[attr])  # floor
        partes_decimales[attr] = cromosoma[attr] - partes_enteras[attr]
        suma_parcial += partes_enteras[attr]

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

def generar_hijos(padres, K, total_puntos):
    """Genera K hijos utilizando crossover aritmético."""
    hijos = []
    for i in range(0, len(padres), 2):
        padre1, padre2 = padres[i], padres[(i+1) % len(padres)]
        hijo1, hijo2 = crossover_aritmetico_reparacion(padre1, padre2, total_puntos)
        hijos.extend([hijo1, hijo2])
        if len(hijos) >= K:  # Limitar a K hijos
            break
    return hijos[:K]

def reparar_suma_atributos(hijo, atributos_sum, suma_total_original):
    """Repara los atributos de un hijo para que la suma sea igual a la suma total original."""
    suma_actual = sum(hijo[attr] for attr in atributos_sum)
    factor_ajuste = suma_total_original / suma_actual

    # Ajustar los atributos multiplicando por el factor de ajuste
    for attr in atributos_sum:
        hijo[attr] *= factor_ajuste

    return hijo