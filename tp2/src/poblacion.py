import random
import math
import numpy as np
#random.seed(42)

# Definir los atributos y los límites para la altura
atributos = ["fuerza", "destreza", "inteligencia", "vigor", "constitución"]
altura_min = 1.3
altura_max = 2.0

def asignar_puntos_simultaneos(total_puntos, atributos):
    proporciones = [random.random() for _ in range(len(atributos))]
    suma_proporciones = sum(proporciones)
    proporciones_normalizadas = [p / suma_proporciones for p in proporciones]

    distribucion_atributos = {}
    puntos_asignados = 0
    for i, atributo in enumerate(atributos):
        puntos_para_atributo = round(total_puntos * proporciones_normalizadas[i])
        distribucion_atributos[atributo] = puntos_para_atributo
        puntos_asignados += puntos_para_atributo
            # Ajustar los puntos restantes si es necesario (si hay diferencia por el redondeo)
    diferencia = total_puntos - puntos_asignados
    if diferencia != 0:
        atributo_ajustado = random.choice(atributos)
        distribucion_atributos[atributo_ajustado] += diferencia

    return distribucion_atributos

def crear_cromosoma(total_puntos):
    distribucion_atributos = asignar_puntos_simultaneos(total_puntos, atributos)
    altura = round(random.uniform(altura_min, altura_max), 2)
    distribucion_atributos["altura"] = altura
    return distribucion_atributos

def crear_poblacion_inicial(N, total_puntos):
    return [crear_cromosoma(total_puntos) for _ in range(N)]