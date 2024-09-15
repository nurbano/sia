from .seleccion import ajustar_suma_final
import random

def mutacion_redistribucion(cromosoma,  prob_mutacion, total_puntos):
    atributos = ["fuerza", "destreza", "inteligencia", "vigor", "constitución"]

    if random.random() < prob_mutacion:
        attr_mutado = random.choice(atributos)
        cambio = random.randint(-5, 5)

        nuevo_valor = max(0, cromosoma[attr_mutado] + cambio)
        diferencia = nuevo_valor - cromosoma[attr_mutado]
        cromosoma[attr_mutado] = nuevo_valor

        restantes = [attr for attr in atributos if attr != attr_mutado]
        for attr in restantes:
            cromosoma[attr] -= diferencia / len(restantes)
            cromosoma[attr] = max(0, cromosoma[attr])

        cromosoma = ajustar_suma_final(cromosoma, atributos, total_puntos)

    # Asegurar que todos los valores sean enteros
    for attr in atributos:
        cromosoma[attr] = round(cromosoma[attr])
    #print(cromosoma)
    return cromosoma

def aplicar_mutacion(generacion,  prob_mutacion, total_puntos):
    nueva_generacion_mutada = []
    for cromosoma in generacion:
        cromosoma_mutado = mutacion_redistribucion(cromosoma, prob_mutacion, total_puntos)
        nueva_generacion_mutada.append(cromosoma_mutado)
    return nueva_generacion_mutada