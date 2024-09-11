import numpy as np

def fitness(clase, cromosoma):

    fuerza, destreza, inteligencia, vigor, constitucion, h= cromosoma
    fuerza_t= 100*np.tanh(0.01*fuerza)
    destreza_t= np.tanh(0.01*destreza)
    inteligencia_t= 0.6*np.tanh(0.01*inteligencia)
    vigor_t= np.tanh(0.01*vigor)
    constitucion= 100*np.tanh(0.01*constitucion)

    atm= 0.5-np.power(3*h-5, 4)+ np.power(3*h-5,2) +h/2
    dem= 2+np.power(3*h-5,4) - np.power(3*h-5,2) - h/2

    ataque= (destreza_t+inteligencia_t)*fuerza_t*atm
    defensa= (vigor_t+inteligencia_t)*constitucion*dem
    dict_fitness= {
        "guerrero": [0.6,0.4],
        "arquero": [0.9, 0.1],
        "guardian": [0.1, 0.9],
        "mago": [0.8,0.3]
    } 
    fitness= dict_fitness[clase][0]*ataque+dict_fitness[clase][1]*defensa
    return fitness