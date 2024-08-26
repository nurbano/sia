# Algoritmo Greedy
from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

import heapq
from .sokoban import Node
from .sokoban import State
from .sokoban import NodeDepth

def greedy_search(heuristic, stateObj, obstacles, storages):
    heuristic_dict={
        "manhattan": heuristic_man,
        "minimum_matching": heuristic_minimum_matching,
        "grouping": heuristic_grouping,
        "deadlock": heuristic_with_overestimation
    }
    heuristic_= heuristic_dict[heuristic]
    open_set = []
    startNode = Node(stateObj, None)
    start_h_score = heuristic_(stateObj, storages, obstacles)
    heapq.heappush(open_set, (start_h_score, 0, startNode))  # Solo la heurística h(n) se usa en Greedy
    taboo = set()
    counter = 1

    while open_set:
        _, _, currentNode = heapq.heappop(open_set)

        if currentNode.state.isGoalState(storages):
            return currentNode, taboo, len(open_set)

        taboo.add(currentNode.state)

        for childState in currentNode.state.possibleMoves(storages, obstacles):
            if childState in taboo or childState.isDeadLock(storages, obstacles):
                continue

            h_score = heuristic_(childState, storages, obstacles)
            childNode = Node(childState, currentNode)
            heapq.heappush(open_set, (h_score, counter, childNode))
            counter += 1

    return None, taboo, len(open_set)

# Algoritmo A*
def A_star(heuristic, stateObj, obstacles, storages):
    heuristic_dict={
        "manhattan": heuristic_man,
        "minimum_matching": heuristic_minimum_matching,
        "grouping": heuristic_grouping,
        "deadlock": heuristic_with_overestimation
    }
    heuristic_= heuristic_dict[heuristic]
    open_set = []
    startNode = Node(stateObj, None)
    start_f_score = heuristic_(stateObj, storages, obstacles)
    heapq.heappush(open_set, (start_f_score, 0, startNode))  # El segundo valor es un contador único para evitar comparaciones directas
    taboo = set()
    g_scores = {startNode.state: 0}
    counter = 1

    while open_set:
        _, _, currentNode = heapq.heappop(open_set)

        if currentNode.state.isGoalState(storages):
            return currentNode, taboo, len(open_set)

        taboo.add(currentNode.state)

        for childState in currentNode.state.possibleMoves(storages, obstacles):
            if childState in taboo or childState.isDeadLock(storages, obstacles):
                continue

            tentative_g_score = g_scores[currentNode.state] + 1

            if childState not in g_scores or tentative_g_score < g_scores[childState]:
                g_scores[childState] = tentative_g_score
                f_score = tentative_g_score + heuristic_(childState, storages, obstacles)
                childNode = Node(childState, currentNode)
                heapq.heappush(open_set, (f_score, counter, childNode))
                counter += 1

    return None, taboo, len(open_set)

# Resto de los algoritmos (DFS, BFS, IDS) y función readBoard como en la celda original...

def DFS(heuristic, stateObj, obstacles, storages):
    startNode = Node(stateObj, None)
    tree = deque([startNode])
    taboo = set()

    while tree:
        currentNode = tree.pop()
        taboo.add(currentNode.state)

        if currentNode.state.isDeadLock(storages, obstacles):
            continue
        if currentNode.state.isGoalState(storages):
            return currentNode, taboo, len(tree)

        validMovesStates = currentNode.state.possibleMoves(storages, obstacles)
        validMovesStates.reverse()
        for childState in validMovesStates:
            childNode = Node(childState, currentNode)
            if childNode.state in taboo:
                continue
            tree.append(childNode)
    return None, taboo, len(tree)

def BFS(heuristic, stateObj, obstacles, storages):
    startNode = Node(stateObj, None)
    tree = deque([startNode])
    taboo = set()

    while tree:
        currentNode = tree.popleft()
        taboo.add(currentNode.state)

        if currentNode.state.isDeadLock(storages, obstacles):
            continue
        if currentNode.state.isGoalState(storages):
            return currentNode, taboo, len(tree)

        validMovesStates = currentNode.state.possibleMoves(storages, obstacles)
        for childState in validMovesStates:
            childNode = Node(childState, currentNode)
            if childNode.state in taboo:
                continue
            tree.append(childNode)
    return None, taboo, len(tree)

def IDS(heuristic, stateObj, obstacles, storages, limit=10, increase=3):
    startNode = NodeDepth(stateObj, None, 0)
    limitTree = deque([startNode])
    taboo = set()
    limit -= increase

    while limitTree:
        tree = limitTree.copy()
        limitTree = deque()
        limit += increase

        while tree:
            currentNode = tree.pop()
            taboo.add(currentNode.state)

            if currentNode.state.isDeadLock(storages, obstacles):
                continue
            if currentNode.depth >= limit:
                limitTree.append(currentNode)
                continue
            if currentNode.state.isGoalState(storages):
                return currentNode, taboo, len(tree)

            validMovesStates = currentNode.state.possibleMoves(storages, obstacles)  # Corrección realizada aquí
            validMovesStates.reverse()
            for childState in validMovesStates:
                childNode = NodeDepth(childState, currentNode, currentNode.depth + 1)
                if childNode.state in taboo:
                    continue
                tree.append(childNode)
    return None, taboo, len(tree)

# Función heurística de Manhattan
def heuristic_man(state, storages, obstacles):
    total_dist = 0
    for box in state.boxes:
        min_dist = min(manhattan_dist(box, storage) for storage in storages)
        total_dist += min_dist
    return total_dist

# Función para calcular la distancia de Manhattan entre dos puntos
def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# Heurística de Agrupación de Almacenes sin Penalización por Posición Subóptima
def heuristic_grouping(state, storages, obstacles):
    total_dist = 0
    grouping_bonus = 0
    penalty_threshold = 2  # Umbral para considerar almacenes cercanos

    for box in state.boxes:
        distances_to_storages = [manhattan_dist(box, storage) for storage in storages]
        min_dist = min(distances_to_storages)
        total_dist += min_dist

        # Bonificación por caja con acceso cercano a múltiples objetivos
        close_storages = [dist for dist in distances_to_storages if dist <= penalty_threshold]
        grouping_bonus += len(close_storages) * 0.5  # Bonificación por almacenes cercanos

    return total_dist - grouping_bonus


# Función heurística utilizando Emparejamiento Mínimo (Minimum Matching)
def heuristic_minimum_matching(state, storages, obstacles):
    # Crear una matriz de costos (distancias de Manhattan) de cajas a almacenes
    cost_matrix = []
    for box in state.boxes:
        cost_row = [manhattan_dist(box, storage) for storage in storages]
        cost_matrix.append(cost_row)

    # Convertir la lista en una matriz NumPy
    cost_matrix = np.array(cost_matrix)

    # Resolver el problema de emparejamiento mínimo usando el algoritmo de asignación húngaro
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # La heurística será la suma de los costos de las asignaciones óptimas
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return total_cost

# Función heurística con sobreestimación: suma de distancias de Manhattan y penalización por patrones problemáticos
def heuristic_with_overestimation(state, storages, obstacles):
    total_dist = 0
    penalty = 0
    for box in state.boxes:
        # Calcular la distancia mínima desde la caja hasta cualquier objetivo
        min_dist = min(manhattan_dist(box, storage) for storage in storages)
        total_dist += min_dist

        # Añadir penalizaciones basadas en patrones problemáticos
        x, y = box
        if ((x-1, y) in obstacles or (x+1, y) in obstacles) and ((x, y-1) in obstacles or (x, y+1) in obstacles):
            # Penalización por caja en esquina sin ser objetivo
            if box not in storages:
                penalty += 20  # Ajusta este valor según la severidad del patrón antes 10

        if ((x-1, y) in obstacles and (x+1, y) in obstacles) or ((x, y-1) in obstacles and (x, y+1) in obstacles):
            # Penalización por caja atrapada entre paredes sin ser objetivo
            if box not in storages:
                penalty += 0  # Ajusta este valor según la severidad del patrón, antes 5

        # Puedes agregar más patrones problemáticos y penalizaciones aquí según lo necesites

    return total_dist + penalty