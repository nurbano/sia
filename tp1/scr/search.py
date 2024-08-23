# Algoritmo Greedy
from collections import deque
import heapq
from .sokoban import Node
from .sokoban import State
from .sokoban import NodeDepth

def greedy_search(stateObj, obstacles, storages):
    open_set = []
    startNode = Node(stateObj, None)
    start_h_score = heuristic(stateObj, storages)
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

            h_score = heuristic(childState, storages)
            childNode = Node(childState, currentNode)
            heapq.heappush(open_set, (h_score, counter, childNode))
            counter += 1

    return None, taboo, len(open_set)

# Algoritmo A*
def A_star(stateObj, obstacles, storages):
    open_set = []
    startNode = Node(stateObj, None)
    start_f_score = heuristic(stateObj, storages)
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
                f_score = tentative_g_score + heuristic(childState, storages)
                childNode = Node(childState, currentNode)
                heapq.heappush(open_set, (f_score, counter, childNode))
                counter += 1

    return None, taboo, len(open_set)

# Resto de los algoritmos (DFS, BFS, IDS) y función readBoard como en la celda original...

def DFS(stateObj, obstacles, storages):
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

def BFS(stateObj, obstacles, storages):
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

def IDS(stateObj, obstacles, storages, limit=10, increase=3):
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
def heuristic(state, storages):
    total_dist = 0
    for box in state.boxes:
        min_dist = min(manhattan_dist(box, storage) for storage in storages)
        total_dist += min_dist
    return total_dist

# Función para calcular la distancia de Manhattan entre dos puntos
def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
