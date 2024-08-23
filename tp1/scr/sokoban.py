# Implementación de Clases para la Gestión de Estados y Nodos en Problemas de Búsqueda
class State():
    def __init__(self, player, boxes, movement):
        self.player = player
        self.boxes = boxes
        self.movement = movement

    def __eq__(self, otherState):
        return self.player == otherState.player and self.boxes == otherState.boxes

    def __hash__(self):
        return hash((self.player, tuple(self.boxes)))

    def possibleMoves(self, storages, obstacles):
        possibleMoves = []
        for directions in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            newPlayerPos = (self.player[0] + directions[0], self.player[1] + directions[1])
            if newPlayerPos in obstacles:
                continue
            newBoxesPos = dict(self.boxes)
            if newPlayerPos in self.boxes:
                newBoxPos = (newPlayerPos[0] + directions[0], newPlayerPos[1] + directions[1])
                if newBoxPos in obstacles or newBoxPos in self.boxes:
                    continue
                i = newBoxesPos.pop(newPlayerPos)
                newBoxesPos[newBoxPos] = i
            newState = State(newPlayerPos, newBoxesPos, directions)
            possibleMoves.append(newState)
        return possibleMoves

    def isDeadLock(self, storages, obstacles):
        for coordinateBox in self.boxes:
            if coordinateBox in storages:
                continue
            l = tuple(map(int.__add__, coordinateBox, (0, -1)))
            r = tuple(map(int.__add__, coordinateBox, (0, 1)))
            up = tuple(map(int.__add__, coordinateBox, (-1, 0)))
            bot = tuple(map(int.__add__, coordinateBox, (1, 0)))

            dur = tuple(map(int.__add__, coordinateBox, (-1, 1)))
            dul = tuple(map(int.__add__, coordinateBox, (-1, -1)))
            dbr = tuple(map(int.__add__, coordinateBox, (1, 1)))
            dbl = tuple(map(int.__add__, coordinateBox, (1, -1)))

            if (r in obstacles and bot in obstacles) or (l in obstacles and bot in obstacles) or \
               (l in obstacles and up in obstacles) or (r in obstacles and up in obstacles):
                return True
            if (r in obstacles or r in self.boxes) and (bot in obstacles or bot in self.boxes) and (dbr in obstacles or dbr in self.boxes):
                return True
            if (l in obstacles or l in self.boxes) and (bot in obstacles or bot in self.boxes) and (dbl in obstacles or dbl in self.boxes):
                return True
        return False

    def isGoalState(self, storages):
        for box in self.boxes:
            if box not in storages:
                return False
        return True

    def getMap(self, obstaclesIn, storagesIn, highIn, widthIn):
        matrix = [[' ' for col in range(widthIn)] for row in range(highIn)]
        for obstacles in obstaclesIn:
            matrix[obstacles[0]][obstacles[1]] = 'W'
        for storages in storagesIn:
            matrix[storages[0]][storages[1]] = 'X'
        for box in self.boxes:
            matrix[box[0]][box[1]] = 'B'
        matrix[self.player[0]][self.player[1]] = 'I'
        return matrix

class Node():
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent

    def getPath(self):
        path = [self.state.movement]
        actual = self.parent
        while actual:
            path.append(actual.state.movement)
            actual = actual.parent
        path.reverse()
        return path

    def getMoves(self):
        path = self.getPath()
        nameOfMoves = {(0, 0): '', (0, -1): 'L', (1, 0): 'D', (0, 1): 'R', (-1, 0): 'U'}
        formatMoves = ''.join([nameOfMoves[moves] for moves in path])
        return formatMoves

    def getPathMaps(self, obstaclesIn, storagesIn, highIn, widthIn):
        pathOfStates = [self.state.getMap(obstaclesIn, storagesIn, highIn, widthIn)]
        actual = self.parent
        while actual:
            pathOfStates.append(actual.state.getMap(obstaclesIn, storagesIn, highIn, widthIn))
            actual = actual.parent
        pathOfStates.reverse()
        return pathOfStates

    def isParent(self):
        last = self.parent
        actual = self.state
        while last:
            if last.state.boxes == actual.boxes and last.state.player == actual.player:
                return True
            last = last.parent
        return False

class NodeDepth(Node):
    def __init__(self, state, parent, depth):
        Node.__init__(self, state, parent)
        self.depth = depth