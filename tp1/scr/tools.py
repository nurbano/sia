import time
import tracemalloc
import pandas as pd  # Necesario para crear y mostrar la tabla
from .sokoban import State
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


# Función para leer el tablero desde una lista en lugar de archivo
def readBoard(board, obstacles, storages, stateObj):
    agent = ()
    boxes = {}
    numline = 0
    numstorages = 0
    for line in board:
        if line[0] == "W":
            for i, char in enumerate(line):
                if char == "W":
                    obstacles.append((numline, i))
                elif char == "X":
                    storages[(numline, i)] = numstorages
                    numstorages += 1
            numline += 1
        else:
            coords = line.split(",")
            if agent == ():
                # print(coords)
                agent = (int(coords[0]), int(coords[1]))
            else:
                boxes[(int(coords[0]), int(coords[1]))] = numstorages
                numstorages += 1
    stateObj = State(agent, boxes, (0, 0))
    return obstacles, storages, stateObj, len(board), len(board[0])

def measure_algorithm_performance(algorithm, heuristic, stateObj, obstacles, storages, optimal_depth=None):
    start_time = time.time()
    tracemalloc.start()

    result, taboo, frontier_size = algorithm(heuristic, stateObj, obstacles, storages)

    memory_used = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    end_time = time.time()

    if result:
        moves = result.getMoves()
        depth = len(result.getPath()) - 1  # Restamos 1 porque la profundidad inicial es 0
        nodes_expanded = len(taboo)

        # Comparar la profundidad de la solución con la profundidad óptima dada por BFS
        if optimal_depth is not None:
            optimality = "Sí" if depth == optimal_depth else "No"
        else:
            optimality = "No"  # Si no tenemos referencia, no podemos determinar optimalidad

        return {
            "Profundidad de la Solución": depth,
            "Nodos Expandidos": nodes_expanded,
            "Nodos Frontera": frontier_size,
            "Tiempo de Ejecución (s)": end_time - start_time,
            "Memoria Utilizada (MB)": memory_used/1024,
            "Optimalidad": optimality,
                               
        }, moves
    else:
        return None
    
def board_to_array(state, obstacles, storages, high, width):
    board = state.getMap(obstacles, storages, high, width)
    board_array = np.array([[1 if cell == 'W' else 2 if cell == 'X' else 3 if cell == 'B' else 4 if cell == 'I' else 0
                             for cell in row] for row in board])
    return board_array


def visualize_moves_subplot(save, board, algo,moves, state, obstacles, storages, high, width):
    num_moves = len(moves)
    cols = 8  # Número de columnas en la cuadrícula de subplots (8 movimientos por fila)
    rows = num_moves // cols + (num_moves % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    fig.suptitle('Sokoban BFS Movements Visualization')
    axes = axes.flatten()

    current_state = state

    for i, move in enumerate(moves):
        ax = axes[i]
        ax.set_title(f"Move {i+1}: {move}")
        ax.axis('off')

        if move == 'R':
            direction = (0, 1)
        elif move == 'L':
            direction = (0, -1)
        elif move == 'U':
            direction = (-1, 0)
        elif move == 'D':
            direction = (1, 0)

        new_player_pos = (current_state.player[0] + direction[0], current_state.player[1] + direction[1])
        new_boxes = dict(current_state.boxes)

        if new_player_pos in current_state.boxes:
            new_box_pos = (new_player_pos[0] + direction[0], new_player_pos[1] + direction[1])
            new_boxes.pop(new_player_pos)
            new_boxes[new_box_pos] = 0

        current_state = State(new_player_pos, new_boxes, direction)
        board_array = board_to_array(current_state, obstacles, storages, high, width)
        ax.imshow(board_array, cmap='viridis')

        # Dibujar los casilleros objetivos en rojo
        for storage in storages:
            ax.add_patch(plt.Rectangle((storage[1] - 0.5, storage[0] - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))

    # Ocultar los subplots no usados
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    if save:
       fig.savefig(f'{board}_{algo}.png')

def make_frames(moves, state, obstacles, storages, high, width):
  MAT=[]
  current_state = state
  for i, move in enumerate(moves):
    if move == 'R':
      direction = (0, 1)
    elif move == 'L':
      direction = (0, -1)
    elif move == 'U':
      direction = (-1, 0)
    elif move == 'D':
      direction = (1, 0)

    new_player_pos = (current_state.player[0] + direction[0], current_state.player[1] + direction[1])
    new_boxes = dict(current_state.boxes)

    if new_player_pos in current_state.boxes:
      new_box_pos = (new_player_pos[0] + direction[0], new_player_pos[1] + direction[1])
      new_boxes.pop(new_player_pos)
      new_boxes[new_box_pos] = 0

    current_state = State(new_player_pos, new_boxes, direction)
    board_array = board_to_array(current_state, obstacles, storages, high, width)

    MAT.append(board_array)

  return(MAT)

def animate_func(i, tup):
  moves,ax, storages, mat= tup
  if i<len(moves):
    ax.set_title(f"Move {i+1}: {moves[i]}")
    ax.imshow(mat[i])
    for storage in storages:
      ax.add_patch(plt.Rectangle((storage[1] - 0.5, storage[0] - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))
  else:
    ax.imshow(mat[len(mat)-1])
  ax.axis('off')
  return [ax]

def make_video(save, board, algo, moves, state, obstacles, storages, high, width):
    mat = make_frames(moves, state, obstacles, storages,high, width )
    #Creo una figura
    fig,ax = plt.subplots(1, figsize=(9,6))
    a = mat[0]
    #Inicializo imshow con el primer frame
    ax.imshow(a,  cmap='viridis',interpolation='none', vmin= 0, vmax=4)
    tup= [[moves,ax, storages, mat]]
    anim = animation.FuncAnimation(
                            fig,
                            animate_func,
                            frames = len(moves)+5,
                            fargs= tup,
                            interval = 1000/10, # in ms
                            repeat=False)
    plt.show()
    if save:
       anim.save(f'{board}_{algo}.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
   
def show_map(state, obstacles, storages, high, width):
   board_array= board_to_array(state, obstacles, storages, high, width)
   fig,ax= plt.subplots(1)
   ax.imshow(board_array, cmap='viridis')
   for storage in storages:
      ax.add_patch(plt.Rectangle((storage[1] - 0.5, storage[0] - 0.5), 1, 1, fill=False, edgecolor='red', lw=2))

   plt.show()