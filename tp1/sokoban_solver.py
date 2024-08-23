
from scr.tools import readBoard
from scr.tools import measure_algorithm_performance, visualize_moves_subplot
from scr.tools import make_video, show_map
from scr.search import BFS, DFS, IDS, greedy_search, A_star
import matplotlib.animation as animation
import matplotlib as plt

import pandas as pd
import json
import argparse
import sys

cmd_args= sys.argv[1:]
parser = argparse.ArgumentParser(description="Sokoban Solver"+ "\n"+"Autores:"+"\n"+"- Juan Dusau" "\n"+ "- Nicolás Urbano Pintos")
parser.add_argument("--board", default="simple", help="Board name (simple | easy | medium | diagonal | zigzag)")
#Agregar opción heuristica
parser.add_argument("--algorithm", default="greedy_search", help="Algorithm name (BFS | DFS | IDS | greedy_search | A_star)")
parser.add_argument("--media", default= "none", help="Media Output (none | image | animation)")
parser.add_argument("--save", default= False, help= "Save output to file (True | False)")
parser.add_argument("--show", default= False, help= "Show Map before solver (True | False)")
args= parser.parse_args(cmd_args)


f= open("./boards.json", 'r')
j=json.load(f)

niveles_dict={}
for i, niveles in enumerate(j):
  niveles_dict.update({niveles["name"]:i})

print(f'Board name: {args.board}')
print(f'Algorithm name: {args.algorithm}')


board= j[niveles_dict[str(args.board)]]['config']

f.close()

obstacles, storages, stateObj = [], {}, None
obstacles, storages, state, high, width = readBoard(board, obstacles, storages, stateObj)

if args.show:
   show_map(state, obstacles, storages, high, width)


algo_dict={
  "BFS": BFS, 
  "DFS":  DFS,
  "IDS" : IDS, 
  "greedy_search": greedy_search,  
  "A_star": A_star

}
algo= algo_dict[args.algorithm]
save= args.save
result, moves = measure_algorithm_performance(algo, state, obstacles, storages)
data = {
    "Criterio": [
        "Profundidad de la Solución",
        "Nodos Expandidos",
        "Nodos Frontera",
        "Tiempo de Ejecución",
        "Memoria Utilizada",
        "Optimalidad"
    ],
        str(args.algorithm): list(result.values()) if result else ["N/A"]*6,
}

df = pd.DataFrame(data)
pd.set_option('display.colheader_justify', 'center')  # Centramos los encabezados
print(df.to_string(index=False))

print(moves)
if args.media== "image":
    visualize_moves_subplot(save, args.board, args.algorithm, moves, state, obstacles, storages, high, width)
if args.media== "animation":
    make_video(save, args.board, args.algorithm, moves, state, obstacles, storages, high, width)   


