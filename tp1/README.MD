# TP1: Métodos de Búsqueda

## Autores: 
- Juan Dusau 
- Nicolás Urbano Pintos

## Requerimientos

Primero es necesario clonar el respositorio:

```console
git clone https://github.com/nurbano/sia
```
Luego hay que ir al directorio tp1:

```console
cd sia/tp1
```
Es necesario instalar primero los paquetes requeridos:

```console
pip install -r requirements.txt
```

## Uso Sokoban Solver

Para ejecturar el programa utilizo:

```console
python sokoban_solver.py  --help
usage: sokoban_solver.py [-h] [--board BOARD] [--algorithm ALGORITHM] [--heuristic HEURISTIC] [--media MEDIA] [--save SAVE] [--show SHOW]
                         [--i I]

Sokoban Solver Autores: - Juan Dusau - Nicolás Urbano Pintos

options:
  -h, --help            show this help message and exit
  --board BOARD         Board name (simple | easy | medium | diagonal | zigzag)
  --algorithm ALGORITHM
                        Algorithm name (BFS | DFS | IDS | greedy_search | A_star)
  --heuristic HEURISTIC
                        Heuristic for A* or Greedy Serach (manhattan | minimum_matching | grouping | deadlock)
  --media MEDIA         Media Output (none | image | animation)
  --save SAVE           Save output to file (True | False)
  --show SHOW           Show Map before solver (True | False)
  --i I                 Repeat the search i times
```
Por ejemplo: 
```console
python sokoban_solver.py  --board easy --algorithm A_star --heuristic minimum_matching
```

## Análisis de variabilidad
Para realizar el análisis de variablidad es necesario ejecutar son un mismo tablero los 5 algoritmos, definiendo i, cómo la cantidad de iteraciones de cada uno.
Luego se pueden graficar los boxplots desde el notebook Variability.ipynb.
