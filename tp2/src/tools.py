import json
import math
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def import_json(file_name):

    f= open(file_name, 'r')
    j=json.load(f)  
    f.close()
    return { atr: j[atr] for atr in j}

def calcular_atributos_totales(cromosoma):
    return {
        "fuerza_total": 100 * math.tanh(0.01 * cromosoma["fuerza"]),
        "destreza_total": math.tanh(0.01 * cromosoma["destreza"]),
        "inteligencia_total": 0.6 * math.tanh(0.01 * cromosoma["inteligencia"]),
        "vigor_total": math.tanh(0.01 * cromosoma["vigor"]),
        "constitucion_total": 100 * math.tanh(0.01 * cromosoma["constitución"])
    } 



def plot_band_error_generation(x, y, y_upper, y_lower):
    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # y = [1, 2, 7, 4, 5, 6, 7, 8, 9, 10]
    # y_upper = [2, 3, 8, 5, 6, 7, 8, 9, 10, 11]
    # y_lower = [0, 1, 5, 3, 4, 5, 6, 7, 8, 9]
    # print(x[0], y[0], y_upper[0], y_lower[0])
    # print(np.concatenate((x,x[::-1])))
    # print(np.concatenate((y,y[::-1])))
    # fig = go.Figure([
    #     for i in range(6):
    #         go.Scatter(
    #             x=x,
    #             y=y[:,i],
    #             line=dict(color='rgb(0,100,80)'),
    #             mode='lines'
    #         ),
    #         go.Scatter(
    #             x=np.concatenate((x,x[::-1])), # x, then x reversed
    #             y=np.concatenate((y_upper[:,i],y_lower[:,i][::-1])), # upper, then lower reversed
    #             fill='toself',
    #             fillcolor='rgba(0,100,80,0.2)',
    #             line=dict(color='rgba(255,255,255,0)'),
    #             hoverinfo="skip",
    #             showlegend=False
    #         )
    # ])
    # fig.show()
    colors = ['rgb(0,100,80)', 'rgb(100,0,80)', 'rgb(80,100,0)', 'rgb(0,80,100)', 'rgb(80,0,100)', 'rgb(100,80,0)']
    fill_colors = ['rgba(0,100,80,0.2)', 'rgba(100,0,80,0.2)', 'rgba(80,100,0,0.2)', 'rgba(0,80,100,0.2)', 'rgba(80,0,100,0.2)', 'rgba(100,80,0,0.2)']
    atributos = ["fuerza", "destreza", "inteligencia", "vigor", "constitución", "altura"]

    # Crear subplots en una cuadrícula 3x2
    fig = make_subplots(rows=3, cols=2, shared_xaxes=True, shared_yaxes=False, subplot_titles=atributos)

    # Añadir trazas a los subplots
    for i in range(len(atributos)):
        row = (i // 2) + 1  # Calcula la fila
        col = (i % 2) + 1   # Calcula la columna

        # Añadir la línea
        fig.add_trace(go.Scatter(
            x=x,
            y=y[:, i],
            line=dict(color=colors[i]),
            mode='lines',
            name=atributos[i]  # Asignar nombre
        ), row=row, col=col)

        # Añadir el área sombreada
        fig.add_trace(go.Scatter(
            x=np.concatenate((x, x[::-1])),  # x, luego x invertido
            y=np.concatenate((y_upper[:, i], y_lower[:, i][::-1])),  # upper, luego lower invertido
            fill='toself',
            fillcolor=fill_colors[i],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ), row=row, col=col)

    # Actualizar etiquetas de los ejes
    # for row_ in range(3):
    #     fig.update_xaxes(title_text="Generación", row=row_, col=1)
    #     fig.update_xaxes(title_text="Generación", row=row_, col=2)
    # fig.update_yaxes(title_text="Puntos", row=1, col=1)
    # fig.update_yaxes(title_text="Puntos", row=2, col=1)
    # fig.update_yaxes(title_text="Puntos", row=3, col=1)
    for r in range(1, 4):  # Filas
        for c in range(1, 3):  # Columnas
            fig.update_xaxes(title_text="Generación", row=r, col=c,  tickvals = x)
            fig.update_yaxes(title_text="Puntos", row=r, col=c)
    fig.update_yaxes(title_text="Altura [m]", row=3, col=2)

    # Actualizar el layout
    fig.update_layout(height=900, width=1000, title_text="Distribución de Atributos", showlegend=True)

    # Mostrar la figura
    fig.show()
