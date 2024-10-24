import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import json
#Función para importar parámetros de un json
def import_json(file_name):

    f= open(file_name, 'r')
    j=json.load(f)  
    f.close()
    return { atr: j[atr] for atr in j}

def plot_dataset(dataset):
    fig, axes = plt.subplots(4, 7, figsize=(10, 5))
    plt.title("Dataset Completo")
    axes = axes.flatten()
    for i, row in dataset.iterrows():
        letter = row['letter']
        matrix = np.array(row[1:].astype(int)).reshape(5, 5)  # Reshape back to 5x5
        ax = axes[i]
        ax.imshow(matrix, cmap='Greys', vmin=-1, vmax=1)
        ax.set_title(f"Letter: {letter}")
        ax.axis('off')  

    plt.tight_layout()
    plt.show()
def plot_comb(title, comb, dataset):
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    plt.title(title)
    axes = axes.flatten()

    j=0
    for worst_letter in comb:
        for i, row in dataset.iterrows():
            letter = row['letter']
            if worst_letter==letter:
                matrix = np.array(row[1:].astype(int)).reshape(5, 5)  
                ax = axes[j]
                ax.imshow(matrix, cmap='Greys', vmin=-1, vmax=1)
                ax.set_title(f"Letter: {letter}")
                ax.axis('off') 
                j+=1

    plt.tight_layout()
    plt.show()
def plot_estados(estados):
    print(len(estados))
    fig, axes = plt.subplots(1, len(estados), figsize=(10, 5))
    plt.title("Estados")
    axes = axes.flatten()
    for i in range(len(estados)):
        matrix = estados[i].reshape(5, 5)  
        ax = axes[i]
        ax.imshow(matrix, cmap='Greys', vmin=-1, vmax=1)
        ax.axis('off') 
   
    plt.tight_layout()
    plt.show()

def calculate_orto(dataset, save=False):
    
    AVG_dot_product = []
    MAX_dot_product = []
    COMB= []
    letters = dataset['letter'].unique()

    all_groups = list(itertools.combinations(letters, 4))


    for g in all_groups:
        comb= g
        COMB.append(comb)
        group= np.zeros((4,25))
        ind=0
        for i, row in dataset.iterrows():
            letter = row['letter']
            if letter in comb:
                group[ind]= np.array(row[1:])
                ind+=1
                
        orto_matrix= group.dot(group.T)
        np.fill_diagonal (orto_matrix, 0)  
        #print(f'{comb}\n{orto_matrix}\n------------')
        row,_= orto_matrix.shape
        avg_dot_product= np.abs(orto_matrix).sum()/(orto_matrix.size-row)         
        AVG_dot_product.append(avg_dot_product)
        max_v = np.abs(orto_matrix).max()
        MAX_dot_product.append((max_v, np.count_nonzero(np.abs(orto_matrix) == max_v) / 2))


    df_orto= pd.DataFrame({"Comb": COMB, "AVG":AVG_dot_product, "MAX": MAX_dot_product})
    if save==True:
        df_orto.to_csv("./data/calc_orto.csv")
    return df_orto

def obtain_letter_matrix(pattern_letter, dataset):
    for i, row in dataset.iterrows():
        letter = row['letter']
        if letter==pattern_letter:
            matrix = np.array(row[1:].astype(int))
    return matrix