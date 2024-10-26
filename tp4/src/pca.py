import numpy as np
from numpy import linalg as LA


class PCA():
    def __init__(self, n) -> None:
        self.components= n
        
    #Calculo de matriz de correlación en datos normalizados
    def train(self, X_standard):
        s= np.corrcoef(X_standard, rowvar=False)
        #Computo de autovalores y actovectores.
        autovalores, autovectores = LA.eig(s)
        #Ordeno autovalores
        self.autovalores_sort= np.sort(autovalores)[::-1]
        #Ordenamiento de autovectores, según autovalores ordenados.
        self.autovectores_sort = [autovectores[:, i] for i in np.argsort(autovalores)[::-1]]
    #Obtenemos PC1, cómo el auto vector formado por el mayor autovalor. De módo análogo obtenemos PC2
        self.loadings_PC1=  self.autovectores_sort[0]
        self.loadings_PC2=  self.autovectores_sort[1]
    def autovalores(self):
        return self.autovalores_sort[:self.components]
    def autovectores(self):
        return self.autovectores_sort[:self.components]
    def ratio(self):
        total= np.sum(self.autovalores_sort)
        return (self.autovalores_sort/total)
    def calc_PC1(self, X_standard):
        #Calculo de PC1
        PC1= np.dot(X_standard, self.loadings_PC1)
        return PC1
    def calc_PC2(self, X_standard):
        #Calculo de PC2
        PC1= np.dot(X_standard, self.loadings_PC2)
        return PC1

