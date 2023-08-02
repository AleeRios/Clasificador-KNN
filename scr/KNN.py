#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 19:13:13 2023

Elaborado por: Rios Campusano Beckham Alejandro
    
@author: alebe
"""

import numpy as np

class Knn:
    
    def __init__(self):
        pass
    
    def entrenar(self, datos, etiq, K=5):
        self.datos = datos #DataFrame de datos numericos
        self.etiq = etiq #DataFrame de etiquetas
        self.K = K #Numero de clases
        self.clase = "" #Clase ubicada
        
    def calculaDistancias(self, xi):
        datos = self.datos.values # Obtener los datos del DataFrame
        distancias = [] #Distancias
        for xj in datos: 
            d = self.distanciaEuclidiana(xi, xj) #alcular la distancia
            distancias.append(d) #Agregamos a la lista
        return distancias
    
    def distanciaEuclidiana(self, xi, xj):
        return np.sqrt(np.sum(np.power(xi - xj, 2))) #Calculo de la distancias
    
    def distanciasMenores(self, distancias):
        indices = [] #Lista de indices
        for k in range(self.K): #Numero de clases
            idx = distancias.index(np.min(distancias)) #Obtenemos el indice de la distancia mas pequeña
            distancias[idx] = np.max(distancias) #Ontenemos el indice
            indices.append(idx) #Agregamos a la lista
        return indices #Regresamos el indice
    
    def claseFrecuente(self, indices):
        for i in indices:
            print(self.etiq[i])
            self.clase = self.etiq[i] #Onbtemos la clase que resulto
        
    def predecir(self, datos):
        #etiq_pred = [] #Etiqueta predecida
        datos_test = datos.values #DataFrame de los datos para predecir
        
        for xi in datos_test:
            #print(xi)
            distancias = self.calculaDistancias(xi) #Calculamos la distancia
            indices = self.distanciasMenores(distancias) #Calculamos los indices
            self.claseFrecuente(indices) #Obtenemos la distancia predecida
        
        #return etiq_pred