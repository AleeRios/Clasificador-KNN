#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 22:04:22 2023

Elaborado por: Rios Campusano Beckham Alejandro
    
@author: alebe
"""

import os
import cv2
import numpy as np

class Tratamiento:
    
    def __init__(self):
        self.rutas = []
        self.lista_imagenes = []
    
    def tratar(self, ruta, typ=cv2.THRESH_BINARY, etiq=""):
        cad = ""
        self.rutas.append(ruta)
        self.imgs = os.listdir(ruta)
        
        for i in self.imgs:
            img = cv2.imread(ruta + "/" + i)
            img = cv2.resize(img, (100, 100)) #Ajustar el tamaño
            b, g, r = cv2.split(img) #Extraer los planos de color
            img = cv2.GaussianBlur(img, (5,5), 0) # Elimiar el ruido
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Imagen escala de grises
            _, img = cv2.threshold(img, 127, 255, typ) #Binarizar imagenes
            img = cv2.erode(img, (3,3)) #Erocionar imagen
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (5,5)) #Clausura
            con, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Extraer los contornos
            cnt = con[0]
            Mom = cv2.moments(cnt) #Extraer los momentos
            Hu = cv2.HuMoments(Mom).flatten() #Momentos de Hu
            
            if Hu[0] != 0:
                area = str(cv2.contourArea(cnt))
                peri = str(round(cv2.arcLength(cnt, True), 4))
                promB = str(round(np.sum(b) / len(b), 4))
                promG = str(round(np.sum(g) / len(g), 4))
                promR = str(round(np.sum(r) / len(r), 4))
                m0 = str(Hu[0])
                m1 = str(Hu[1])
                m2 = str(Hu[2])
                m3 = str(Hu[3])
                m4 = str(Hu[4])
                m5 = str(Hu[5])
                m6 = str(Hu[6])
                self.lista_imagenes.append(img)
            
                #plt.xticks([])
                #plt.yticks([])
                #plt.imshow(img, cmap='gray', interpolation='bicubic')
                #plt.show()
                cad += area + "," + peri + "," + promB + "," + promG + "," + promR
                cad += "," + m0 + "," + m1 + "," + m2 + "," + m3 + "," + m4 + ","
                cad += m5 + "," + m6 + "," + etiq + "\n"
            else:
                continue
        return cad
    
    def tratarImagenesEntrenamiento(self):
        cad = "Area,Perimetro,promB,promG,promR,m0,m1,m2,m3,m4,m5,m6,clase\n" 
        rutaC = "C:/Users/alebe/Downloads/Proyecto/Imagenes/entrenamiento/coca"
        rutaE = "C:/Users/alebe/Downloads/Proyecto/Imagenes/entrenamiento/extasis"
        rutaMar = "C:/Users/alebe/Downloads/Proyecto/Imagenes/entrenamiento/mari"
        rutaMet = "C:/Users/alebe/Downloads/Proyecto/Imagenes/entrenamiento/meta"
        cad += self.tratar(rutaC, etiq="Cocaina")
        cad += self.tratar(rutaE, etiq="Extasis") 
        cad += self.tratar(rutaMar, cv2.THRESH_BINARY_INV, etiq="Marihuana")
        cad += self.tratar(rutaMet, etiq="Metanfetamina")
        
        ruta = "C:/Users/alebe/Downloads/Proyecto/Imagenes/entrenamiento/Entrenamiento.csv"
        a = open(ruta, "w")
        a.write(cad)
        a.close()
        return ruta
        
        
    def tratarImagenesPrueba(self):
        cad = "Area,Perimetro,promB,promG,promR,m0,m1,m2,m3,m4,m5,m6,clase\n" 
        rutaC = "C:/Users/alebe/Downloads/Proyecto/Imagenes/prueba/coca"
        rutaE = "C:/Users/alebe/Downloads/Proyecto/Imagenes/prueba/extasis"
        rutaMar = "C:/Users/alebe/Downloads/Proyecto/Imagenes/prueba/mari"
        rutaMet = "C:/Users/alebe/Downloads/Proyecto/Imagenes/prueba/meta"
        cad += self.tratar(rutaC, etiq="Cocaina")
        cad += self.tratar(rutaE, etiq="Extasis") 
        cad += self.tratar(rutaMar, cv2.THRESH_BINARY_INV, etiq="Marihuana")
        cad += self.tratar(rutaMet, etiq="Metanfetamina")
        
        ruta = "C:/Users/alebe/Downloads/Proyecto/Imagenes/prueba/Prueba.csv"
        a = open(ruta, "w")
        a.write(cad)
        a.close()
        return ruta
    
    def tratarImagen(self, ruta, etiq="s/i"):
        cad = "Area,Perimetro,promB,promG,promR,m0,m1,m2,m3,m4,m5,m6,clase\n"
        img = cv2.imread(ruta)
        img = cv2.resize(img, (100, 100))
        b, g, r = cv2.split(img)
        img = cv2.GaussianBlur(img, (5,5), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, (3,3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (5,5))
        con, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = con[0]
        Mom = cv2.moments(cnt)
        Hu = cv2.HuMoments(Mom).flatten()
        
        #if Hu[0] != 0:
        area = str(cv2.contourArea(cnt))
        peri = str(round(cv2.arcLength(cnt, True), 4))
        promB = str(round(np.sum(b) / len(b), 4))
        promG = str(round(np.sum(g) / len(g), 4))
        promR = str(round(np.sum(r) / len(r), 4))
        m0 = str(Hu[0])
        m1 = str(Hu[1])
        m2 = str(Hu[2])
        m3 = str(Hu[3])
        m4 = str(Hu[4])
        m5 = str(Hu[5])
        m6 = str(Hu[6])
        #self.lista_imagenes.append(img)
    
        #plt.xticks([])
        #plt.yticks([])
        #plt.imshow(img, cmap='gray', interpolation='bicubic')
        #plt.show()
        cad += area + "," + peri + "," + promB + "," + promG + "," + promR
        cad += "," + m0 + "," + m1 + "," + m2 + "," + m3 + "," + m4 + ","
        cad += m5 + "," + m6 + "," + etiq + "\n"
        
        ruta_c = "C:/Users/alebe/Downloads/Proyecto/image.csv"
        a = open(ruta_c, "w")
        a.write(cad)
        a.close()
        return ruta_c