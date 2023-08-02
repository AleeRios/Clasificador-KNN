#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 01:51:51 2023

Elaborado por: Rios Campusano Beckham Alejandro
    
@author: alebe
"""

from Tratamiento import Tratamiento as tt
from KNN import Knn as k_nn
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import cv2
import imutils
from PIL import Image, ImageTk

#Tratamos las imagenes de prueba para obtener sus datos y las guardatos en un csv
t = tt()
rE = t.tratarImagenesEntrenamiento()
rP = t.tratarImagenesPrueba()

#Cargar csv
datosE = pd.read_csv(rE)
datosP = pd.read_csv(rP)
#Separar rasgos de clases
datos_E = datosE.iloc[:, 0:-1]
etiq_E = datosE.iloc[:, -1]

datos_P = datosP.iloc[:, 0:-1]
etiq_P = datosP.iloc[:, -1]

#del datosE
#del datosP

#KNN
knn = k_nn()
knn.entrenar(datos_E, etiq_E, K=1)
#knn.predecir(datos_P)
#datosP.iloc[0:1, 0:-1]
#knn.predecir(datosP.iloc[0:1, 0:-1])

def abrir():
    #Seleccionar archivo
    file = filedialog.askopenfilename(initialdir="C:/Users/alebe/Downloads/Proyecto/Imagenes/prueba", title="Selecciona el archivo",filetypes=(("image files","*.jpg"),("all files","*.*")))
    rI = t.tratarImagen(file)
    datosI = pd.read_csv(rI)
    datos_I = datosI.iloc[:, 0:-1]
    knn.predecir(datos_I) #Predecir seleccion de las imagenes
    
    
    text = tk.Label(frame, font=("Arial", 12)) #Etiqueta para resultado
    text.config(text="")
    text.config(text=knn.clase)
    text.place(x=250, y=100)
    
    
    #Actualizar la imagen
    img = cv2.imread(file)

    imagen = tk.Label(frame, bg="black", width=230, height=235)
    imagen.place(x=1, y=1)
    img = imutils.resize(img, width=230, height=235)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    imagen.configure(image=imgtk)
    imagen.Image = imgtk

raiz = tk.Tk()
raiz.title("Clasificador de drogas")
raiz.geometry("400x272")

frame = tk.Frame()
frame.pack()
frame.config(width="590", height="390",bg="sky blue", bd=20, relief="sunken")

img = cv2.imread("C:/Users/alebe/Downloads/Proyecto/drugs.jpg")

imagen = tk.Label(frame, bg="black", width=230, height=235)
imagen.place(x=1, y=1)
img = imutils.resize(img, width=230, height=235)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=img)
imagen.configure(image=imgtk)

btn1 = tk.Button(frame, text="Seleccionar", command=abrir, bg="snow")
btn1.place(x=300, y=214)

raiz.mainloop()