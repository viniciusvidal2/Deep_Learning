# Rotina para reconhecer o poste em uma foto com rede treinada.

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from cv2 import *
import time
# Ajustar o tamanho da imagem ao tamanho da entrada da rede
from keras.preprocessing.image import img_to_array

def scales(windowSize_base):
	for scale in range(1, 5, 1):
		yield windowSize_base*scale # precisa ser array

def sliding_window(image, stepSizeW, stepSizeH, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0]-2*stepSizeH, stepSizeH):
		for x in range(0, image.shape[1]-2*stepSizeW, stepSizeW):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def show_posts(foto, xp, yp, wsp, scr):
	if len(xp) > 0:
	    for p in range(len(xp)):
		    rectangle(foto, ( xp[p], yp[p] ), ( xp[p]+wsp[p][0], yp[p]+wsp[p][1] ) , (255, 0, 0), 2)
	    scr = np.array(scr)
	    bests = 2
	    scr_index = scr.argsort()[::-1][:bests]
	    for p in range(bests):
		    rectangle(foto, ( xp[scr_index[p]], yp[scr_index[p]] ),
			    	  ( xp[scr_index[p]]+wsp[scr_index[p]][0], yp[scr_index[p]]+wsp[scr_index[p]][1] ), (0, 0, 255), 2)
	imshow("postes total", foto)
	waitKey(0)
	destroyAllWindows()

# Carregando o modelo de interesse
model = load_model("Melhores_redes/atual.hdf5") # Cuidado com a manipulacao do arquivo

# Carregando imagem de teste
foto = imread("/home/vinicius/Desktop/Deep_Learning/datasets/Testes/frame0227.jpg")

# imshow("teste ok", foto)
# waitKey(0)
# destroyAllWindows()

# Definindo janela inicial
window_size = np.array((50, 250))

# Guardando locais onde se encontra o poste
x_p = []; y_p = []; window_size_p = []; score = []

print("[INFO] Comecando a varrer a foto...")
# Varrendo a foto escolhida
for sc in scales(window_size):
	# Calculando passo sobre a foto
	many_steps_w = 3*foto.shape[0]//sc[0]
	step_size_w  = foto.shape[0]//many_steps_w

	many_steps_h = 3*foto.shape[1]//sc[1]
	step_size_h  = foto.shape[1]//many_steps_h

	for (x, y, window) in sliding_window(foto, step_size_w, step_size_h, window_size):
		# AJUSTAR A IMAGEM DE ENTRADA
		window = resize(window, (50, 250))
		window = window.astype("float")/255.0
		window = img_to_array(window)
		window = np.expand_dims(window, axis=0) # 4 dimensions pagina 284 livro
		# Ver predicao
		poste = model.predict(window)
		if poste[0][1] > poste[0][0]: # estamos com um poste aqui
			x_p.append(x)
			y_p.append(y)
			window_size_p.append(sc)
			score.append(poste[0][1])

		clone = foto.copy()
		rectangle(clone, (x,y), (x + sc[0], y + sc[1]), (0,255,0),2)
		imshow("Slide", clone)
		waitKey(1)

destroyWindow("Slide")
# Onde estao os postes Luiz?
print("[INFO] Foto varrida, esses sao os postes...")
show_posts(foto=foto, xp=x_p, yp=y_p, wsp=window_size_p, scr=score)

espresso = 1
