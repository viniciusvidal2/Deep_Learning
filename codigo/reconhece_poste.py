# Rotina para reconhecer o poste em uma foto com rede treinada.

import numpy as np
from keras.models import load_model
from cv2 import *
import os
import time

# Ajustar o tamanho da imagem ao tamanho da entrada da rede
from keras.preprocessing.image import img_to_array

def podeOuNaoPode(x_, y_, shape_):
	if (x_ < shape_[1]/2 - 150 or x_ > shape_[1]/2 + 150) and y_ < 450:
		return True
	else:
		return False

def scales(windowSize_base):
	for scale in range(2, 4, 1):
		yield np.array((windowSize_base[0], int(windowSize_base[1]*scale/2))) # precisa ser array

def sliding_window(image, stepSizeW, stepSizeH, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0]-2*stepSizeH, stepSizeH):
		for x in range(0, image.shape[1]-2*stepSizeW, stepSizeW):
			# yield the current window
			if podeOuNaoPode(x, y, image.shape):
				yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def show_posts(foto, xp, yp, wsp, scr, st, npo, nn):
	t = 0.0
	if len(xp) > 0:
		scr = np.array(scr)
		bests = 1
		scr_index = scr.argsort()[::-1][:bests]
		for p in range(bests):
			rectangle(foto, ( xp[scr_index[p]], yp[scr_index[p]] ),
			    	  ( xp[scr_index[p]]+wsp[scr_index[p]][0], yp[scr_index[p]]+wsp[scr_index[p]][1] ), (0, 0, 255), 2)
		t = time.time() - st
		putText(foto, "Score: "+str(scr[scr_index[0]]), (10, 450), FONT_HERSHEY_COMPLEX, 1, (0, 255,   0), thickness=2)
		putText(foto, "Tempo: "+str(t)                , (10, 500), FONT_HERSHEY_COMPLEX, 1, (0, 255, 200), thickness=2)
		# putText(foto, "Postes: " + str(npo), (400, 300), FONT_HERSHEY_COMPLEX, 1, (100, 0,   0), thickness=2)
		# putText(foto, "Naos:   " + str(nn ), (400, 350), FONT_HERSHEY_COMPLEX, 1, (  0, 0, 100), thickness=2)
		# print("Tempo: %.3f"%(time.time()-st))
	imshow("video", foto)
	if len(xp) > 0:
		keypressed = waitKey(1)
		if keypressed == 112: # poste
			npo=npo+1
		elif keypressed == 110: # nao poste
			nn=nn+1
	else:
		waitKey(1)

	return (npo, nn, t)

# COntagem dos postes e nao postes
npostes = 0; nnao = 0;

# Carregando o modelo de interesse
model = load_model("Melhores_redes/profunda_normal.hdf5") # Cuidado com a manipulacao do arquivo

# Varrendo o video com os frames na pasta devida
print("[INFO] Comecando a varrer o video...")
# video_folder = "/home/vinicius/Desktop/Deep_Learning/datasets/image5_r/"
# video_folder = "/home/vinicius/Desktop/Deep_Learning/datasets/play7_rail3_r/"
# video_folder = "/home/vinicius/Desktop/Deep_Learning/datasets/play8_rail2_r/"
# video_folder = "/home/vinicius/Desktop/Deep_Learning/datasets/play1_rail2/"
video_folder = "/home/vinicius/Desktop/Deep_Learning/datasets/play6_rail3/"

# Para analise estatistica apos processamento
score_vector = []
tempo_vetor  = []

for frame in sorted(os.listdir(video_folder)):
	# Carregando frame atual
	foto = imread(video_folder+frame)

	# Definindo janela inicial
	window_size = np.array((35, 250))

	# Guardando locais onde se encontra o poste
	x_p = []; y_p = []; window_size_p = []; score = []

	# Varrendo a foto escolhida
	for sc in scales(window_size):
		# Calculando passo sobre a foto
		many_steps_w = 2*foto.shape[0]//sc[0]
		step_size_w  = foto.shape[0]//many_steps_w

		many_steps_h = 3*foto.shape[1]//sc[1]
		step_size_h  = foto.shape[1]//many_steps_h

		# Medindo tempo da foto
		start = time.time()

		for (x, y, window) in sliding_window(foto, step_size_w, step_size_h, window_size):
			# AJUSTAR A IMAGEM DE ENTRADA
			window = resize(window, (35, 250))
			window = window.astype("float")/255.0
			window = img_to_array(window)
			window = np.expand_dims(window, axis=0) # 4 dimensions pagina 284 livro
			# Ver predicao
			poste = model.predict(window)
			if poste[0][1] > poste[0][0] and poste[0][1] > 0.94: # estamos com um poste aqui
				x_p.append(x)
				y_p.append(y)
				window_size_p.append(sc)
				score.append(poste[0][1])
				score_vector.append(poste[0][1])

			# clone = foto.copy()
			# rectangle(clone, (x,y), (x + sc[0], y + sc[1]), (0,255,0),2)
			# imshow("Slide", clone)
			# waitKey(1)

	# destroyWindow("Slide")
	# Onde estao os postes Luiz?
	# print("[INFO] Foto varrida, esses sao os postes...")
	(npostes, nnao, tempo) = show_posts(foto=foto, xp=x_p, yp=y_p, wsp=window_size_p, scr=score, st=start, npo=npostes, nn=nnao)
	tempo_vetor.append(tempo)

espresso = 1
print("Media de scores: %.2f"%np.mean(np.array(score_vector)))
print("Desvio padrao  : %.3f"%np.sqrt(np.var(np.array(score_vector))))
print("Quantos poste: %d"%npostes)
print("Quantos nao poste: %d"%nnao)
print("Tempo medio: %.2f"%np.mean(np.array(tempo_vetor)) * (len(tempo_vetor))/(npostes+nnao) )