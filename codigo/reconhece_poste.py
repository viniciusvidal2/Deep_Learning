# Rotina para reconhecer o poste em uma foto com rede treinada.

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from cv2 import *
# Ajustar o tamanho da imagem ao tamanho da entrada da rede
from keras.preprocessing.image import img_to_array

def scales(windowSize_base):
	for scale in range(1, 3, 1):
		yield windowSize_base*scale # precisa ser array

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def show_posts(foto, xp, yp, wsp):
	for p in range(len(xp)):
		rectangle(foto, ( xp[p], yp[p] ), ( xp[p]+wsp[p][0], yp[p]+wsp[p][1] ) , (255, 0, 0), 2)
	imshow("postes total", foto)
	waitKey(0)
	destroyAllWindows()

# Carregando o modelo de interesse
model = load_model("Melhores_redes/melhor_atual.hdf5") # Cuidado com a manipulacao do arquivo

# Carregando imagem de teste
# foto = imread("/home/vinicius/Desktop/Deep_Learning/datasets/Testes/frame0003.jpg")
foto = imread("/home/vinicius/Desktop/Deep_learning/datasets/Testes/frame0003.jpg")
# imshow("teste ok", foto)
# waitKey(0)
# destroyAllWindows()

# Definindo janela inicial
window_size = np.array((60, 190))

# Guardando locais onde se encontra o poste
x_p = []; y_p = []; window_size_p = [];

print("[INFO] Comecando a varrer a foto...")
# Varrendo a foto escolhida
for sc in scales(window_size):
	step_size = foto.shape[0]//window_size[0]
	for (x, y, window) in sliding_window(foto, step_size, window_size):
		# AJUSTAR A IMAGEM DE ENTRADA
		window = resize(window, (65, 190))
		window = window.astype("float")/255.0
		window = img_to_array(window)
		window = np.expand_dims(window, axis=0) # 4 dimensions pagina 284 livro
		# Ver predicao
		poste = model.predict(window)
		if poste[0][1] > poste[0][0]: # estamos com um poste aqui
			x_p.append(x)
			y_p.append(y)
			window_size_p.append(window_size)

		clone = foto.copy()
		# rectangle(clone, (x,y), (x + window_size[0], y + window_size[1]), (0,255,0),2)
		# imshow("Slide", clone)
		# waitKey(1)
		# time.sleep(0.025)

# Onde estao os postes Luiz?
print("[INFO] Foto varrida, esses sao os postes...")
show_posts(foto=clone, xp=x_p, yp=y_p, wsp=window_size_p)

espresso = 1
