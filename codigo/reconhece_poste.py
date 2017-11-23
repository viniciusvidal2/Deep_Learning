# Rotina para reconhecer o poste em uma foto com rede treinada.

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
from cv2 import *
# Ajustar o tamanho da imagem ao tamanho da entrada da rede
from tools.preprocessing import ImageToArrayPreprocessor
from tools.preprocessing import SimplePreprocessor
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

# Carregando o modelo de interesse
model = load_model("Melhores_redes/melhor_atual.hdf5") # Cuidado com a manipulacao do arquivo
sp = SimplePreprocessor(65, 190)
iap = ImageToArrayPreprocessor()

# Carregando imagem de teste
foto = imread("/home/vinicius/Desktop/Deep_Learning/datasets/Testes/frame0003.jpg")
imshow("teste ok", foto)
waitKey(0)
destroyAllWindows()

# Definindo janela inicial
window_size = np.array((60, 190))

# Varrendo a foto escolhida
for sc in scales(window_size):
	step_size = foto.shape[0]//window_size[0]
	for (x, y, window) in sliding_window(foto, step_size, window_size):
		# AJUSTAR A IMAGEM DE ENTRADA
		window = resize(window, (65, 190))
		window = img_to_array(window)
		# Ver predicao
		(neg, poste) = model.predict(window, batch_size=20, verbose=1)

