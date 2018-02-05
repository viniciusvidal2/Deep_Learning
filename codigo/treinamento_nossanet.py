# Rotina de treinamento da nossa net, salvando e plotando o modelo para seguir em frente
# no reconhecimento dos postes

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

# Adiciona caminhos e arquivos necessarios dentro das pastas
from tools.preprocessing import ImageToArrayPreprocessor
from tools.preprocessing import SimplePreprocessor
from tools.datasets      import SimpleDatasetLoader
from tools.conv import NossaNet
from tools.conv import LeNet
from tools.conv import MiniVGGNet
from tools.conv import ShallowNet

# Callback para salvar melhor rede
from keras.callbacks import ModelCheckpoint
# Carregar a rede e plotar
from keras.models import load_model
from keras.utils import plot_model

# grab the list of images that we'll be describing
print("[INFO] loading images...")
path = "../datasets/POSTE/"
imagePaths = list(paths.list_images(path))


# initialize the image preprocessors
sp = SimplePreprocessor(35, 250) # Tentativa para focar bem no poste
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=50)
data = data.astype("float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# Separando o conjunto de treino e validacao de forma aleatoria
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=30)

# Numero de epocas pra ficar algo profissional
epochs = 100

# initialize the optimizer
print("[INFO] compiling model...")
opt = SGD(lr=0.001, decay=1/epochs, momentum=0.9, nesterov=True)

# Utilizando o modelo do caso atual
# model = ShallowNet.build(width=35, height=250, depth=3, classes=2)
# model = MiniVGGNet.build(width=35, height=250, depth=3, classes=2)
# model = LeNet.build(width=35, height=250, depth=3, classes=2)
model = NossaNet.build(width=35, height=250, depth=3, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Criando callback para salvar melhor rede baseado na validation_loss
melhor = ModelCheckpoint("Melhores_redes/atual.hdf5", save_best_only=True, verbose=2, monitor='val_loss')

# train the network
batch = 20
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=batch, epochs=epochs, callbacks=[melhor], verbose=2)

# Carregando a melhor rede para avaliar a partir dela
model = load_model("Melhores_redes/atual.hdf5")
plot_model(model, "Melhores_redes/arquitetura.png", show_shapes=True)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=["nao", "postes"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
