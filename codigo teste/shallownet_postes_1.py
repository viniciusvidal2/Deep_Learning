# USAGE
# python shallownet_animals.py --dataset ../datasets/animals

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
#  import argparse

# Adiciona caminhos e arquivos necessarios dentro das pastas
from tools.preprocessing import ImageToArrayPreprocessor
from tools.preprocessing import SimplePreprocessor
from tools.datasets      import SimpleDatasetLoader
from tools.conv       import ShallowNet

# grab the list of images that we'll be describing
print("[INFO] loading images...")
path = "../datasets/POSTE/"
imagePaths = list(paths.list_images(path))


# initialize the image preprocessors
sp = SimplePreprocessor(130, 380)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=25)
data = data.astype("float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, random_state=42)

# convert the labels from integers to vectors
# trainY = LabelBinarizer().fit_transform(trainY)
# testY = LabelBinarizer().fit_transform(testY)

# Numero de epocas pra ficar algo profissional
epochs = 25

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.001, decay=1/epochs, momentum=0.9, nesterov=True)
model = ShallowNet.build(width=130, height=380, depth=3, classes=2)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
batch = 20
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=batch, epochs=epochs, verbose=2)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=["nao2", "postes"]))

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