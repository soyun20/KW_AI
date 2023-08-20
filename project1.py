import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

#load data
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks':(), 'yticks':()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
  ax.imshow(image)
  ax.set_title(people.target_names[target])

plt.show()

print("people.images.shape: {}".format(people.images.shape))
print("Class: {}".format(len(people.target_names)))

# Calculate the number of times each target appears
counts = np.bincount(people.target)
# Output name and number of times by target
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='     ')
    if (i + 1) % 3 == 0:
        print()

mask = np.zeros(people.target.shape, dtype=np.bool)
# Select 20 images for each person to address bias
for target in np.unique(people.target):
  mask[np.where(people.target==target)[0][:20]]=1

x_people = people.data[mask]
y_people = people.target[mask]

x_people = x_people / 255.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# data split
x_train, x_test, y_train, y_test = train_test_split(x_people, y_people, stratify = y_people, random_state=0)

# PCA (Learning to KNN without PCA whitening)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
print("{:.2f}".format(knn.score(x_test, y_test)))

from sklearn.decomposition import PCA

# PCA whitening
pca = PCA(n_components=100, whiten=True, random_state=0).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

print("x_train_pca.shape: {}".format(x_train_pca.shape))

fig, axes = plt.subplots(4, 25, figsize=(30, 20), subplot_kw={'xticks':(), 'yticks':()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
  ax.imshow(component.reshape(image_shape), cmap='viridis')
  ax.set_title("PC {}".format(i+1))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train_pca, y_train)
print("{:.2f}".format(knn.score(x_test_pca, y_test)))

from sklearn.decomposition import PCA

# PCA whitening, change n_components parameter
pca = PCA(n_components=125, whiten=True, random_state=0).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

print("x_train_pca.shape: {}".format(x_train_pca.shape))

fig, axes = plt.subplots(5, 25, figsize=(30, 20), subplot_kw={'xticks':(), 'yticks':()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
  ax.imshow(component.reshape(image_shape), cmap='viridis')
  ax.set_title("PC {}".format(i+1))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train_pca, y_train)
print("{:.2f}".format(knn.score(x_test_pca, y_test)))

# CNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# load data
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
# Select 20 images for each person to address bias
for target in np.unique(people.target):
  mask[np.where(people.target==target)[0][:20]]=1

x_people = people.data[mask]
y_people = people.target[mask]

x_people = x_people / 255.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# data split
x_train, x_test, y_train, y_test = train_test_split(x_people, y_people, stratify = y_people, random_state=0)

# CNN
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

input_shape = (87, 65, 1)

batch_size = 128
num_classes = 62
epochs = 1000

# model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# reshape train dataset and test dataset
x_train = x_train.reshape(x_train.shape[0], 87, 65, 1)
x_test = x_test.reshape(x_test.shape[0], 87, 65, 1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ',score[1])
