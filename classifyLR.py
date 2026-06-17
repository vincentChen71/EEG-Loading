import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import layers

CLASSES = ["left","none","right"]

train_data = []
valid_data = []

train_counts = [0,0,0]
valid_counts = [0,0,0]

for idx, c in enumerate(CLASSES):
    train_dir = os.path.join("model_data/data", c)
    onehot = np.zeros((3))
    onehot[idx] = 1
    runningTrainCount = 0
    for i in os.listdir(train_dir):
        d = np.load(os.path.join(train_dir,i))
        sys.stdout.write("\033[K")
        print("Loaded",os.path.join(train_dir,i),end="\r")
        runningTrainCount = runningTrainCount + len(d)
        for e in d:
            train_data.append([e,onehot])
    train_counts[idx] = runningTrainCount

    valid_dir = os.path.join("model_data/validation_data", c)
    runningValidCount = 0
    for i in os.listdir(valid_dir):
        d = np.load(os.path.join(valid_dir,i))
        sys.stdout.write("\033[K")
        print("Loaded",os.path.join(valid_dir,i),end="\r")
        runningValidCount = runningValidCount + len(d)
        for e in d:
            valid_data.append([e,onehot])
    valid_counts[idx] = runningValidCount

print("")
print("---------------------------")
print("Training Data Lengths:", train_counts)
print("Valid Data Lengths:", valid_counts)

# shuffle
np.random.shuffle(train_data)
np.random.shuffle(valid_data)

# Splitting out into X and Y vectors after shuffling
train_X = []
train_Y = []
for x,y in train_data:
    train_X.append(x)
    train_Y.append(y)
valid_X = []
valid_Y = []
for x,y in valid_data:
    valid_X.append(x)
    valid_Y.append(y)

# Convert to numpy arrays
train_X = np.array(train_X)
valid_X = np.array(valid_X)

train_Y = np.array(train_Y)
valid_Y = np.array(valid_Y)

print(np.shape(train_X))

model = keras.Sequential()
model.add(layers.GRU(32)) # encoding into 32-dim features
model.add(layers.Dense(3)) # output layer
model.add(layers.Activation('softmax')) # softmax output layer

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 10
batch_size = 32
for epoch in range(epochs):
    model.fit(train_X, train_Y, batch_size=batch_size, epochs=1, validation_data=(valid_X, valid_Y))
    score = model.evaluate(valid_X, valid_Y, batch_size=batch_size)
    print(score)