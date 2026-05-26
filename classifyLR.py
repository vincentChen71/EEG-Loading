import os
import sys
import numpy as np

CLASSES = ["left","right","none"]

train_data = []
valid_data = []

train_counts = [0,0,0]
valid_counts = [0,0,0]

for idx, c in enumerate(CLASSES):
    train_dir = os.path.join("model_data/data", c)
    onehot = np.zeros((3,1))
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