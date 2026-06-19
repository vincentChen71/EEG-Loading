import os
import sys
import numpy as np
from brainflow.data_filter import AggOperations, DataFilter

CLASSES = ["left","none","right"]

for idx, c in enumerate(CLASSES):
    train_dir = os.path.join("model_data/data", c)
    for i in os.listdir(train_dir):
        d = np.load(os.path.join(train_dir,i))
        dShape = np.shape(d)
        bpData = np.zeros([dShape[0],dShape[1]])
        for e in range(dShape[1]):
            for t in range(dShape[0]):
                denoisedData = d[t,e,:]
                DataFilter.perform_rolling_filter(denoisedData,3,AggOperations.MEAN.value)
                inTuple = (denoisedData,np.array(range(dShape[2]),np.float64))
                bp = DataFilter.get_band_power(inTuple,0,dShape[2])
                bpData[t,e] = bp
        np.save(os.path.join("processed_data/data", c, i),bpData)
        sys.stdout.write("\033[K")
        print("Processed",os.path.join(train_dir,i),end="\r")
        

    valid_dir = os.path.join("model_data/validation_data", c)
    for i in os.listdir(valid_dir):
        d = np.load(os.path.join(valid_dir,i))
        dShape = np.shape(d)
        bpData = np.zeros([dShape[0],dShape[1]])
        for e in range(dShape[1]):
            for t in range(dShape[0]):
                denoisedData = d[t,e,:]
                DataFilter.perform_rolling_filter(denoisedData,3,AggOperations.MEAN.value)
                inTuple = (denoisedData,np.array(range(dShape[2]),np.float64))
                bp = DataFilter.get_band_power(inTuple,0,dShape[2])
                bpData[t,e] = bp
        np.save(os.path.join("processed_data/validation_data", c, i),bpData)
        sys.stdout.write("\033[K")
        print("Processed",os.path.join(valid_dir,i),end="\r")