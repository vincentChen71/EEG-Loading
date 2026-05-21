import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from brainflow.data_filter import AggOperations, DataFilter

loadedData = np.load("./model_data/data/left/1572814991.npy")

NUM_CHANNELS = np.shape(loadedData)[1]
NUM_TIMESTEPS = np.shape(loadedData)[0]
NUM_FREQS = np.shape(loadedData)[2]

# print(np.shape(loadedData))

Plot, Axis = plt.subplots()

f = range(NUM_FREQS)
for i in range(NUM_CHANNELS):
    x = loadedData[0,i,:]
    plt.plot(f,x)

plt.ylim([0, NUM_FREQS])
plt.title("Raw Data")

# Set Up Slider
slider_color = 'White'

axis_position = plt.axes([0.2,0.03,0.65,0.03], facecolor = slider_color)
slider_position = Slider(axis_position, 'Time', 0, NUM_TIMESTEPS, valinit=0, valstep=range(250))

def update(val):
    pos = slider_position.val
    Axis.cla()
    for i in range(NUM_CHANNELS):
        x = loadedData[pos,i,:]
        Axis.plot(f,x)
    Axis.set_ylim(0,NUM_FREQS)
    Axis.set_title("Raw Data")
    Plot.canvas.draw_idle()
    
slider_position.on_changed(update)

# Doing some data processing. Plotting for comparison
Plot2, Axis2 = plt.subplots()
# Denoise
denoisedData = np.zeros(np.shape(loadedData))
for i in range(NUM_CHANNELS):
    for t in range(NUM_TIMESTEPS):
        x = loadedData[t,i,:]
        DataFilter.perform_rolling_filter(x,3,AggOperations.MEAN.value)
        denoisedData[t,i,:] = x
    plt.plot(f,denoisedData[0,i,:])

plt.ylim([0,60])
plt.title("Denoised Data")

# Setting up slider again
slider_color = 'white'

axis_position = plt.axes([0.2,0.03,0.65,0.03], facecolor = slider_color)
slider_position2 = Slider(axis_position, 'Time', 0, NUM_TIMESTEPS, valinit=0, valstep=range(250))

def updateDenoised(val):
    pos = slider_position2.val
    Axis2.cla()
    for i in range(NUM_CHANNELS):
        x = denoisedData[pos,i,:]
        Axis2.plot(f,x)
    Axis2.set_ylim(0,NUM_FREQS)
    Axis2.set_title("Denoised Data")
    Plot2.canvas.draw_idle()
    
slider_position2.on_changed(updateDenoised)

plt.show()