import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from brainflow.data_filter import DataFilter

loadedData = np.load("./model_data/data/left/1572814991.npy")

# print(np.shape(loadedData))

Plot, Axis = plt.subplots()

f = range(60)
for i in range(16):
    x = loadedData[0,i,:]
    plt.plot(f,x)

plt.ylim([0, 60])

# Set Up Slider
slider_color = 'White'

axis_position = plt.axes([0.2,0.03,0.65,0.03], facecolor = slider_color)
slider_position = Slider(axis_position, 'Time', 0, 250, valinit=0, valstep=range(250))

def update(val):
    pos = slider_position.val
    Axis.cla()
    for i in range(16):
        x = loadedData[pos,i,:]
        Axis.plot(f,x)
    Axis.set_ylim(0,60)
    Plot.canvas.draw_idle()
    
slider_position.on_changed(update)

plt.show()