import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
list=[0.28,0.48,0.64,0.83]
dim=[20,30,40,45]
for i in range(0,len(list)):

    plt.style.use('seaborn-bright')

    fig = plt.figure()
    ax = plt.axes(xlim=(0, dim[i]), ylim=(-5, 5))
    line, = ax.plot([], [], lw=2)

    global data_read
    data_read=np.loadtxt("Results/ExcelFiles/GMData at "+str(list[i])+".csv",delimiter=",",skiprows=1)

    # initialization function
    def init():
        # creating an empty plot/frame
        line.set_data([], [])
        return line,


    # lists to store x and y axis points
    xdata, ydata = [], []


    # animation function
    def animate(i):
        # t is a parameter

        # x, y values to be plotted
        x = data_read[200+i,0]
        y = data_read[200+i,1]

        # appending new points to x, y axes points list
        xdata.append(x)
        ydata.append(y)
        line.set_data(xdata, ydata)
        return line,


    # setting a title for the plot
    plt.title('Trace')
    # hiding the axis details
    plt.axis('on')
    plt.xlabel("x axis")
    plt.ylabel("y axis")

    # call the animator
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=799, interval=2, blit=True)

    # save the animation as mp4 video file

    f = r"Results/animations/Video"+str(list[i])+".gif"
    writergif = animation.PillowWriter(fps=50)
    anim.save(f, writer=writergif)
    plt.cla
    plt.clf()
    plt.close()
    print("DONE!")