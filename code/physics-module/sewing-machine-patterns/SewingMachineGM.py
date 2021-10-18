# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#importing libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math
from drawnow import drawnow

#defining the Geometrical Model
def GeometricalModel(init,t,V_U,Rc):
    """
    inputs:
    -> init: initial conditions
    -> t:    time variable
    -> V_U:  V_belt/Uc (factor indicating Vbelt/Uthread)
    -> Rc:   The Steady Coiling radius
    """
    r=init[0] #asigning first initial value to r
    y=init[1] #assigning second initial value to y
    th=init[2] #assigning third initial value to th
    dr=math.cos(th-y)+math.cos(y)*V_U #the r' differantial equation as defined in paper
    dy=(math.sin(th-y)-math.sin(y)*V_U)/r #the y' differential equation as defined in paper
    dth=(1/Rc)*math.sqrt(r/Rc)*(1+(0.715**2*math.cos(th-y)*r)/(Rc*(1-0.715*math.cos(th-y))))*math.sin(th-y) #the th' differential equation as defined in paper
    return[dr,dy,dth] #returning the values for dr (r'), dy (y'),dth (th')


def main_GM_run(v_u):
    Uc=1.
    t=np.linspace(0.,200,10000)
    V_U=v_u/100.
    V=V_U*Uc
    Rc=1.
    if V_U<=0.6:
        init=[1.,math.pi/3,math.pi/6]
    else:
        init=[0.6,math.pi/2,-1]

    s=Uc*t
    r = np.empty_like(t)  # create empty array to fill with r values
    y = np.empty_like(t)  # create empty array to fill with y values
    th = np.empty_like(t)  # create empty array to fill with th values
    r[0] = init[0]  # asigning first initial value to r
    y[0] = init[1]  # asigning first initial value to y
    th[0] = init[2]  # asigning first initial value to th

    for i in range (1,10000):
        tspan=[t[i-1],t[i]]
        z=odeint(GeometricalModel,init,tspan,args=(V_U,Rc,))
        r[i] = z[1][0]
        y[i] = z[1][1]
        th[i] = z[1][2]
        init = z[1]

    xi=r*np.cos(y)
    yi=r*np.sin(y)
    fig1,ax1=plt.subplots(1)
    ax1.set_xlabel("x/Rc")
    ax1.set_ylabel("y/Rc")
    ax1.plot(xi[5000:]/Rc,yi[5000:]/Rc,alpha=0.8,linewidth=1.)
    ax1.set_title("Orbit at V/Uc="+str(V_U))
    ax1.grid(True)

    #fig1.show()
    fig1.savefig("GMData Orbit"+str(V_U)+".png",dpi=1000,bbox_inches="tight")



    xj = xi + V * (t[i] - s / Uc)

    fig2, ax2 = plt.subplots(1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.plot(xj[5000:], yi[5000:], alpha=0.8, linewidth=1.)
    ax2.set_title("Trace at V/Uc=" + str(V_U))
    ax2.grid(True)
    #fig2.show()
    fig2.savefig("GMData Trace" + str(V_U) + ".png", dpi=1000, bbox_inches="tight")



    data = []
    data.append(xj[5000:])
    data.append(yi[5000:])
    data = np.array(data)
    np.savetxt("GMData at "+str(V_U)+".csv", data.T, delimiter=",")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range (80,101):
        main_GM_run(i)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
