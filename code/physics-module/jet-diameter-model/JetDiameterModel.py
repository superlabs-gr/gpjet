"""
DATE:           02.23.2021
AUTHOR:         @THANOS_OIKON
DESCRIPTION:    This file contains code to reproduce in python the matlab code by Nikhil Mayadeo
                regarding jet diameter for melt electrospinning
"""
#importing libraries
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve,root
import matplotlib.pyplot as plt
import math

class GlobalPars:
    """
    Messager Object. Its purpose is to carry all glabal parameters wherever there is the need to be transferred
    """
    def __init__(self,H,R,Tnozzle,deltaT,gamma,vair,Na,Pe,beta,Bi,De,alpha,Pec,chi,Re,Bo,Ca,Fe,betaE,Qpm,a,b,Tair,thetainfinity):
        self.H=H
        self.R=R
        self.Tnozzle=Tnozzle
        self.deltaT=deltaT
        self.gamma=gamma
        self.vair=vair
        self.Na=Na
        self.Pe=Pe
        self.beta=beta
        self.Bi=Bi
        self.De=De
        self.alpha=alpha
        self.Pec=Pec
        self.chi=chi
        self.Re=Re
        self.Bo=Bo
        self.Ca=Ca
        self.Fe=Fe
        self.betaE=betaE
        self.Qpm=Qpm
        self.a=a
        self.b=b
        self.Tair=Tair
        self.thetainfinity=thetainfinity

def heaviside_step_function(x):
    return 1*(x<0)

def problemdR(dR,par):
    """
    Function solving eq.33
    :param dR: dR initial in order to get true value via non-linear solving
    :param par: global parameters of the model
    :return: true dR as found from solution of equation 33
    """
    fdR = 6 * dR ** 2 + ((1 / par.Ca) + par.Fe) * dR + ((2 * par.Fe) / ((1 + dR ** 2) ** (0.5))) * (1 - (par.betaE / ((1 + dR ** 2) ** (0.5)))) # solving equation 33 from thesis
    return fdR


def NonIsothermalElectrospinningIC(parameters):
    """
    Function setting problems initial conditions
    :param parameters: global parameters of the model
    :return: initial values for R,theta,dR,tpzz,tprr
    """
    dRguess=-1. #set an initial value for dR, so that true value will be found via non-linear solving
    R=1
    theta=0

    result=root(problemdR,dRguess,args=(parameters,))
    dR=float(result.x)
    print(dR)
    tpzz=float(2*(1-parameters.beta)*(-2*dR))
    tprr=float(-1*(1-parameters.beta)*(-2*dR))
    return R,theta,dR,tpzz,tprr

def NonIsothermalElectrospinningODEIVP(init,x,par):
    """
    Function contains all ordinary differential equations as dydx1,dydx2,dydx3,dydx4,dydx5
    :param x: timesteps
    :param init: initial conditions. lists containing [R,theta,tpzz,tprr,dR]
    :param par: global parameters of the model
    :return: dydx1,dydx2,dydx3,dydx4,dydx5 which actually are R', T', tpzz',tprr',dR'
    """

    dydx1 = init[4] # Equation(34)
    f = math.exp((par.H / (par.R * par.deltaT)) * ((1 / (init[1] + par.gamma)) - (1 / par.gamma))) # Equation(27)
    Qp = (par.Qpm * (heaviside_step_function(x - par.a) - heaviside_step_function(x - par.b))) #Dimensionless heat source parameter for PLA
    dydx2 = ((-2 * par.Na * init[4]) / (par.Pe * init[0])) * (init[2] - init[3] - ((6 * par.beta) / init[0] ** 3) * f * init[4]) - ((2 * (par.Bi * ((1 / init[0] ** 4) ** (1 / 3)) * (((1 + (8 * par.vair * init[0] ** 2) ** 2) / (1
                                                                                          + (8 * par.vair) ** 2)) ** (1 / 6))) * init[0]) / par.Pe) * (init[1] - par.thetainfinity) + (Qp / par.Pe) * init[0] ** 2 # Equation(38)
    dydx3 = ((init[0] ** 2 / f) * ((init[1] + par.gamma) / (par.De * par.gamma))) * (((-4 / init[0] ** 3) * (1 - par.beta) * f * init[4]) - init[2] - ((par.De * par.gamma) / (init[1] +par.gamma)) * (((par.alpha * init[2] ** 2) / (1 - par.beta)) +
                                                     ((4 / init[0] ** 3) * (init[2] * f * init[4])) - ((1 / init[0] ** 2) * f * (init[2] * dydx2 / (init[1] + par.gamma))))) # Equation(39)
    dydx4 = ((init[0] ** 2 / f) * ((init[1] + par.gamma) / (par.De * par.gamma))) * (((2 / init[0] ** 3) * (1 - par.beta) * f * init[4]) - init[3] - (((par.De * par.gamma) / (init[1] +
                                       par.gamma)) * (((par.alpha * init[3] ** 2) / (1 - par.beta)) + ((-2 / init[0] ** 3) * (init[3] * f * init[4]))))) # Equation(40)
    Et = 1 / ((1 + 2 * x - x ** 2 / par.chi) * ((1 + init[4] ** 2) ** 0.5)) #Equation(19)
    dEt = (-2 + (2 * x / par.chi)) / (((1 + 2 * x - x ** 2 / par.chi) ** 2) * ((1 + init[4] ** 2) ** 0.5)) # Equation(20)
    sigma = init[0] - (1 / par.Pec) * init[0] ** 3 * Et # Equation(17)
    dsigma = init[4] - (1 / par.Pec) * ((3 * init[0] ** 2 * init[4] * Et) + (init[0] ** 3 * dEt)) # Equation(18)
    df = math.exp((par.H / (par.R * par.deltaT)) * ((1 / (init[1] + par.gamma)) - (1 / par.gamma))) * (par.H / (par.R * par.deltaT)) * (-1 / ((init[1] + par.gamma) ** 2)) * dydx2 # Equation(28)
    dydx5 = (init[0] ** 3 / (6 * par.beta)) * (1 / f) * (((2 * par.Re * init[4]) / init[0] ** 5) + par.Bo + ((2 * init[4] / init[0]) * (init[2] - init[3] - (6 / init[0] ** 3) * (f * par.beta * init[4]))) + (dydx3 - dydx4 +
                                                    ((18 / init[0] ** 4) * par.beta * f * init[4] ** 2) - (((6 * par.beta) / init[0] ** 3) * df * init[4])) + (init[4] / (par.Ca * init[0] ** 2)) +
                                                    (par.Fe * ((sigma * dsigma) + (par.betaE * Et * dEt) + (2 * sigma * Et) / (init[0])))) # Equation(41)

    return [dydx1,dydx2,dydx3,dydx4,dydx5]


def main(i):
    V0 = 4.216 * math.pow(10,-12) # Volumetric flow rate # change for video S2 to 4.991*math.pow(10,-12)
    Rnozzle = (0.413/2.) * math.pow(10,-3) # nozzle's inner radius
    distance = 3.5 * math.pow(10,-3) # nozzle to collector distance # change for video S2 to 4.5 * math.pow(10,-3)
    v0 = V0 / (math.pi * math.pow(Rnozzle,2)) # jet velocity at the nozzle
    print (v0)
    E0 = 2*7250/(Rnozzle*math.log(1+4*distance/Rnozzle))

    Tnozzle = 87 + 273 # nozzle temperature
    Tair = (20 + 273)  # Surrounding air temperature
    vair = 0  # air velocity

    H = 7938.4 * 8.314  # activation energy of flow
    R = 8.314  # universal gas constant
    beta = 0.001  # viscocity ratio
    alpha = 0.015  # Mobility factor
    k = 0.14 # thermal conductivity
    K = 9.5 * math.pow(10,-9)
    g = 0.0435 # surface tension
    betaE = 2.9 # dielectric constant ratio
    h0 = 148.4 #thermal transfer coefficient
    ita0 = 1900 #zero shear rate viscocity
    lambda0 = 0.019 # relaxation time
    density = 1145 # material density
    Cp = 1340 # heat capacity



    deltaT = (Tnozzle ** 2) / (H / R) # temperature change necessary to substantially alter the rheological properties of the polymer melt
    gamma = Tnozzle / deltaT # temperature factor
    thetainfinity = (Tair - Tnozzle) / deltaT


    chi = int(distance / Rnozzle) + 1 # Ratio of length of experimental setup to initial radius of the polymer melt

    Qp  = i #Magnitude of heat source
    a = 5 # Start of heat
    b = a + 5 # End of heat

    Re = density * v0 * Rnozzle / ita0  # Reynolds number of flow
    Bo = density * 9.81 * math.pow(Rnozzle, 2) / (ita0 * v0)  # Bond number (it is equal to 0 since the flow is not influenced by gravity
    Ca = v0 * ita0/g  # Capillary number
    Fe = 8.8541878 * math.pow(10,-12) * Rnozzle * math.pow(E0,2) / (v0 * ita0) # Electrostatic force parameter
    Pec = 2 * 8.8541878 * math.pow(10, -12) * v0 / (K * Rnozzle)  # Peclet number for electrical conductivity
    Bi = h0 * Rnozzle / k  # Biot number
    De = lambda0 * v0 / Rnozzle  # Deborah number
    Na = ita0 * math.pow(v0, 2) / (k * deltaT)  # Nahme- Griffith number
    Pe = density * Cp * Rnozzle * v0 / k  # Peclet number for thermal conductivity
    print(Bi, De, Bo, Re, Ca, Na, Fe, gamma, Pe, Pec)
    iter_number = 94 # time steps

    x = np.linspace(0., chi, iter_number)

    Radius = np.zeros_like(x)
    T = np.zeros_like(x)

    global_params = GlobalPars(H, R, Tnozzle, deltaT, gamma, vair,
                               Na, Pe, beta, Bi, De, alpha, Pec, chi,
                               Re, Bo, Ca, Fe, betaE, Qp, a, b, Tair,
                               thetainfinity) # Putting global parameters in a class so that they will be portable among functions
    R,theta,dR,tpzz,tprr = NonIsothermalElectrospinningIC(global_params) #initial conditions
    initial_values = np.array([R,theta,tpzz,tprr,dR])
    Radius[0] = R
    T[0] = theta

    Mass = np.zeros((5,5),dtype=np.float32)
    for i in range(0,5):
        for j in range(0,5):
            if i == j:
                Mass[i,j] = 1


    for i in range(1,iter_number):
        xspan = [x[i-1],x[i]]
        results = odeint(NonIsothermalElectrospinningODEIVP,initial_values,xspan,args=(global_params,), atol=True) # Solving the system of ODEs using an implicit time stepper

        initial_values = np.array([results[1][0],results[1][1],results[1][2],results[1][3],results[1][4]]) # putting results as initial step for next step

        Radius[i] = results[1][0]

        T[i] = results[1][1]


    return x, Radius, T, global_params




if __name__=="__main__":
    list=[0]
    Vel=[]
    Radius=[]
    Temper=[]
    for i in list:
        x,r,t,par=main(i)
        vel=[]
        for j in range(0,len(r)):
            vel.append(1./math.pow(r[j],2))
        Vel.append(np.array(vel))

        Radius.append(r)
        Temper.append(t)

    data = [np.array(x), np.array(r)]
    print(data)
    np.savetxt("C:/Users/thano/Desktop/ComputerVisionMEW_Paper/MEW_Tool/Geometrical_Models/Jet Diameter Model/ModelDiameters.csv",np.array(data),delimiter=',')
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('z/R0',fontsize=14)
    ax1.set_ylabel('R/R0', color=color,fontsize=14)
    ax1.plot(x,Radius[0], color=color,label="Radius")
    ax1.tick_params(axis='y')
    ax1.set_title("Radius of Jet \n(Low fidelity model)",fontsize=16)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    """
    color = 'tab:red'
    ax2.set_ylabel('v/v0', color=color,fontsize=14)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y')
    ax2.plot(x,Vel[0], color=color,label="Velocity")
    """


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

