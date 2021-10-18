"""
Created on:     28.04.2021
Created by:     @thanos_oikon
Description:    Performing simple GP regression on jet radius data
"""
# importing libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import GPy

# --- creating proper matplotlib style ---

mpl.rcParams['font.family'] = 'Verdana'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 7
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 7
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.top'] = 'on'
mpl.rcParams['xtick.minor.top'] = 'on'
mpl.rcParams['ytick.major.right'] = 'on'
mpl.rcParams['ytick.minor.right'] = 'on'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.titlepad'] = 10

def high_fidelity_data(data):
    """
    Function that clean up the data a bit
    :param data: diameters from Computer Vision Algorithm
    :return: mean diameter values across z-axis
    """
    real_data = []
    for i in range(0, data.shape[0]):
        if data[i, -1] < 5 * 10 ** (-5) and data[i, -1] > 10 ** -5:
            real_data.append(data[i, :]/ data[i,0])
    real_data = np.array(real_data)

    mean=[]
    std=[]
    for i in range(0,real_data.shape[1]):
        mean.append(np.mean(real_data[:,i]))
        std.append(np.std(real_data[:,i]))
    mean=np.array(mean)
    std=np.array(std)
    return mean

def plot_gp(X, mean, covariance, training_points=None):
    """
    Plotting utility to plot a GP fit with 95% confidence interval
    :param X: x-values, inputs of the gaussian process
    :param mean: mean predictions of the gaussian process
    :param covariance: covariance predictions of the gaussian process
    :param training_points: training points to scatter
    """
    # plot 95% confidence interval
    plt.fill_between(X[:, 0], mean[:, 0] - 1.96 * np.sqrt(np.diag(covariance)),
                     mean[:, 0] + 1.96 * np.sqrt(np.diag(covariance)), color='y', alpha=0.3)
    # plot GP mean and initial training points
    plt.plot(X, mean, '--', c='y')
    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["GP fit", "observations"])
    else:
        plt.legend(labels=["GP fit"])


def create_GP():
    # import csv dataset
    data1= np.loadtxt('Total_Diameters.csv', delimiter=',')
    data2= np.loadtxt('ModelDiameters.csv', delimiter=',')

    # print(data)
    speed_ratio_tot = data2[0,:].reshape(-1, 1) # speed ratio is x input values
    mean_m_dist_tot = high_fidelity_data(data1).reshape(-1,1) # mean distance is y output values
    speed_ratio_tot = speed_ratio_tot[::10,:]
    mean_m_dist_tot = mean_m_dist_tot[::10,:]

    # print(speed_ratio.shape)
    # print(mean_m_dist.shape)
    # create figure of data scattered
    plt.figure(figsize=(8,6))
    plt.plot(speed_ratio_tot, mean_m_dist_tot, 'ro', ms=6)
    plt.xlabel(r'$\mathregular{speed-ratio}$', fontsize=12)
    plt.ylabel(r'$\mathregular{lag-distance}$', fontsize=12)
    plt.savefig('Figure_1_GPs.jpg', dpi=300,
                bbox_inches='tight')
    plt.show()
    # create kernel
    k = GPy.kern.RBF(1,name='rbf',lengthscale=20, variance=1)
    # create model
    m = GPy.models.GPRegression(speed_ratio_tot, mean_m_dist_tot, k)
    m.Gaussian_noise.fix(0)  # add noise
    # optimize hyperparameters
    m.optimize(messages=True)
    # run optimization multiple times (5) to get a good estimates
    # it can be quite time consuming
    m.optimize_restarts(5, robust=False)
    print(m) # printing model optimized hyperparameters

    # create array of speed ratios so that we can run predictions, here we define 100 points
    speed_ratio = np.linspace(0, np.max(speed_ratio_tot), 100).reshape(-1, 1)

    # run prediction
    mean, Cov = m.predict(speed_ratio,full_cov=True)
    # create figure of predictions (mean values and uncertainty bounds) and data points
    plt.figure(figsize=(8, 6))
    plot_gp(speed_ratio, mean, Cov)
    #plt.plot(speed_ratio_train, mean_m_dist_train, 'kx', ms=10)
    plt.plot(speed_ratio_tot, mean_m_dist_tot, "ro", ms=6)
    plt.legend(labels=["GP fit", "observations"])
    plt.xlabel(r'$\mathregular{Z/R_0}$', fontsize=18)
    plt.ylabel(r'$\mathregular{R_j/R_0}$', fontsize=18)
    plt.savefig('C:/Users/thano/Desktop/ComputerVisionMEW_Paper/Paper_Images/figure_GPs1.jpg', dpi=300,
                bbox_inches='tight')
    plt.show()
    # sampling a number of functions from the GP
    mean, Cov = m.predict_noiseless(speed_ratio,full_cov=True)
    Z = np.random.multivariate_normal(mean.ravel(), Cov,20).T
    # create figure of sampled functions and data points
    plt.figure(figsize=(8, 6))
    plt.plot(speed_ratio, Z)
    plt.plot(speed_ratio_tot, mean_m_dist_tot, 'ro', ms=6)
    plt.xlabel(r'$\mathregular{Z/R_0}$', fontsize=18)
    plt.ylabel(r'$\mathregular{R_j/R_0}$', fontsize=18)
    plt.savefig('C:/Users/thano/Desktop/ComputerVisionMEW_Paper/Paper_Images/figure_GPs2.jpg', dpi=300,
                bbox_inches='tight')
    plt.show()
# main function for running pyfile
if __name__ == "__main__":
    create_GP()