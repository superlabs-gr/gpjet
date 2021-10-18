"""
Created on:     15.04.2021
Created by:     @thanos_oikon
Description:    Multifidelity modeling and regression on jet radius data, comparison with standart GP regression
"""

# --- importing libraries ---
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# importing Gaussian Processes library
import GPy
# importing emulator library to perform multifidelity modeling
from emukit import multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

# --- creating proper matplotlib style ---

mpl.rcParams['font.family'] = 'Verdana'
mpl.rcParams['font.size'] = 18
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

def create_multi_fidelity_model():
    # load low fidelity data
    data_lf = np.loadtxt('ModelDiameters.csv', delimiter=',') # low fidelity data
    # load high fidelity data
    data_hf = np.loadtxt('Total_Diameters.csv', delimiter=',') # high fidelity data

    z_dist_plot = data_lf[0,:][:,None] # input values for plotting and predicting
    radius_plot_l = data_lf[1,:].reshape(-1,1) # output values of low fidelity for plotting and predicting
    #print(radius_plot_l.shape)
    z_dist_train_l = z_dist_plot[::4] # input values of low fidelity
    radius_train_l = radius_plot_l[::4] # output values of low fidelity
    #print(z_dist_train_l.shape, radius_train_l.shape)
    # just experimenting with different show cases of high fidelity datapoints chosen
    radius_plot_h = high_fidelity_data(data_hf).reshape(-1,1)  # output values of high fidelity for plotting and predicting
    z_dist_train_h = z_dist_plot[(0,5,20,40,60,80,93),:]
    radius_train_h = radius_plot_h[(0,5,20,40,60,80,93),:]



    # convert data to proper form for  multifidelity modelling
    Z_dist_train, Radius_train = convert_xy_lists_to_arrays([z_dist_train_l, z_dist_train_h],
                                                            [radius_train_l, radius_train_h])

    # create kernel for both models
    kernels = [GPy.kern.RBF(1, lengthscale=20, variance=1), GPy.kern.RBF(1, lengthscale=20, variance=1)]
    #kernels = [GPy.kern.Exponential(1, lengthscale=20, variance=0.01), GPy.kern.Exponential(1, lengthscale=20, variance=0.01)]
    # create the linear multifidelity kernel
    lin_mf_kernel = multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    # create the linear multifidelity model
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(Z_dist_train, Radius_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    # wrap the model using the given 'GPyMultiOutputWrapper'
    lin_mf_model = model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)
    # fit the model
    lin_mf_model.optimize()

    # convert x_plot to its ndarray representation
    Z_dist_l = np.concatenate([np.atleast_2d(z_dist_plot), np.zeros((z_dist_plot.shape[0], 1))], axis=1)
    Z_dist_h = np.concatenate([np.atleast_2d(z_dist_plot), np.ones((z_dist_plot.shape[0], 1))], axis=1)

    # compute mean predictions and associated variance
    lf_mean_lin_mf_model, lf_var_lin_mf_model = lin_mf_model.predict(Z_dist_l)
    lf_std_lin_mf_model = np.sqrt(lf_var_lin_mf_model)
    hf_mean_lin_mf_model, hf_var_lin_mf_model = lin_mf_model.predict(Z_dist_h)
    hf_std_lin_mf_model = np.sqrt(hf_var_lin_mf_model)

    # plot the posterior mean and variance
    plt.figure(figsize=(8, 6))
    plt.fill_between(z_dist_plot.flatten(), (lf_mean_lin_mf_model - 1.96 * lf_std_lin_mf_model).flatten(),
                     (lf_mean_lin_mf_model + 1.96 * lf_std_lin_mf_model).flatten(), facecolor='g', alpha=0.2)
    plt.fill_between(z_dist_plot.flatten(), (hf_mean_lin_mf_model - 1.96 * hf_std_lin_mf_model).flatten(),
                     (hf_mean_lin_mf_model + 1.96 * hf_std_lin_mf_model).flatten(), facecolor='c', alpha=0.3)

    plt.plot(z_dist_plot, radius_plot_l, 'b')
    plt.plot(z_dist_plot, radius_plot_h, 'r')
    plt.plot(z_dist_plot, lf_mean_lin_mf_model, '--', color='g')
    plt.plot(z_dist_plot, hf_mean_lin_mf_model, '--', color='c')
    plt.scatter(z_dist_train_l, radius_train_l, color='b', s=40)
    plt.scatter(z_dist_train_h, radius_train_h, color='r', s=40)
    plt.xlabel(r'$\mathregular{Z/R_o}$', fontsize=18)
    plt.ylabel(r'$\mathregular{R_j/R_o}$', fontsize=18)
    plt.legend(['Low Fidelity Physics Model Data', 'High Fidelity Observations', 'GP fit Low Fidelity', 'GP fit High Fidelity'])
    plt.savefig('C:/Users/thano/Desktop/ComputerVisionMEW_Paper/Paper_Images/figure_MFM_2a.jpg', dpi=300,
                bbox_inches='tight')
    plt.show()

    # create standard GP model using only high-fidelity data
    kernel = GPy.kern.RBF(1,lengthscale=20, variance=1)
    #kernel = GPy.kern.Exponential(1,lengthscale=20, variance=1)+GPy.kern.Linear(1)

    high_gp_model = GPy.models.GPRegression(z_dist_train_h, radius_train_h, kernel)
    high_gp_model.Gaussian_noise.fix(0)
    # Fit the GP model
    high_gp_model.optimize_restarts(5)
    # compute mean predictions and associated variance
    hf_mean_high_gp_model, hf_var_high_gp_model = high_gp_model.predict(z_dist_plot)
    hf_std_hf_gp_model = np.sqrt(hf_var_high_gp_model)

    # plot the posterior mean and variance for the high-fidelity GP model
    plt.figure(figsize=(8, 6))
    plt.fill_between(z_dist_plot.flatten(), (hf_mean_lin_mf_model - 1.96 * hf_std_lin_mf_model).flatten(),
                     (hf_mean_lin_mf_model + 1.96 * hf_std_lin_mf_model).flatten(), facecolor='c', alpha=0.3)
    plt.fill_between(z_dist_plot.flatten(), (hf_mean_high_gp_model - 1.96 * hf_std_hf_gp_model).flatten(),
                     (hf_mean_high_gp_model + 1.96 * hf_std_hf_gp_model).flatten(), facecolor='y', alpha=0.3)
    plt.plot(z_dist_plot, radius_plot_h, color='r')
    plt.plot(z_dist_plot, hf_mean_lin_mf_model, '--', color='c')
    plt.plot(z_dist_plot, hf_mean_high_gp_model, '--', color='y')
    plt.scatter(z_dist_train_h, radius_train_h, color='r', s=40)
    plt.xlabel(r'$\mathregular{Z/R_o}$', fontsize=18)
    plt.ylabel(r'$\mathregular{R_j/R_o}$', fontsize=18)
    plt.legend(['High Fidelity Observations', 'Linear Multi-fidelity GP fit', 'GP fit'])
    #plt.title('Comparison of linear multi-fidelity model and high fidelity GP')
    plt.savefig('C:/Users/thano/Desktop/ComputerVisionMEW_Paper/Paper_Images/figure_MFM_2b.jpg', dpi=300,
                bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    create_multi_fidelity_model()