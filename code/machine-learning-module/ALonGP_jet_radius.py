"""
Created on: 06.10.2021
Created by: @thanos_oikon
description: ...
"""
# importing libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
np.random.seed(12345)

# importing Gaussian Processes library
import GPy
from GPy.models import GPRegression

# importing active learning library
import emukit
# from emukit import parameter space variable
from emukit.core import ParameterSpace, ContinuousParameter, DiscreteParameter,InformationSourceParameter
from emukit.core.loop.loop_state import create_loop_state
# import optimization loop and acquisition function
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper


from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound,ExpectedImprovement, MultipointExpectedImprovement,MaxValueEntropySearch,ProbabilityOfImprovement,ProbabilityOfFeasibility
from emukit.bayesian_optimization.acquisitions.entropy_search import MultiInformationSourceEntropySearch
from emukit.experimental_design.acquisitions import ModelVariance, IntegratedVarianceReduction
from emukit.core.acquisition import Acquisition

# import multifidelity functions
from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.model_wrappers import GPyMultiOutputWrapper

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



def active_learning_on_GPs():
    FIG_SIZE = (8, 6)  # figsize to plot consistently
    max_iter = 110  # bayesian optimization loop parameter
    beta = 0.1  # tradeoff parameter for NCLB acq. opt.
    update_interval = 1  # how many results before running hyperparam. opt.
    batch_size = 1  # how many points to return as candidate points to evaluate at each iteration

    # load low fidelity data
    data_lf = np.loadtxt('ModelDiameters.csv', delimiter=',')
    # load high fidelity data
    data_hf = np.loadtxt('Total_Diameters.csv', delimiter=',')
    # define plotting variables
    x_plot = data_lf[0, :][:, None]
    print(x_plot.shape)
    y_plot_high = high_fidelity_data(data_hf).reshape(-1, 1)
    # plot available data
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x_plot, y_plot_high, 'r')
    plt.legend(['High fidelity'])

    plt.title('High fidelity Data')
    plt.xlabel(r'$\mathregular{Z/R_o}$')
    plt.ylabel(r'$\mathregular{R_j/R_o}$')
    plt.show()
    n_fidelities = 2
    parameter_space = ParameterSpace([ContinuousParameter('x', 0, np.max(x_plot))])

    # parameter_space = ParameterSpace([DiscreteParameter('x', x_plot.flatten()),
    #                                  InformationSourceParameter(n_fidelities)])
    x_high = x_plot[(0,3,6,93),:]
    y_high = y_plot_high[(0,3,6,93),:]

    kern_high = GPy.kern.RBF(1, lengthscale=20, variance=1)

    # build GP model
    gpy_model = GPy.models.GPRegression(x_high, y_high, kern_high, noise_var=1e-6)
    # build emulation model
    model = GPyModelWrapper(gpy_model)

    model.optimize()
    x_plot_high = x_plot
    def plot_model(x_high, y_high, i):
        mean_high, var_high = model.predict(x_plot_high)

        plt.figure(figsize=FIG_SIZE)
        plt.plot(x_plot, y_plot_high, 'r')
        plt.scatter(x_high, y_high, color='r')
        def plot_with_error_bars(x, mean, var, color):
            plt.plot(x, mean, '--', color=color)
            plt.fill_between(x.flatten(), mean.flatten() - 1.96 * np.sqrt(var).flatten(),
                             mean.flatten() + 1.96 * np.sqrt(var).flatten(),
                             alpha=0.3, color=color)

        plot_with_error_bars(x_plot_high[:, 0], mean_high, var_high, 'y')

        plt.legend(['High Fidelity Observations','GP fit'])
        plt.xlabel(r'$\mathregular{Z/R_o}$')
        plt.ylabel(r'$\mathregular{R_j/R_o}$')
        plt.savefig('figure_simple_active_learning_er_iter' + str(i) + '.png', dpi=300,
                    bbox_inches='tight')
        plt.show()

    plot_model(x_high, y_high,0)
    # define acquisition function
    acquisition = ModelVariance(model)

    def plot_acquisition(loop, x_new, i):
        colours = ['b', 'r']
        plt.plot(x_plot_high[:, 0], loop.candidate_point_calculator.acquisition.evaluate(x_plot_high), 'r')

        previous_x_collected = x_new

        fidelity_idx = 1
        plt.scatter(previous_x_collected[0, 0],
                    loop.candidate_point_calculator.acquisition.evaluate(previous_x_collected),
                    color=colours[fidelity_idx])
        #plt.title('Acquisition Function at Iteration ' + str(loop_state.iteration))
        plt.xlabel(r'$\mathregular{z/R_0}$')

        plt.ylabel(r'$\mathregular{Acquisition Value}$')
        plt.tight_layout()
        plt.savefig('figure_simple_active_learning_acq_iter' + str(i) + '.png', dpi=300,
                    bbox_inches='tight')
        plt.show()

    # create outer loop
    initial_loop_state = create_loop_state(x_high, y_high)
    acquisition_optimizer =  optimizer = GradientAcquisitionOptimizer(parameter_space) # build optimizer

    bo_loop = BayesianOptimizationLoop(
        model=model,
        space=parameter_space,
        acquisition=acquisition,
        acquisition_optimizer=acquisition_optimizer,
        update_interval=update_interval,
        batch_size=batch_size,
    )
    # Run BO loop

    results = None
    n = 1 # bo_loop.model.X.shape[0] uncomment to count number of points evaluated instead of iterationsu
    while n < max_iter:
        print(f"Optimizing: n={n}")

        # run prediction
        mean_hf_model, var_hf_model = model.predict(x_plot_high)
        std_hf_model = np.sqrt(var_hf_model)

        #print(mean_hf_model,mean_lf_model)
        x_batch = bo_loop.get_next_points(results) # pick next point to evaluate
        plot_acquisition(bo_loop, x_batch, n)
        for i in range(0,x_plot.shape[0]): # pick next point from available dataset closest to the proposed one and add to the training set
            if abs(x_batch[0][0] - x_plot[i]) < 0.18:
                print(i)
                x_high = np.vstack((x_high,x_plot[i]))
                y_high = np.vstack((y_high, y_plot_high[i]))
                break
        # if np.max(np.diag(var_hf_model))<0.001:  #uncomment to break when variance bellow a certain value
        #    break
        model.set_data(x_high, y_high)
        plot_model(x_high,y_high,n)

        n+=1

if __name__=="__main__":
    active_learning_on_GPs()
