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



def active_learning_on_GPs():
    FIG_SIZE = (8, 6)  # figsize to plot consistently
    max_iter = 110  # bayesian optimization loop parameter
    beta = 0.1  # tradeoff parameter for NCLB acq. opt.
    update_interval = 1  # how many results before running hyperparam. opt.
    batch_size = 1 # how many points to return as candidate points to evaluate at each iteration
    # load data
    data = np.loadtxt('GPs_Dataset.csv',skiprows=1,delimiter=',',usecols=(1, 2, 3, 4, 5, 6))
    # print(data)
    speed = data[:,0].reshape(-1,1)
    x_h = data[:,3].reshape(-1,1)
    y_h = data[:,4].reshape(-1,1)
    std_y = data[:,5]
    print(x_h.shape)
    # plot data
    plt.figure(figsize=FIG_SIZE)
    plt.plot(x_h, y_h, 'ro')
    plt.legend(['High fidelity'])

    plt.title('High fidelity Data')
    plt.xlabel(r'$\mathregular{U_c/V_{jm}}$')
    plt.ylabel(r'$\mathregular{L_j[mm]}$')
    plt.show()
    n_fidelities = 2
    x_plot = np.linspace(np.min(x_h), np.max(x_h), 100).reshape(-1,1)

    parameter_space = ParameterSpace([ContinuousParameter('x', np.min(x_plot), np.max(x_plot))])

    # parameter_space = ParameterSpace([DiscreteParameter('x', x_plot.flatten()),
    #                                  InformationSourceParameter(n_fidelities)])
    x_high = x_h[(5,),:]
    y_high = y_h[(5,),:]
    speed_high = speed[(5,),:]
    copy_x = x_h
    copy_y = y_h
    copy_speed = speed

    copy_x = np.delete(copy_x,5)
    copy_y = np.delete(copy_y,5)
    copy_speed = np.delete(copy_speed, 5)
    # define kernels
    kern_high = GPy.kern.RBF(1, lengthscale=10, variance=1)

    # build GP model
    gpy_model = GPy.models.GPRegression(x_high, y_high, kern_high)
    gpy_model.Gaussian_noise.fix(1e-5)  # add noise
    gpy_model.optimize_restarts(5, robust=True)


    # build emulation model
    model = GPyModelWrapper(gpy_model)

    model.optimize()
    x_plot_high = x_plot
    def plot_model(x_high, y_high, i):
        mean_high, var_high = model.predict(x_plot_high)

        plt.figure(figsize=FIG_SIZE)
        #plt.plot(x_plot, y_plot_high, 'r')
        plt.scatter(x_high, y_high, color='r')
        def plot_with_error_bars(x, mean, var, color):
            plt.plot(x, mean, '--', color=color)
            plt.fill_between(x.flatten(), mean.flatten() - 1.96 * np.sqrt(var).flatten(),
                             mean.flatten() + 1.96 * np.sqrt(var).flatten(),
                             alpha=0.3, color=color)

        plot_with_error_bars(x_plot_high[:, 0], mean_high, var_high, 'y')

        plt.legend(['GP fit', 'Observations'],loc='lower right')
        #plt.title('Low and High Fidelity Models')
        plt.xlabel(r'$\mathregular{U_c/V_{jm}}$')
        plt.ylabel(r'$\mathregular{L_j[mm]}$')
        plt.savefig('figure_simple_active_learning_er_iter' + str(i) + '.png', dpi=300,
                    bbox_inches='tight')
        plt.show()

    plot_model(x_high, y_high,0)


    # choose acquisition function to use
    acquisition = ModelVariance(model) # for purely exploratory purposes
    #acquisition = ExpectedImprovement(model) # to perform true bayesian optimization, to find minimum as fast as possible
    #acquisition = ProbabilityOfImprovement(model) # to perform true bayesian optimization, to find minimum as fast as possible, not as good as EI

    def plot_acquisition(loop, x_new,i):
        colours = ['b', 'r']
        plt.plot(x_plot_high[:, 0], loop.candidate_point_calculator.acquisition.evaluate(x_plot_high), 'r')

        previous_x_collected = x_new
        fidelity_idx = 1
        plt.scatter(previous_x_collected[0, 0],
                    loop.candidate_point_calculator.acquisition.evaluate(previous_x_collected),
                    color=colours[fidelity_idx])
        #plt.title('Acquisition Function at Iteration ' + str(loop_state.iteration))
        plt.xlabel(r'$\mathregular{speed ratio}$')

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
    n = 1 # bo_loop.model.X.shape[0] uncomment to count number of points evaluated instead of iterations
    while n < max_iter:
        print(f"Optimizing: n={n}")


        mean_hf_model, var_hf_model = model.predict(x_plot_high)
        std_hf_model = np.sqrt(var_hf_model)

        #print(mean_hf_model,mean_lf_model)
        x_batch = bo_loop.get_next_points(results)
        plot_acquisition(bo_loop, x_batch, n)
        index=0
        min=np.abs(x_batch[0][0]- copy_x[0])
        for i in range(0,copy_x.shape[0]): # pick next point from available dataset closest to the proposed one and add to the training set

            if abs(x_batch[0][0] - copy_x[i]) < min:
                index = i
                min = np.abs(x_batch[0][0] - copy_x[i])
        #if np.max(np.diag(var_hf_model))<0.001: #uncomment to break when variance bellow a certain value
        #    break
        #if np.min(x_high)<0.5: #uncomment to break when point bellow a certain value is found
        #    break
        x_high = np.vstack((x_high,copy_x[index]))
        y_high = np.vstack((y_high, copy_y[index]))
        print(copy_speed[index])
        copy_x = np.delete(copy_x,index)
        copy_y = np.delete(copy_y,index)
        copy_speed = np.delete(copy_speed,index)
        model.set_data(x_high, y_high)
        plot_model(x_high,y_high,n)

        n += 1

if __name__ == "__main__":
    active_learning_on_GPs()