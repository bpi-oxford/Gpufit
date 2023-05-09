"""
Single Plane Illumination Autocorrelation function (SPIM-ACF) gpu fitting example

Requires custom edited pyGpufit spim_acf function(https://github.com/bpi-oxford/Gpufit), Numpy and Matplotlib.

"""

import numpy as np
import pygpufit.gpufit as gf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math
from scipy import special

PI= 3.14159
E = 2.71828

def generate_spim_acf(p,tau,a=0.24,sigma_xy=1.0,sigma_z=1.0):
    """
    Generates a SPIM ACF.
    http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d

    :param p: Parameters (diffusion coef D, number of particles N, convergence value of the ACF for long times G_inf)
    :param a: side length of square pixel in object space, in um
    :param sigma_xy: radius of psf in xy plane
    :param sigma_z: radius of psf in z plane
    :return: SPIM-ACF of the given parameters
    """
    D, N, G_inf = p

    arg_xy = D*tau + sigma_xy**2
    arg_z = 1.0 + D*tau/sigma_z**2
    prefix = 4.0*(a**2)*math.sqrt(PI)*N

    term_a = 2.0*a*special.erf(0.5*a*arg_xy**(-1/2))
    term_b = 4.0*np.sqrt(arg_xy)/np.sqrt(PI) * (np.exp(-1.0 * a**2/(4.0*arg_xy))-1.0)
    g_xy = term_a+term_b

    # print(prefix,term_a[0],term_b[0],g_xy[0])

    G = 1./prefix * (g_xy)**2 * np.power(arg_z,-0.5) + G_inf
    # print(G[0])
    return G

def main():
    # cuda available checks
    print('CUDA available: {}'.format(gf.cuda_available()))
    print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))

    D = 1e2
    N = 1.0
    G_inf = 1e-6

    # number of fits, number of points per fit
    number_fits = 50
    number_points = 50

    # model ID and number of parameter
    model_id = gf.ModelID.SPIM_ACF
    number_parameters = 3

    # true parameters
    true_parameters = np.array((D,N,G_inf), dtype=np.float32)

    # initialize random number generator
    np.random.seed(0)

    # initial parameters (relative randomized)
    initial_parameters = np.tile(true_parameters, (number_fits, 1))
    perturb = 0.01
    initial_parameters *= (1-perturb) + 2*perturb*np.random.rand(number_fits,3)
    
    # generate tau values
    tau = np.logspace(-5,3,number_points)

    # generate data
    data = generate_spim_acf(true_parameters,tau)
    data = np.tile(data, (number_fits,1))

    # add noise to data
    snr = 1e5
    noise_std_dev = 1.0 / (snr * np.log(10.0))
    noise = noise_std_dev * np.random.standard_normal(data.shape)
    data = data + noise
    data = data.astype(np.float32)

    # tolerance
    tolerance = 1e-9

    # maximum number of iterations
    max_number_iterations = 500

    # estimator ID
    estimator_id = gf.EstimatorID.LSE
    # estimator_id = gf.EstimatorID.MLE

    # run Gpufit
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit(
        data, None, model_id,
        initial_parameters,
        tolerance, max_number_iterations, None,
        estimator_id, None)

    print(initial_parameters)
    print(parameters.shape)

    # print fit results
    converged = states == 0
    print('***************** SPIM ACF Gpufit *****************')

    # print summary
    print('\nmodel ID:        {}'.format(model_id))
    print('number of fits:  {}'.format(number_fits))
    print('fit size:        {}'.format(number_points))
    print('mean chi_square: {:.2f}'.format(np.mean(chi_squares[converged])))
    print('iterations:      {:.2f}'.format(np.mean(number_iterations[converged])))
    print('time:            {:.2f} s'.format(execution_time))

    # get fit states
    number_converged = np.sum(converged)
    print('\nratio converged         {:6.2f} %'.format(number_converged / number_fits * 100))
    print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / number_fits * 100))
    print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / number_fits * 100))
    print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / number_fits * 100))

    # mean, std of fitted parameters
    converged_parameters = parameters[converged, :]
    converged_parameters_mean = np.mean(converged_parameters, axis=0)
    converged_parameters_std = np.std(converged_parameters, axis=0)
    print('\nparameters of SPIM ACF')
    for i in range(number_parameters):
        print('|p{} | true {:6.2E} | mean {:6.2E} | std {:6.2E}|'.format(i, true_parameters[i], converged_parameters_mean[i],
                                                                 converged_parameters_std[i]))
    
    data_fit = []
    for param in parameters:
        data_fit.append(generate_spim_acf(param,tau))
    data_fit = np.asarray(data_fit)

    # make a figure of function values
    fig, axs = plt.subplots(1,2)
    axs[0].plot(np.tile(tau, (number_fits,1)).T, data.T, 's', color=(0.5,0.5,0.5), markersize=4, linewidth=1, label="noisy data")
    axs[0].plot(np.tile(tau, (number_fits,1)).T, data_fit.T, '-', color=(0.0,0.0,0.0), markersize=4, linewidth=1, label="fit data")
    axs[0].set_xscale('log')
    axs[0].set_xlabel(r'$\tau[s]$')
    axs[0].set_ylabel(r'$G(\tau)$')
    axs[1].scatter(initial_parameters[:,0],parameters[:,0])
    axs[1].set_xlabel("Initial fitting diffusivity")
    axs[1].set_ylabel("Fitted diffusivity")

    # ax.legend()
    fig.tight_layout()
    save_path = "./spim_acf_fit.png"
    print("\nSaving plot...")
    fig.savefig(save_path)

    # plot(x2, gauss_final_fit, '--.y', 'MarkerSize', 8, 'LineWidth', 2);
    # hold on;
    # plot(x2, noisy_psf(:, 1), 'ks', 'MarkerSize', 8, 'LineWidth', 2);
    # plot(x2, initial_spline_fit,'--sg', 'MarkerSize', 8, 'LineWidth', 2);
    # plot(x2, final_fit_cpufit,'-xc', 'MarkerSize', 8, 'LineWidth', 2);
    # plot(x2, final_fit_gpufit,'--+b', 'MarkerSize', 8, 'LineWidth', 2);
    # plot(x2, spline_model, ':r', 'MarkerSize', 8, 'LineWidth', 1.5);
    # ylim([0, max(initial_spline_fit)]);
    print("Results plot save to {}".format(save_path))

if __name__=="__main__":
    main()