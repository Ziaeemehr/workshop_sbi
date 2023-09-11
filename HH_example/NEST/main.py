from sbi.utils.user_input_checks import process_prior
from sbi.inference.base import infer
from multiprocessing import Pool
import sbi.utils as utils
import pylab as plt
import numpy as np
import torch
import lib


par_sim = {
    'dt': 0.05,
    'delay': 0.1,
    'i_app': 100.0,
    't_on': 150.0,
    't_simulation': 600.0,
}
# g_Na, g_K
par_var = np.array([[9000., 4500.],
                    [5000., 1500.],
                    [1500., 100.]])


def test_plot(par, filename="f.png"):

    fig, ax = plt.subplots(1, figsize=(7, 3))
    for i in range(len(par)):
        obs = lib.HH_simulator(par_sim=par_sim, par_var=par[i])
        # print(lib.calculate_summary_statistics(obs))
        lib.plot_data(obs, ax=ax)
    fig.savefig(filename)
    plt.close()


def run(n_simulations, n_workers=4):

    # Prior over model parameters
    prior_min = [1500., 100.]
    prior_max = [15000., 10000.]
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                        high=torch.as_tensor(prior_max))
    prior, _, _ = process_prior(prior)
    
    theta = prior.sample((n_simulations,)).numpy()
    with Pool(n_workers) as p:
        x = p.map(lib.simulation_wrapper, theta)

    x = np.array(x)

    x = torch.from_numpy(x).float()
    theta = torch.from_numpy(theta).float()
    x = x[:, 3].reshape(-1, 1)

    print(theta, x)

    # print(x.dtype, theta.dtype, x.shape, theta.shape)
    posterior = lib.train(prior, x, theta, num_threads=n_workers)




    # get observed data
    # true_params = np.array([9000.0, 4500.0])
    # labels_params = [r'$g_{Na}$', r'$g_{K}$']
    # observation_trace = lib.HH_simulator(par_sim, true_params)
    # obs_stats = lib.calculate_summary_statistics(observation_trace)
    # lib.plot_data(observation_trace)

    # # Analysis of the posterior given the observed data
    # samples = posterior.sample((10000,),
    #                            x=obs_stats)
    # fig, axes = utils.pairplot(samples,
    #                            limits=[[prior_min[0], prior_max[0]],
    #                                    [prior_min[1], prior_max[1]]],
    #                            ticks=[[prior_min[0], prior_max[0]],
    #                                   [prior_min[1], prior_max[1]]],
    #                            fig_size=(5, 5),
    #                            points=true_params,
    #                            points_offdiag={'markersize': 6},
    #                            points_colors='r')
    # fig.savefig("inference_HH.png")

    # torch.save(samples, 'samples.pt')

# ------------------------------------------------------------------#


if __name__ == "__main__":

    # run some test plot and visualize the membrane potential
    # test_plot(par_var)

    # run simulation wrapper for different parameter values
    # obs = lib.simulation_wrapper(par_var[0])
    # print(obs)

    # run simulation wrapper for different parameter values
    # for i in range(len(par_var)):
    #     obs = lib.simulation_wrapper(par_var[i])
    #     print(obs)

    # make the loop parallel
    # from multiprocessing import Pool
    # with Pool(4) as p:
    #     obs = p.map(lib.simulation_wrapper, par_var)
    # print(obs)

    # make loop parallel with joblib

    run(10)

    exit(0)
    
    # try:
    #     samples = torch.load("data/samples.pt")
    # else:
    #     print("no input file!")
    #     exit(0)
    # fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    # max_values = lib.plot_posterior(
    #     samples, ax, labels=[r'$g_{Na}$', r'$g_K$'])
    # fig.savefig("data/posterior.png")


    
    # par = np.array([[9000.0, 4500.0], max_values])
    # test_plot(par, file_name="data/compare.png")
