from sbi.inference.base import infer
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
par_var = np.array([[9000., 4500.], [5000., 1500.], [1500., 100.]])


def test_plot():
    
    fig, ax = plt.subplots(1, figsize=(7, 3))
    for i in range(len(par_var)):
        obs = lib.HH_simulator(par_sim=par_sim, par_var=par_var[i])
        print(lib.calculate_summary_statistics(obs))
        lib.plot_data(obs, ax=ax)
    fig.savefig("example.png")
    plt.close()

def main():

    # Prior over model parameters
    prior_min = [1500., 100.]
    prior_max = [15000., 10000.]
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                        high=torch.as_tensor(prior_max))
    # Inference
    posterior = infer(lib.simulation_wrapper,
                      prior,
                      method='SNPE',
                      num_simulations=300,
                      num_workers=4)

    # get observed data
    true_params = np.array([9000.0, 4500.0])
    labels_params = [r'$g_{Na}$', r'$g_{K}$']
    observation_trace = lib.HH_simulator(par_sim, true_params)
    obs_stats = lib.calculate_summary_statistics(observation_trace)
    lib.plot_data(observation_trace)

    # Analysis of the posterior given the observed data
    samples = posterior.sample((10000,),
                               x=obs_stats)
    fig, axes = utils.pairplot(samples,
                               limits=[[prior_min[0], prior_max[0]],
                                       [prior_min[1], prior_max[1]]],
                               ticks=[[prior_min[0], prior_max[0]],
                                      [prior_min[1], prior_max[1]]],
                               fig_size=(5, 5),
                               points=true_params,
                               points_offdiag={'markersize': 6},
                               points_colors='r')
    fig.savefig("inference_HH.png")

# ------------------------------------------------------------------#

if __name__ == "__main__":
    
    # test_plot()
    main()

