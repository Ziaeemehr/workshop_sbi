from sbi.inference.base import infer
import sbi.utils as utils
import pylab as plt
import numpy as np
import torch
import lib


sim_params = {
    'dt': 0.01,
    'v0': -70.0,
    'g_l': 0.3,
    # 'g_k': 36.0,
    # 'g_na': 120.0,
    'i_app': 7.0,
    't_on': 50.0,
    't_final': 200.0,
    'spikeThreshold': -50.0,
}


def test_example():
    '''
    simulate and plot the voltage for 3 values of g_k and g_na
    '''

    # g_na, g_k
    par = np.array([[110., 30.], [50., 15.], [10., 5.]])

    fig, ax = plt.subplots(1, figsize=(7, 3))
    for i in range(len(par)):
        sim = lib.HH_simulator(params=sim_params, par_var=par[i])
        samples = sim['data']
        time = sim['time']

        states = lib.calculate_summary_statistics(sim)
        lib.plot_data(time, samples, ax=ax)
        # print(states)
    plt.savefig('example.png')
    plt.show()
# ------------------------------------------------------------------#


def main():

    # Prior over model parameters
    prior_min = [10, 1.]
    prior_max = [150., 50.]
    prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                        high=torch.as_tensor(prior_max))
    # Inference
    posterior = infer(lib.simulation_wrapper,
                      prior,
                      method='SNPE',
                      num_simulations=300,
                      num_workers=8)

    # get observed data
    true_params = np.array([120.0, 36.0])
    labels_params = [r'$g_{Na}$', r'$g_{K}$']
    observation_trace = lib.HH_simulator(sim_params, true_params)
    obs_stats = lib.calculate_summary_statistics(observation_trace)
    lib.plot_data(observation_trace['time'],
                  observation_trace['data'])

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


if __name__ == "__main__":

    # test_example()
    main()
