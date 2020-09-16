import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import sbi.utils as utils
from sbi.inference.base import infer
from HH_helper_functions import HHsimulator
from HH_helper_functions import syn_current
from HH_helper_functions import calculate_summary_statistics


# remove top and right axis from plots
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

I, t_on, t_off, dt, t, A_soma = syn_current()


def run_HH_model(params):

    params = np.asarray(params)

    # input current, time step
    I, t_on, t_off, dt, t, A_soma = syn_current()

    t = np.arange(0, len(I), 1)*dt

    # initial voltage
    V0 = -70

    states = HHsimulator(V0, params.reshape(1, -1), dt, t, I, seed=0)
    # print("states", states.shape)
    # print(states.reshape(-1).shape)

    return dict(data=states.reshape(-1), time=t, dt=dt, I=I.reshape(-1))


def simulation_wrapper(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    obs = run_HH_model(params)
    summstats = torch.as_tensor(calculate_summary_statistics(obs))
    return summstats


prior_min = [.5, 1e-4]
prior_max = [80., 15.]
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                    high=torch.as_tensor(prior_max))
posterior = infer(simulation_wrapper,
                  prior,
                  method='SNPE',
                  num_simulations=300,
                  num_workers=4)


# true parameters and respective labels
true_params = np.array([50., 5.])
labels_params = [r'$g_{Na}$', r'$g_{K}$']

observation_trace = run_HH_model(true_params)
obs_sum_stat = calculate_summary_statistics(observation_trace)

samples = posterior.sample((10000,), x=obs_sum_stat)
print(type(samples))
print(samples.shape)
print(type(samples[0]))
print(samples[0].shape)
print(samples[0])


fig, axes = utils.pairplot(samples,
                           limits=[[.5, 80], [1e-4, 15.]],
                           ticks=[[.5, 80], [1e-4, 15.]],
                           fig_size=(5, 5),
                           points=true_params,
                           points_offdiag={'markersize': 6},
                           points_colors='r')
plt.show()
