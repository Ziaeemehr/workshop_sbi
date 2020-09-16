import nest
import torch
import numpy as np
import matplotlib.pyplot as plt
from main import par_sim
from scipy.integrate import odeint
from scipy.stats import (kurtosis, skew)

# ------------------------------------------------------------------#


def simulation_wrapper(par):
    '''
    Returns summary statistics from conductance values in `par`.
    Summarizes the output of the HH simulator and converts it 
    to `torch.Tensor`.
    '''
    obs = HH_simulator(par_sim=par_sim, par_var=par)
    stats = torch.as_tensor(calculate_summary_statistics(obs))
    return stats
# ------------------------------------------------------------------#


def HH_simulator(par_sim, par_var):

    dt = par_sim['dt']
    t_on = par_sim['t_on']
    delay = par_sim['delay']
    i_app = float(par_sim['i_app'])  # pA
    t_simulation = par_sim['t_simulation']

    nest.ResetKernel()
    nest.SetKernelStatus(
        {'resolution': dt,
         "overwrite_files": True,
         "print_time": False,
         })
    nest.set_verbosity('M_WARNING')

    neuron = nest.Create('hh_psc_alpha_gap')

    parameters = nest.GetDefaults("hh_psc_alpha_gap")
    # for i in parameters:
    #     print(i, parameters[i])

    spikedet = nest.Create('spike_detector')
    dc = nest.Create('dc_generator')
    nest.SetStatus(spikedet, {'to_memory': False})

    nest.SetStatus(neuron, 'g_Na', float(par_var[0]))
    nest.SetStatus(neuron, 'g_Kv3', float(par_var[1]))
        

    multimeter = nest.Create('multimeter')
    nest.SetStatus(multimeter, {"withtime": True,
                                "record_from": ["V_m"],
                                'interval': dt})
    nest.SetStatus(dc, {
        'start': t_on,
        'amplitude': i_app,
        'stop': t_simulation
    })

    nest.Connect(neuron, spikedet, syn_spec={'weight': 1.0,
                                             'delay': delay})
    nest.Connect(multimeter, neuron)
    nest.Connect(dc, neuron)
    nest.Simulate(t_simulation)

    n_events = nest.GetStatus(spikedet, keys={'n_events'})[0][0]
    dmm = nest.GetStatus(multimeter)[0]
    Vms = dmm['events']['V_m']
    times = dmm['events']['times']

    return dict(
        time=times,
        dt=par_sim['dt'],
        n_events=n_events,
        data=np.asarray(Vms).reshape(-1),
        i_app=i_app,
        t_on=t_on,
    )
# ------------------------------------------------------------------#


def plot_data(monitor,
              ax=None,
              file_name='f',
              xlim=None,
              ylim=(100, 50)):

    V = monitor['data']
    times = monitor['time']

    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(7, 3))
        save_fig = True

    ax.plot(times, V, lw=1)

    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("V [mV]")
    ax.set_yticks(range(-100, 100, 50))
    plt.tight_layout()

    if save_fig:
        plt.savefig(file_name)
        plt.close()
# ------------------------------------------------------------------#


def calculate_summary_statistics(obs):
    """Calculate summary statistics

    Parameters
    ----------
    x : output of the simulator

    Returns
    -------
    np.array, summary statistics
    """

    n_momemt = 4
    n_summary = 7

    n_summary = np.minimum(n_summary, n_momemt + 3)

    t = obs["time"]
    dt = obs["dt"]
    I = obs['i_app']
    t_on = obs['t_on']
    num_spikes = obs['n_events']
    

    # initialise array of spike counts
    v = np.array(obs["data"])
    v_on = v[t > t_on]

    # spike_times = spike_detection(v, dt, spikeThreshold)
    # spike_times = spike_times[spike_times > t_on]

    # resting potential and std
    rest_pot = np.mean(v[t < t_on])
    rest_pot_std = np.std(v[int(0.9 * t_on / dt): int(t_on / dt)])

    v_mean = np.mean(v_on)
    v_std = np.std(v_on)
    v_kurtosis = kurtosis(v_on)
    v_skew = skew(v_on)

    sum_stats_vec = np.concatenate([[num_spikes, rest_pot, rest_pot_std,
                                     v_mean, v_std, v_skew, v_kurtosis]])
    sum_stats_vec = sum_stats_vec[0:n_summary]

    return sum_stats_vec
# ------------------------------------------------------------------#
