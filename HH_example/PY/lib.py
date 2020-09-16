import scipy 
import torch
import numpy as np
import pylab as plt
from numpy import exp
from main import sim_params
from scipy.integrate import odeint
from scipy.stats import (kurtosis, skew)

# ------------------------------------------------------------------#


def simulation_wrapper(par):
    '''
    Returns summary statistics from conductance values in `par`.
    Summarizes the output of the HH simulator and converts it 
    to `torch.Tensor`.
    '''
    obs = HH_simulator(params=sim_params, par_var=par)
    stats = torch.as_tensor(calculate_summary_statistics(obs))
    return stats
# ------------------------------------------------------------------#


def beta_n(v): return 0.125 * exp(-(v + 70.0) / 80.0)
def beta_m(v): return 4.0 * exp(-(v + 70.0) / 18.0)
def beta_h(v): return 1. / (exp(-(v + 40.0) / 10.0) + 1.0)
def alpha_n(v): return 0.01 * (-60.0 - v) / (exp((-60.0 - v) / 10.0) - 1.0)
def alpha_m(v):
    if np.abs(v+45.0) > 1.0e-8:
        return (v + 45.0) / 10.0 / (1.0 - exp(-(v + 45.0) / 10.0))
    else:
        return 1.0
def alpha_h(v): return 0.07*exp(-(v+70)/20)
def h_inf(v): return alpha_h(v) / (alpha_h(v) + beta_h(v))
def m_inf(v): return alpha_m(v) / (alpha_m(v) + beta_m(v))
def n_inf(v): return alpha_n(v) / (alpha_n(v) + beta_n(v))

def i_ext(t, t_on, amp):
    I = amp if (t > t_on) else 0.0
    return I 
# ------------------------------------------------------------------#


def HH_simulator(params, par_var):

    dt = params['dt']
    v0 = params['v0']
    g_l = params['g_l']
    t_on = params['t_on']
    i_app = params['i_app']
    t_final = params['t_final']

    g_na, g_k = par_var

    v_l = -59.0
    v_k = -82.0
    v_na = 45.0
    c = 1.0

    initial_condition = np.asarray([v0, m_inf(v0), n_inf(v0), h_inf(v0)])

    def derivative(x0, t):
        '''
        define HH Model
        '''
        v, m, n, h, = x0
        dv = (i_ext(t, t_on, i_app) - g_na * h * m ** 3 *
              (v - v_na) - g_k * n ** 4 * (v - v_k) - g_l * (v - v_l)) / c
        dm = alpha_m(v) * (1.0 - m) - beta_m(v) * m
        dn = alpha_n(v) * (1.0 - n) - beta_n(v) * n
        dh = alpha_h(v) * (1.0 - h) - beta_h(v) * h

        return np.asarray([dv, dm, dn, dh])

    times = np.arange(0, t_final, dt)
    sol = odeint(derivative, initial_condition, times)
    sol = sol[:, 0].reshape(-1, 1)

    return dict(
        dt=dt,
        t_on=t_on,
        time=times,
        i_app=i_app,
        data=sol.reshape(-1),
        spikeThreshold=params['spikeThreshold'])
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
    spikeThreshold = obs['spikeThreshold']

    # initialise array of spike counts
    v = np.array(obs["data"])
    v_on = v[t > t_on]
    

    spike_times = spike_detection(v, dt, spikeThreshold)
    spike_times = spike_times[spike_times > t_on]
    num_spikes = len(spike_times)
    

    # resting potential and std
    rest_pot = np.mean(v[t < t_on])
    rest_pot_std = np.std(v[int(0.9 * t_on / dt) : int(t_on / dt)])

    v_mean = np.mean(v_on)
    v_std = np.std(v_on)
    v_kurtosis = kurtosis(v_on)
    v_skew = skew(v_on)

    sum_stats_vec = np.concatenate([[num_spikes, rest_pot, rest_pot_std,
                     v_mean, v_std, v_skew, v_kurtosis]])
    sum_stats_vec = sum_stats_vec[0:n_summary]

    return sum_stats_vec
# ------------------------------------------------------------------#


def plot_data(time, data,
              ylim=(-100, 50),
              xlim=None,
              ax=None,
              file_name="f"):
    
    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(7, 3))
        save_fig = True
        
    ax.plot(time, data, lw=1)
    
    if xlim is not None: ax.set_xlim(xlim)    
    ax.set_ylim(ylim)

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("v [mV]")
    ax.set_yticks(range(-100, 100, 50))
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(file_name)
        plt.close()
# ------------------------------------------------------------------#


def spike_detection(V, dt, spikeThreshold):
    
    # assert (len(V.shape) == 1)
    nSteps = len(V)
    nSpikes = 0
    tSpikes = []
    for i in range(1, nSteps):
        if (V[i - 1] <= spikeThreshold) & (V[i] > spikeThreshold):
            nSpikes += 1
            ts = ((i - 1) * dt * (V[i - 1] - spikeThreshold) +
                        i * dt * (spikeThreshold - V[i])) / (V[i - 1] - V[i])
            tSpikes.append(ts)
    return np.asarray(tSpikes)
# ------------------------------------------------------------------#



    

# if __name__ == "__main__":

    # t = np.arange(0, t_final, dt)
    # sol = odeint(derivative, x0, t)
    # v = sol[:, 0]

    # pl.figure(figsize=(7, 3))
    # pl.plot(t, v, lw=2, c="k")
    # pl.xlim(min(t), max(t))
    # pl.ylim(-100, 50)
    # pl.xlabel("time [ms]")
    # pl.ylabel("v [mV]")
    # pl.yticks(range(-100, 100, 50))
    # pl.tight_layout()
    # pl.savefig("fig_1_3.png")
    # pl.show()
