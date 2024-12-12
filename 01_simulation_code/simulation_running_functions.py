# This module contains the functions needed to run the simulation and save the output
import pandas as pd
import numpy as np
from datetime import datetime
import re
from joblib import Parallel, delayed
import multiprocessing
import AR1model


def create_potential_threshold_values(threshold_min, threshold_max, num_thresholds):
    """
    Generate a range of potential threshold values.

    Parameters:
    ----------
    threshold_min : float
        Minimum threshold value.
    threshold_max : float
        Maximum threshold value.
    num_thresholds : int
        Number of thresholds to generate.

    Returns:
    -------
    potential_threshold_values : ndarray
        Array of evenly spaced threshold values between the minimum and maximum.
    """
    potential_threshold_values = np.linspace(start=threshold_min, stop=threshold_max, num=num_thresholds)
    return potential_threshold_values



def assess_reward(rew_hist, reward_assessment, num_tstep):
    """
    Assess rewards based on the specified evaluation method.

    Parameters:
    ----------
    rew_hist : ndarray
        Reward history matrix with shape (num_tstep, num_agents).
    reward_assessment : str
        Method for assessing rewards ('last' or 'cumulative').
    num_tstep : int
        Number of time steps.

    Returns:
    -------
    output : ndarray
        Reward assessments for agents.
    """
    if reward_assessment == 'last':
        output = rew_hist[-1:, :]
    elif reward_assessment == 'cumulative':
        output = rew_hist.sum(axis=0) / num_tstep  # Normalize by time steps
    else:
        raise ValueError("The parameter `reward_assessment` must be either `last` or `cumulative`.")
    return output


def initialize_bins(potential_threshold_values):
    """
    Initialize bins for aggregating results.

    Parameters:
    ----------
    potential_threshold_values : ndarray
        Array of potential threshold values.

    Returns:
    -------
    agents_in_bin : ndarray
        Array to count agents in each bin.
    output_combined : ndarray
        Array to aggregate rewards in each bin.
    exploit_frac_combined : ndarray
        Array to aggregate exploitation fractions in each bin.
    """
    num_bins = len(potential_threshold_values) - 1
    agents_in_bin = np.zeros(num_bins)
    output_combined = np.zeros(num_bins)
    exploit_frac_combined = np.zeros(num_bins)
    return agents_in_bin, output_combined, exploit_frac_combined


def bin_rewards(output, threshold, potential_threshold_values, agents_in_bin, output_combined, exploit_frac_combined, mean_exploit_frac):
    """
    Bin rewards, thresholds, and exploitation fractions based on potential threshold values.

    Parameters:
    ----------
    output : ndarray
        Array of reward assessments.
    threshold : ndarray
        Array of threshold values for agents.
    potential_threshold_values : ndarray
        Array of potential threshold values.
    agents_in_bin : ndarray
        Array to count agents in each bin.
    output_combined : ndarray
        Array to aggregate rewards in each bin.
    exploit_frac_combined : ndarray
        Array to aggregate exploitation fractions in each bin.
    mean_exploit_frac : ndarray
        Array of mean exploitation fractions.

    Returns:
    -------
    agents_in_bin, output_combined, exploit_frac_combined : ndarrays
        Updated arrays after binning.
    """
    for i in range(len(output)):
        vals, bins = np.histogram(a=threshold[i], bins=potential_threshold_values)
        agents_in_bin += vals
        output_combined += vals * output[i]
        exploit_frac_combined += vals * mean_exploit_frac[i]
    return agents_in_bin, output_combined, exploit_frac_combined


def scale_by_agents_in_bin(to_scale, agents_in_bin):
    """
    Scale values by the number of agents in each bin.

    Parameters:
    ----------
    to_scale : ndarray
        Array of values to scale.
    agents_in_bin : ndarray
        Array of agent counts in each bin.

    Returns:
    -------
    scaled_values : ndarray
        Values scaled by the number of agents in each bin.
    """
    return to_scale / agents_in_bin



def compute_exploit_frac(num_tstep, num_agents, rew_hist):
    """
    Compute the fraction of agents exploiting rewards at each time step.

    Parameters:
    ----------
    num_tstep : int
        Number of time steps.
    num_agents : int
        Number of agents.
    rew_hist : ndarray
        Reward history matrix with shape (num_tstep, num_agents).

    Returns:
    -------
    exploit_frac : ndarray
        Fraction of agents exploiting at each time step.
    """
    exploit_frac = np.zeros((num_tstep - 1, num_agents))
    for a in range(num_agents):
        for t in range(num_tstep - 1):
            exploit_frac[t, a] = AR1model.find_exploiting_frac(rew_hist, t + 1)
    return exploit_frac

def generate_measurables(rew_hist, threshold, agents_in_bin, output_combined, exploit_frac_combined, num_tstep, num_agents, reward_assessment, potential_threshold_values):
    """
    Compute and bin measurable quantities.

    Parameters:
    ----------
    rew_hist : ndarray
        Reward history matrix.
    threshold : ndarray
        Array of agent thresholds.
    agents_in_bin, output_combined, exploit_frac_combined : ndarrays
        Initialized arrays for binning.
    num_tstep : int
        Number of time steps.
    num_agents : int
        Number of agents.
    reward_assessment : str
        Method for assessing rewards ('last' or 'cumulative').
    potential_threshold_values : ndarray
        Array of potential threshold values.

    Returns:
    -------
    Updated binning arrays.
    """
    exploit_frac = compute_exploit_frac(num_tstep, num_agents, rew_hist)
    mean_exploit_frac = np.mean(exploit_frac, axis=0)
    output = assess_reward(rew_hist, reward_assessment, num_tstep)
    agents_in_bin, output_combined, exploit_frac_combined = bin_rewards(
        output, threshold, potential_threshold_values, agents_in_bin, output_combined, exploit_frac_combined, mean_exploit_frac
    )
    return agents_in_bin, output_combined, exploit_frac_combined


def generate_corr_and_alpha(looped_over_param, corr_values, alpha_values, pl):
    """
    Generate correlation and skewness parameters for simulations.

    Parameters:
    ----------
    looped_over_param : str
        Parameter being looped over ('corr' or 'alpha').
    corr_values, alpha_values : ndarray
        Arrays of possible correlation and skewness parameter values.
    pl : int
        Index of the current parameter value.

    Returns:
    -------
    corr, alpha : float
        Correlation and skewness parameter alpha values for the current iteration.
    """
    if looped_over_param == 'corr':
        corr = corr_values[pl]
        alpha = 0
    elif looped_over_param == 'alpha':
        corr = 0
        alpha = alpha_values[pl]
    return corr, alpha
