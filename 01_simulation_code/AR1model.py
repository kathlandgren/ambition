"""
This module contains functions to simulate agent behaviors using an AR1 model with various 
information types and reward dynamics.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats as sps


def draw_sample(mu, sigma, skew=0, nsample=1):
    """
    Draw samples from a desired distribution, either normal or skew normal.

    Parameters:
    ----------
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation of the distribution.
    skew : float, optional
        Skewness parameter alpha for the skew-normal distribution (default is 0, which produces a normal distribution).
    nsample : int, optional
        Number of samples to draw (default is 1).

    Returns:
    -------
    sample : ndarray
        Array of sampled values.
    """
    if skew == 0:
        sample = np.random.normal(mu, sigma, nsample)
    else:
        delta = skew / np.sqrt(1 + skew**2)
        scale = np.sqrt(sigma**2 / (1 - (2 * delta**2 / np.pi)))
        loc = mu - scale * delta * np.sqrt(2 / np.pi)
        sample = sps.skewnorm.rvs(skew, loc, scale, size=nsample)

    return sample


def zscore(value, mu, sigma):
    """
    Compute the z-score of a value.

    Parameters:
    ----------
    value : float
        The value for which the z-score is calculated.
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation of the distribution.

    Returns:
    -------
    z : float
        The z-score.
    """
    return (value - mu) / sigma


def take_a_step(prevreward, corr, thresh, mu=0, sigma=1, skew=0, info=None, infosigma=None, forcemove=False):
    """
    Simulate an agent's step based on previous reward and information given.

    Parameters:
    ----------
    prevreward : float
        Reward at the previous step.
    corr : float
        Correlation between successive steps on the landscape.
    thresh : float
        Threshold to trigger movement.
    mu : float, optional
        Mean of the reward distribution (default is 0).
    sigma : float, optional
        Standard deviation of the reward distribution (default is 1).
    skew : float, optional
        Skewness of the reward distribution (default is 0).
    infotype : str, optional
        Type of information guiding the step: statistics about the whole landscape, cohort, or own history only.
    info : float, optional
        Information that the agent has about the mean of the reward distribution.
    infosigma : float, optional
        Information that the agent has about the standard deviation of the reward distribution.
    forcemove : bool, optional
        Force movement regardless of threshold (default is False).

    Returns:
    -------
    newreward : float
        Updated reward after taking a step.
    """
    if (prevreward < (info + thresh * infosigma)) or forcemove:
        newreward = corr * prevreward + (1 - corr) * draw_sample(mu, sigma, skew)
    else:
        newreward = prevreward
    return newreward


def agent_simulation(num_tstep, num_agents, thresh, corr, mu, sigma, infotype='landscape', skew=0, info=None, infosigma=None, infosummary='mean'):
    """
    Simulate agent behaviors over multiple time steps with specified parameters.

    Parameters:
    ----------
    num_tstep : int
        Number of time steps in the simulation.
    num_agents : int
        Number of agents in the simulation.
    thresh : float or array-like
        Threshold(s) for triggering movement.
    corr : float
        Correlation between successive steps on the landscape.
    mu : float
        Mean of the reward distribution.
    sigma : float
        Standard deviation of the reward distribution.
    skew : float, optional
        Skewness parameter alpha for the reward distribution.
    infotype : str, optional
        Type of information guiding the step: statistics about the whole landscape, cohort, or own history only.
    info : float, optional
        Information that the agent has about the mean of the reward distribution.
    infosigma : float, optional
        Information that the agent has about the standard deviation of the reward distribution.
    infosummary : str, optional
        Summary statistic for information ('mean' or 'median', default is 'mean').

    Returns:
    -------
    reward_history : ndarray
        Array of rewards for all agents across time steps.
    """
    assert infotype in ['landscape', 'group_hist', 'group_hist_upward', 'own_hist'], \
        "Invalid infotype. Must be one of 'landscape', 'group_hist', 'group_hist_upward', 'own_hist'."
    assert infosummary in ['mean', 'median'], "Invalid infosummary. Must be 'mean' or 'median'."

    reward_history = np.zeros((num_tstep, num_agents))
    reward_history[0, :] = draw_sample(mu, sigma, skew, nsample=num_agents)

    if infosummary == 'mean':
        info_fun = np.mean
    elif infosummary == 'median':
        info_fun = np.median

    if infotype == 'landscape':
        info = mu
        infosigma = sigma
        forcemove = False

    if isinstance(thresh, (int, float)):
        thresh = np.repeat(thresh, num_agents)

    for t in range(1, num_tstep):
        for a in range(num_agents):
            if infotype == 'group_hist':
                info = info_fun(reward_history[t - 1, np.arange(num_agents) != a])
                infosigma = np.std(reward_history[t - 1, np.arange(num_agents) != a])
                forcemove = False

            if infotype == 'group_hist_upward':
                info = info_fun(find_greater_elements(reward_history[t - 1, :], a))
                infosigma = np.std(find_greater_elements(reward_history[t - 1, :], a))
                forcemove = False

            if infotype == 'own_hist':
                info = info_fun(reward_history[:t, a])
                infosigma = np.std(reward_history[:t, a])
                forcemove = t == 1

            reward_history[t, a] = take_a_step(
                reward_history[t - 1, a], corr, thresh[a], mu, sigma, skew, info, infosigma, forcemove
            )

    return reward_history


def find_greater_elements(reward_hist_slice, a):
    """
    Identify elements in the reward history slice greater than or equal to a given agent's reward.

    Parameters:
    ----------
    reward_hist_slice : ndarray
        Slice of reward history (all agents at a specific time step).
    a : int
        Index of the agent for comparison.

    Returns:
    -------
    doing_better_than_a : list
        List of rewards greater than or equal to the specified agent's reward.
    """
    doing_better_than_a = [x for x in reward_hist_slice if x >= reward_hist_slice[a]]
    return doing_better_than_a if doing_better_than_a else [reward_hist_slice[a]]

