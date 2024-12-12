
"""
This module sets up functions for working with normal and skew normal distributions 
and estimating cumulative rewards.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.special import erfc, erf
from scipy.stats import skew
import matplotlib
import pickle


def normal_pdf(x, mu=0, sigma=1):
    """
    Compute the probability density function (PDF) of a normal distribution.

    Parameters:
    - x: Value at which to evaluate the PDF.
    - mu: Mean of the normal distribution (default is 0).
    - sigma: Standard deviation of the normal distribution (default is 1).

    Returns:
    - The value of the normal PDF at x.
    """
    sqrt_two_pi = np.sqrt(2 * np.pi)
    return (1.0 / (sigma * sqrt_two_pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normal_cdf(x, mu=0, sigma=1):
    """
    Compute the cumulative distribution function (CDF) of a normal distribution.

    Parameters:
    - x: Value at which to evaluate the CDF.
    - mu: Mean of the normal distribution (default is 0).
    - sigma: Standard deviation of the normal distribution (default is 1).

    Returns:
    - The value of the normal CDF at x.
    """
    return 0.5 * (1 + math.erf((x - mu) / (sigma * np.sqrt(2))))

def estimated_rounds_to_threshold(x):
    """
    Estimate the number of rounds required to reach a threshold 
    based on the complement of the normal CDF.

    Parameters:
    - x: Input value to determine the threshold.

    Returns:
    - Estimated number of rounds required to reach the threshold.
    """
    return 1 / (1 - normal_cdf(x)) - 1

def old_estimated_cumulative_reward(x, tmax):
    """
    Compute an older version of the estimated cumulative reward 
    based on a threshold.

    Parameters:
    - x: Input value representing the threshold.
    - tmax: Maximum number of rounds.

    Returns:
    - The estimated cumulative reward, considering the threshold 
      and maximum number of rounds.
    """
    rounds = estimated_rounds_to_threshold(x)
    return min(rounds, tmax) * (-normal_pdf(x) / normal_cdf(x)) + \
           max(tmax - rounds, 0) * (normal_pdf(x) / (1 - normal_cdf(x)))

def scaling(corr):
    """
    Compute the reward scaling factor based on a correlation coefficient.

    Parameters:
    - corr: Correlation coefficient.

    Returns:
    - Scaling factor.
    """
    return np.sqrt((1 - corr) / (1 + corr))

def estimated_cumulative_reward(x, tmax):
    """
    Compute the estimated cumulative reward.

    Parameters:
    - x: Input value representing the threshold.
    - tmax: Maximum number of rounds.

    Returns:
    - The estimated cumulative reward.
    """
    term1 = 2**(-(1/2) - tmax) * np.exp(-(x**2 / 2))
    term2 = 2**(1 + tmax) * (-2 + tmax * erfc(x / np.sqrt(2)))
    term3 = (1 + erf(x / np.sqrt(2)))**tmax
    term4 = 4 + tmax * erfc(x / np.sqrt(2))**2
    denominator = np.sqrt(np.pi) * erfc(x / np.sqrt(2))**2

    return (term1 * (term2 + term3 * term4)) / denominator

def estimated_cumulative_reward_sigma(x, tmax, sigma):
    """
    Compute the estimated cumulative reward for a normal distribution 
    with an adjustable standard deviation (smoothness).

    Parameters:
    - x: Input value representing the threshold.
    - tmax: Maximum number of timesteps.
    - sigma: Standard deviation of the normal distribution.

    Returns:
    - The estimated cumulative reward considering the given standard 
      deviation (smoothness).
    """
    term1 = 2 ** (-(1/2) - tmax) * np.exp(-(x**2 / (2 * sigma**2)))
    term2 = 2 ** (1 + tmax) * sigma * (-2 + tmax * erfc(x / (np.sqrt(2) * sigma)))
    term3 = sigma * (1 + erf(x / (np.sqrt(2) * sigma)))**tmax
    term4 = 4 + tmax * erfc(x / (np.sqrt(2) * sigma))**2
    numerator = term1 * (term2 + term3 * term4)
    denominator = np.sqrt(np.pi) * erfc(x / (np.sqrt(2) * sigma))**2
    return numerator / denominator

def get_optimum_thresh_reward(looped_over_vec, simresults, threshold_vec):
    """
    Find the optimum threshold and corresponding reward among the rewards in a vector.

    Parameters:
    - looped_over_vec: An iterable representing the primary parameter over which the simulation results are computed, either smoothness or skew parameter alpha.
    - simresults: A 2D array where each column corresponds to the rewards for a given value in `looped_over_vec` 
                  and each row corresponds to a threshold value in `threshold_vec`.
    - threshold_vec: An array of threshold values used in the simulation.

    Returns:
    - optimum: A 1D array where each element corresponds to the threshold value that maximizes the reward for a given 
               value in `looped_over_vec`.
    - reward_at_optimum: A 1D array where each element corresponds to the maximum reward at the optimum threshold for 
                         a given value in `looped_over_vec`.
    """
    reward_at_optimum = np.zeros(len(looped_over_vec))
    optimum = np.zeros(len(looped_over_vec))
    for i in range(len(looped_over_vec)):
        optimum_idx = np.argmax(simresults[:, i])
        optimum[i] = threshold_vec[optimum_idx]
        reward_at_optimum[i] = simresults[optimum_idx, i]
    return optimum, reward_at_optimum

def calculate_skewness(alpha):
    """
    Calculate the skewness of a skew-normal distribution based on the shape parameter alpha.

    Parameters:
    - alpha: The shape parameter of the skew-normal distribution. A higher absolute value of `alpha` 
             indicates greater skewness.

    Returns:
    - gamma: The skewness of the skew-normal distribution.
    """
    delta = alpha / np.sqrt(1 + alpha**2)
    # Skewness formula for skew-normal distribution
    gamma = ((4 - np.pi) / 2) * (delta * np.sqrt(2 / np.pi))**3 / ((1 - (2 * (delta**2)) / np.pi)**(3 / 2))
    return gamma

def df_to_simresults(df, num_cols=11):
    """
    Convert a DataFrame into simulation result components.

    Parameters:
    - df: A pandas DataFrame containing the simulation data, with thresholds as columns 
          and parameter values as rows.
    - num_cols: The number of columns in the DataFrame to use for simulation results (default is 11).

    Returns:
    - threshold_vec: A list of threshold values extracted from the column headers of the DataFrame.
    - param_val: An array of parameter values (e.g., correlation or alpha values) extracted from the first row of the DataFrame.
    - simresults: A 2D array of simulation results, where rows correspond to thresholds and columns to parameters.
    """
    df = df.T
    threshold_strings = df[0].index.values[1:]
    threshold_vec = [float(x) for x in threshold_strings]
   
    # Extract correlation or alpha values
    param_val = df.iloc[0].values
    column_names = np.arange(0, num_cols)
    simresults = df[column_names].values[1:]

    return threshold_vec, param_val, simresults
