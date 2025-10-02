import numpy as np
from scipy.special import erf, erfc

# --- Core helpers ------------------------------------------------------------

def Sigma_from_phi(phi):
    """
    Mathematica: Sigma[phi_] = Sqrt[(1 - phi)/(1 + phi)]
    """
    phi = np.asarray(phi, dtype=float)
    return np.sqrt((1.0 - phi) / (1.0 + phi))

def stdnorm_pdf(z):
    # Standard normal pdf
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)

def stdnorm_cdf(z):
    # Standard normal cdf via erf (avoids importing scipy.stats)
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def truncated_tail_clip(p, eps=1e-12):
    # Avoid divisions by 0 in tails
    return np.clip(p, eps, 1.0 - eps)

# --- Mu1, Mu2, p (match your Mathematica definitions) ------------------------

def Mu1(x, sigma):
    """
    Mu1[x, σ] := -σ * φ(z) / Φ(z),  where z = x/σ
    """
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = x / sigma
    phi = stdnorm_pdf(z)
    Phi = truncated_tail_clip(stdnorm_cdf(z))
    return -sigma * (phi / Phi)

def Mu2(x, sigma):
    """
    Mu2[x, σ] :=  σ * φ(z) / (1 - Φ(z)), where z = x/σ
    """
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = x / sigma
    phi = stdnorm_pdf(z)
    Phi = truncated_tail_clip(stdnorm_cdf(z))
    return sigma * (phi / (1.0 - Phi))

def p_tail(x, sigma):
    """
    p[x, σ] = 1 - CDF_Normal(0, σ)(x) = 1 - Φ(x/σ)
    """
    z = np.asarray(x, dtype=float) / np.asarray(sigma, dtype=float)
    return 1.0 - stdnorm_cdf(z)

# --- Single-term summands (mirror your Summand / SummandWithCost) ------------

def summand(x, sigma, tmax, t):
    """
    (Mu1 * t + Mu2 * (tmax - t)) * (1 - p)^t * p
    """
    mu1 = Mu1(x, sigma)
    mu2 = Mu2(x, sigma)
    p = p_tail(x, sigma)
    one_minus_p = 1.0 - p  # = Φ(x/σ)
    return (mu1 * t + mu2 * (tmax - t)) * (one_minus_p**t) * p

def summand_with_cost(x, sigma, tmax, t, c):
    """
    (Mu1 * t + Mu2 * (tmax - t) - c * t) * (1 - p)^t * p
    """
    mu1 = Mu1(x, sigma)
    mu2 = Mu2(x, sigma)
    p = p_tail(x, sigma)
    one_minus_p = 1.0 - p
    return (mu1 * t + mu2 * (tmax - t) - c * t) * (one_minus_p**t) * p

# --- Vectorized cumulative sums over t = 1..tmax -----------------------------

def cumulative_sum_with_cost(x, tmax, sigma, c):
    """
    Returns sum_{t=1}^{tmax} SummandWithCost[x, sigma, tmax, t, c] for each x.
    x can be scalar or 1D array. Output matches shape of x.
    """
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    c = float(c)
    t = np.arange(1, tmax + 1, dtype=float)[:, None]
    vals = summand_with_cost(x[None, :], sigma, tmax, t, c)/tmax
    return np.sum(vals, axis=0)
