#!/usr/bin/env python3
"""
R2 Censored Likelihood — homoscedastic log-normal AFT.

log(T_i) = mu_i + sigma * epsilon_i,  epsilon ~ N(0,1), sigma > 0 common.

Two NLL variants (never mixed in one report):

  1. TTC-scale log-normal NLL (PRIMARY):
     event:     -log f_LN(T_i | mu_i, sigma)
              = log(sigma) + 0.5*log(2pi) + log(T_i) + 0.5*r_i^2
              where r_i = (log(T_i) - mu_i) / sigma
     censored:  -log S_LN(c_i | mu_i, sigma)  =  -Phi_bar_log((log(c_i) - mu_i) / sigma)

  2. Gaussian NLL on log(TTC):
     Same as (1) WITHOUT the log(T_i) Jacobian term.
     Use when comparing log-scale residuals only.

Primary reporting uses TTC-scale (1). Always labeled.

Mills ratio stability: uses log-domain computation via
  log_mills(z) = logpdf(z) - logsf(z)
then for h(z) = phi(z)/S(z) = exp(log_mills(z)).
For extreme z>30, uses asymptotic expansion h(z) ≈ z + 1/z.
No arbitrary clipping.
"""
import numpy as np
from scipy.stats import norm
from scipy.special import log_ndtr   # log(Phi(z)), numerically stable


TTC_FLOOR = 0.05
TTC_CAP = 10.0

NLL_LABELS = {
    "ttc_scale": "TTC-scale log-normal NLL (includes Jacobian log(T))",
    "log_scale": "Gaussian NLL on log(TTC) (no Jacobian)",
}


def log_transform_ttc(ttc, floor=TTC_FLOOR):
    """Transform TTC to log-space with floor."""
    return np.log(np.maximum(np.asarray(ttc, dtype=np.float64), floor))


def _stable_log_survival(z):
    """
    Stable computation of -log(S(z)) = -log(1 - Phi(z)) = -log(Phi(-z)).

    Uses log_ndtr(-z) which is stable for all z.
    """
    return -log_ndtr(-np.asarray(z, dtype=np.float64))


def _stable_mills_ratio(z):
    """
    h(z) = phi(z) / S(z) — stable computation.

    For |z| < 30: exp(logpdf(z) - logsf(z))
    For z >= 30: asymptotic h(z) ≈ z + 1/z - 1/z^3 + ...
    For z <= -30: near zero (S(z) ≈ 1, phi(z) ≈ 0)
    """
    z = np.asarray(z, dtype=np.float64)
    h = np.zeros_like(z)

    # Normal range
    normal = np.abs(z) < 30
    if normal.any():
        log_h = norm.logpdf(z[normal]) - norm.logsf(z[normal])
        h[normal] = np.exp(log_h)

    # Extreme positive z: asymptotic expansion
    big_pos = z >= 30
    if big_pos.any():
        zz = z[big_pos]
        h[big_pos] = zz + 1.0 / zz - 1.0 / (zz ** 3)

    # Extreme negative z: h ≈ 0 (phi decays exponentially, S → 1)
    # h[z <= -30] stays 0

    return h


def _stable_log_mills_ratio(z):
    """
    log(h(z)) = log(phi(z)/S(z)) = logpdf(z) - logsf(z).
    Stable via scipy for |z| < 30. Asymptotic for extreme z.
    """
    z = np.asarray(z, dtype=np.float64)
    result = np.full_like(z, -np.inf)  # default for z << 0

    normal = np.abs(z) < 30
    if normal.any():
        result[normal] = norm.logpdf(z[normal]) - norm.logsf(z[normal])

    big_pos = z >= 30
    if big_pos.any():
        zz = z[big_pos]
        # h(z) ≈ z for large z, so log(h) ≈ log(z)
        result[big_pos] = np.log(zz + 1.0 / zz)

    return result


def censored_lognormal_nll(y_ttc, mu, sigma, censored, censor_time=None,
                            floor=TTC_FLOOR, cap=TTC_CAP,
                            include_jacobian=True, reduction="mean"):
    """
    Censored log-normal NLL.

    Parameters
    ----------
    y_ttc : (n,) observed TTC in seconds
    mu : (n,) predicted mean of log(TTC)
    sigma : float > 0, common scale
    censored : (n,) bool
    censor_time : (n,) or None (defaults to cap for all)
    include_jacobian : bool
        True → TTC-scale log-normal NLL (PRIMARY)
        False → Gaussian NLL on log(TTC)
    """
    y_ttc = np.asarray(y_ttc, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = float(sigma)
    censored = np.asarray(censored, dtype=bool)
    n = len(y_ttc)
    if n == 0:
        return 0.0
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    y_log = log_transform_ttc(y_ttc, floor)
    if censor_time is not None:
        censor_log = log_transform_ttc(np.asarray(censor_time, dtype=np.float64), floor)
    else:
        censor_log = np.full(n, np.log(cap))

    nll = np.zeros(n, dtype=np.float64)
    event_mask = ~censored

    if event_mask.any():
        r = (y_log[event_mask] - mu[event_mask]) / sigma
        nll[event_mask] = np.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5 * r ** 2
        if include_jacobian:
            nll[event_mask] += y_log[event_mask]  # Jacobian: log(T_i)

    if censored.any():
        z = (censor_log[censored] - mu[censored]) / sigma
        nll[censored] = _stable_log_survival(z)

    if not np.all(np.isfinite(nll)):
        n_bad = int(np.sum(~np.isfinite(nll)))
        raise ValueError(f"Non-finite NLL: {n_bad}/{n} samples")

    if reduction == "mean":
        return float(np.mean(nll))
    elif reduction == "sum":
        return float(np.sum(nll))
    return nll


# Back-compat alias
censored_gaussian_nll = lambda y_ttc, mu, sigma, censored, **kw: \
    censored_lognormal_nll(y_ttc, mu, sigma, censored, include_jacobian=True, **kw)


def censored_nll_components(y_ttc, mu, sigma, censored, include_jacobian=True, **kw):
    """Return NLL with event/censored breakdown."""
    nll = censored_lognormal_nll(y_ttc, mu, sigma, censored,
                                  include_jacobian=include_jacobian,
                                  reduction="none", **kw)
    censored = np.asarray(censored, dtype=bool)
    label = NLL_LABELS["ttc_scale"] if include_jacobian else NLL_LABELS["log_scale"]
    return {
        "nll_per_sample": nll,
        "nll_event": float(np.mean(nll[~censored])) if (~censored).any() else np.nan,
        "nll_censored": float(np.mean(nll[censored])) if censored.any() else np.nan,
        "nll_total": float(np.mean(nll)),
        "n_event": int((~censored).sum()),
        "n_censored": int(censored.sum()),
        "n_total": len(nll),
        "nll_label": label,
        "includes_jacobian": include_jacobian,
    }


def predict_exceedance_prob(mu, sigma, tau, floor=TTC_FLOOR):
    """
    F(tau | E_primary=1, features) = Phi((log(tau) - mu) / sigma).
    Conditional prediction. NOT multiplied by p_exposure.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    log_tau = np.log(max(tau, floor))
    z = (log_tau - mu) / sigma
    return norm.cdf(z)


# ── Gradient for scipy optimizer ──

def nll_and_grad(params, X, y_log, censored, censor_log, include_jacobian=True):
    """
    NLL and gradient for scipy.optimize.minimize.

    params: [beta_0, beta_1, ..., beta_p, log_sigma]
    X: (n, p) feature matrix (NO intercept column)

    Returns (nll_mean, grad_mean).
    """
    p = X.shape[1]
    beta = params[:p + 1]
    log_sigma = params[p + 1]
    sigma = np.exp(log_sigma)
    n = len(y_log)

    mu = beta[0] + X @ beta[1:]

    event_mask = ~censored
    nll_total = 0.0
    grad_beta = np.zeros(p + 1)
    grad_log_sigma = 0.0

    # Event
    if event_mask.any():
        r_e = (y_log[event_mask] - mu[event_mask]) / sigma
        nll_event = np.sum(log_sigma + 0.5 * np.log(2 * np.pi) + 0.5 * r_e ** 2)
        if include_jacobian:
            nll_event += np.sum(y_log[event_mask])
        nll_total += nll_event

        # Gradient: d(event NLL)/d(mu) = -r/sigma (Jacobian term has no mu dep)
        dmu_e = -r_e / sigma
        # d(event NLL)/d(log_sigma) = 1 - r^2 (per sample)
        dls_e = 1.0 - r_e ** 2

        Xe = np.column_stack([np.ones(event_mask.sum()), X[event_mask]])
        grad_beta += Xe.T @ dmu_e
        grad_log_sigma += dls_e.sum()

    # Censored
    if censored.any():
        z_c = (censor_log[censored] - mu[censored]) / sigma
        log_surv = norm.logsf(z_c)
        nll_cens = -log_surv.sum()
        nll_total += nll_cens

        # Mills ratio h(z) = phi(z)/S(z) — stable computation
        h = _stable_mills_ratio(z_c)

        # dNLL/dz = h(z), dz/dmu = -1/sigma → dNLL/dmu = -h/sigma
        dmu_c = -h / sigma
        # dz/d(log_sigma) = -z → dNLL/d(log_sigma) = -h*z
        dls_c = -h * z_c

        Xc = np.column_stack([np.ones(censored.sum()), X[censored]])
        grad_beta += Xc.T @ dmu_c
        grad_log_sigma += dls_c.sum()

    if not np.isfinite(nll_total):
        raise ValueError(f"Non-finite NLL: {nll_total}")

    grad = np.concatenate([grad_beta, [grad_log_sigma]])
    if not np.all(np.isfinite(grad)):
        raise ValueError(f"Non-finite gradient: {np.sum(~np.isfinite(grad))} elements")

    return nll_total / n, grad / n
