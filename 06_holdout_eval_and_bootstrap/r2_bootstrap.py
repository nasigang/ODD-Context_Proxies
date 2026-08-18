#!/usr/bin/env python3
"""
R2 Bootstrap — scenario-block paired bootstrap.

Delta_NLL = scenario_mean(NLL_M1 - NLL_M3)
  > 0  => M3 better (lower NLL; traffic context improves fit)
  = 0  => no improvement
  < 0  => M3 worse

This sign convention is used EVERYWHERE: config, evaluator, bootstrap, tables, plots.
"""
import numpy as np
from phase2_womd.r2_censored_likelihood import censored_lognormal_nll


DELTA_SIGN_CONVENTION = "Delta_NLL = NLL_M1 - NLL_M3; positive = M3 better (lower NLL)"


def scenario_block_paired_bootstrap(
    model_a, model_b,
    X_a, X_b,
    y_ttc, censored, scenario_ids,
    n_boot=1000, seed=42, alpha=0.05,
    censor_time=None,
):
    """
    Scenario-block paired bootstrap for Δ NLL.

    Delta = NLL(model_a) - NLL(model_b)
    Positive = model_b is better (lower NLL).

    Primary: model_a = M1, model_b = M3 → positive = M3 better.
    """
    rng = np.random.RandomState(seed)
    scenario_ids = np.asarray(scenario_ids)
    unique_sc = np.unique(scenario_ids)
    n_sc = len(unique_sc)

    # Per-sample NLL
    mu_a, sig_a = model_a.predict(X_a)
    mu_b, sig_b = model_b.predict(X_b)
    nll_a = censored_lognormal_nll(y_ttc, mu_a, sig_a, censored,
                                   censor_time=censor_time, reduction="none")
    nll_b = censored_lognormal_nll(y_ttc, mu_b, sig_b, censored,
                                   censor_time=censor_time, reduction="none")

    # Per-scenario mean NLL, then delta
    sc_nll_a = np.array([np.mean(nll_a[scenario_ids == s]) for s in unique_sc])
    sc_nll_b = np.array([np.mean(nll_b[scenario_ids == s]) for s in unique_sc])
    sc_delta = sc_nll_a - sc_nll_b   # positive = model_b better

    delta_point = float(np.mean(sc_delta))

    boot_deltas = []
    failed = 0
    for _ in range(n_boot):
        idx = rng.choice(n_sc, n_sc, replace=True)
        try:
            boot_deltas.append(float(np.mean(sc_delta[idx])))
        except Exception:
            failed += 1

    boot_deltas = np.array(boot_deltas)
    ci_lo = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))

    return {
        "model_a": model_a.name,
        "model_b": model_b.name,
        "sign_convention": DELTA_SIGN_CONVENTION,
        "delta_nll_mean": delta_point,
        "delta_nll_median": float(np.median(sc_delta)),
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "ci_method": "percentile",
        "ci_alpha": alpha,
        "n_scenarios": n_sc,
        "n_frames": len(y_ttc),
        "n_boot": n_boot,
        "n_failed": failed,
        "seed": seed,
        "boot_deltas": boot_deltas.tolist(),
    }


def compute_conditional_brier(model, X, y_ttc, censored, tau, censor_time=None):
    """
    Conditional Brier at threshold tau.

    Sample: exposed primary rows only (future_contact_event + right_censored,
    overlap_now=false).
    Prediction: F(tau | E_primary=1, C, Z). NOT multiplied by p_exposure.
    Observation: I(event AND TTC <= tau).

    If all censor_times >= tau, censored rows are known non-events before tau
    and naive binary Brier is valid. Otherwise requires IPCW.
    """
    p_pred = model.predict_exceedance(X, tau)

    if censor_time is not None:
        ct = np.asarray(censor_time, dtype=np.float64)
        all_ct_ge_tau = bool(np.all(ct >= tau))
    else:
        all_ct_ge_tau = True  # administrative censoring at 10s >= all tau <=5

    y_obs = ((~censored) & (y_ttc <= tau)).astype(float)
    brier = float(np.mean((p_pred - y_obs) ** 2))

    return {
        "tau": tau,
        "brier_score": brier,
        "n_samples": len(y_ttc),
        "n_events_at_tau": int(y_obs.sum()),
        "n_censored": int(censored.sum()),
        "mean_pred": float(np.mean(p_pred)),
        "event_rate_at_tau": float(y_obs.mean()),
        "calibration_type": "conditional",
        "p_exposure_applied": False,
        "censor_time_ge_tau": all_ct_ge_tau,
        "brier_method": "naive_binary" if all_ct_ge_tau else "IPCW_REQUIRED",
    }
