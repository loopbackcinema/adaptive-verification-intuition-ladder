"""
H4 test only: does routing gain (vs B3) decline monotonically, and become non-positive
at the most severe level, as the cue's own signal degrades post-deployment?
Fixed cell per PREREGISTRATION.md: linear generator, w_conf=1.0, cue a=2.0 (L4), logreg gate.
Degradation: cue noise std multiplied by k in {1, 1.5, 2, 3, 5} (no refit).

Usage: python run_h4_degradation.py <output_dir> [--seeds 20]
"""
import argparse, json, zlib
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--seeds", type=int, default=20)
args = ap.parse_args()

MASTER_SEED = 20261001
W_LATENT = 1.25
W_CONF = 1.0
A_CUE = 2.0
N = {"train": 12000, "val": 6000, "test": 6000}
SEVERITIES = [1.0, 1.5, 2.0, 3.0, 5.0]
PRIMARY_BUDGET = 0.20


def _sq(*parts):
    ints = []
    for p in parts:
        if isinstance(p, str):
            ints.append(zlib.crc32(p.encode()))
        elif isinstance(p, float):
            ints.append(int(round(p * 1000)))
        else:
            ints.append(int(p))
    return np.random.SeedSequence(ints)


_ref_rng = np.random.default_rng(_sq(MASTER_SEED, "ref"))
_Xr = _ref_rng.normal(size=(200000, 8))
_dr = 0.90 * _Xr[:, 4] - 0.70 * _Xr[:, 5] + 0.45 * _Xr[:, 6] + 0.25 * _Xr[:, 7] + _ref_rng.normal(0, 0.35, 200000)
_etar = 1.8 * _Xr[:, 0] - 1.4 * _Xr[:, 1] + 1.1 * _Xr[:, 2] - 0.9 * _Xr[:, 3]
D_MEAN, D_STD = float(_dr.mean()), float(_dr.std())
M_MEAN, M_STD = float(np.abs(_etar).mean()), float(np.abs(_etar).std())
TARGET_FLIP_RATE = float((1 / (1 + np.exp(-(-0.9 + W_LATENT * (_dr - D_MEAN) / D_STD)))).mean())


def flip_rate(b0, w_conf):
    d_z = (_dr - D_MEAN) / D_STD
    m = np.abs(_etar)
    u_z = -(m - M_MEAN) / M_STD
    lin = w_conf * u_z + W_LATENT * d_z
    return float((1 / (1 + np.exp(-(b0 + lin)))).mean()) - TARGET_FLIP_RATE


B0 = brentq(flip_rate, -6.0, 6.0, args=(W_CONF,))


def make_data(rng, n):
    X = rng.normal(size=(n, 8))
    d0 = 0.90 * X[:, 4] - 0.70 * X[:, 5] + 0.45 * X[:, 6] + 0.25 * X[:, 7]
    eta = 1.8 * X[:, 0] - 1.4 * X[:, 1] + 1.1 * X[:, 2] - 0.9 * X[:, 3]
    y_fast = (eta >= 0).astype(int)
    d = d0 + rng.normal(0, 0.35, n)
    d_z = (d - D_MEAN) / D_STD
    m = np.abs(eta)
    u_z = -(m - M_MEAN) / M_STD
    p_flip = 1.0 / (1.0 + np.exp(-(B0 + W_CONF * u_z + W_LATENT * d_z)))
    y = np.where(rng.binomial(1, p_flip), 1 - y_fast, y_fast)
    return X, y, d


rows = []
for seed in range(1, args.seeds + 1):
    ss = _sq(MASTER_SEED, "h4", seed)
    r_tr, r_v, r_t, r_cue = [np.random.default_rng(s) for s in ss.spawn(4)]
    Xtr, ytr, dtr = make_data(r_tr, N["train"])
    Xv, yv, dv = make_data(r_v, N["val"])
    Xt, yt, dt = make_data(r_t, N["test"])

    fast = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr[:, :4], ytr)
    # note: fast model trained against flip-affected y here only for expedience differs slightly
    # from main pipeline (which trains fast model on noiseless y_fast); to stay faithful, refit
    # on the noiseless fast target instead:
    eta_tr = 1.8 * Xtr[:, 0] - 1.4 * Xtr[:, 1] + 1.1 * Xtr[:, 2] - 0.9 * Xtr[:, 3]
    y_fast_tr = (eta_tr >= 0).astype(int)
    fast = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr[:, :4], y_fast_tr)

    def outputs(X, y):
        p = fast.predict_proba(X[:, :4])[:, 1]
        pred = (p >= 0.5).astype(int)
        margin = np.abs(np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9)))
        err = (pred != y).astype(int)
        return pred, margin, err

    pred_tr, margin_tr, err_tr = outputs(Xtr, ytr)
    pred_t, margin_t, err_t = outputs(Xt, yt)
    m_mu, m_sd = float(margin_tr.mean()), float(margin_tr.std() + 1e-12)
    conf_tr = -(margin_tr - m_mu) / m_sd
    conf_t = -(margin_t - m_mu) / m_sd

    D_MEAN_S, D_STD_S = float(dtr.mean()), float(dtr.std() + 1e-12)

    # B3 baseline: learned error predictor on fast model's own inputs
    b3_tr = np.c_[Xtr[:, :4], conf_tr]
    b3_t = np.c_[Xt[:, :4], conf_t]
    b3_model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                               random_state=seed).fit(b3_tr, err_tr)
    s_b3_t = b3_model.predict_proba(b3_t)[:, 1]

    def cue_of(d, noise_mult, rng):
        d_z = (d - D_MEAN_S) / D_STD_S
        return (A_CUE * d_z + noise_mult * rng.normal(size=len(d))) / np.sqrt(A_CUE ** 2 + 1.0)

    # gate fit ONCE at nominal severity (no refit under shift, per prereg)
    q_tr = cue_of(dtr, 1.0, r_cue)
    Z_tr = np.c_[b3_tr, q_tr]
    gate = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Z_tr, err_tr)

    k_seed = np.random.default_rng(_sq(MASTER_SEED, "h4cue", seed))

    def budget_acc(pred, y, s, budget):
        k = int(round(len(s) * budget))
        idx = np.argsort(-s, kind="mergesort")[:k]
        z = pred.copy()
        z[idx] = y[idx]
        return float(np.mean(z == y))

    acc_b3 = budget_acc(pred_t, yt, s_b3_t, PRIMARY_BUDGET)
    for sev in SEVERITIES:
        q_t_degraded = cue_of(dt, sev, k_seed)
        Z_t = np.c_[b3_t, q_t_degraded]
        s_gate_t = gate.predict_proba(Z_t)[:, 1]
        acc_gate = budget_acc(pred_t, yt, s_gate_t, PRIMARY_BUDGET)
        rows.append({"seed": seed, "severity": sev, "gain_vs_B3_pp": (acc_gate - acc_b3) * 100})
    print(f"seed {seed} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{args.out}/h4_degradation.csv", index=False)

rho, p_raw = spearmanr(df.severity, df.gain_vs_B3_pp)
most_severe = df[df.severity == max(SEVERITIES)].gain_vs_B3_pp.to_numpy()
rng = np.random.default_rng(7)
means = np.array([most_severe[rng.integers(0, len(most_severe), len(most_severe))].mean() for _ in range(5000)])
lo, hi = np.quantile(means, [0.025, 0.975])
support = bool(rho <= -0.80 and (lo <= 0 or hi <= 0))

result = {"spearman_severity_vs_gain": float(rho), "p_raw": float(p_raw),
          "most_severe_mean_pp": float(most_severe.mean()), "most_severe_ci95": [float(lo), float(hi)],
          "support": support}
json.dump(result, open(f"{args.out}/h4_result.json", "w"), indent=2)
print(result)
