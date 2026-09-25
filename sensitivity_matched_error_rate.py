"""
Sensitivity check for the v8 grid: hold the fast-model error rate constant across w_conf.

In v8 the flip intercept B0 is fixed, so raising w_conf also raises the base error rate
(33.98% at w_conf = 0.0 to 40.00% at w_conf = 2.0). A reviewer can object that the cells
differ in task difficulty as well as in baseline informativeness. Here B0 is solved per
w_conf so that the expected flip rate matches the w_conf = 0.0 cell, and the primary
20% comparison is recomputed. Everything else is identical to v8.

Usage:  python sensitivity_matched_error_rate.py [output_dir]
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

OUT = sys.argv[1] if len(sys.argv) > 1 else "sensitivity_matched_error_rate"
os.makedirs(OUT, exist_ok=True)

SEED = 20260925
N = {"train": 12000, "val": 6000, "test": 6000, "ood": 6000}
LEVELS = {"L0": 0.0, "L1": 0.0, "L2": 0.45, "L3": 1.0, "L4": 2.0}
W_CONF = [0.0, 0.5, 1.1, 2.0]
W_LATENT = 1.25
B0_REF = -0.9
BUDGET = 0.20
BOOT = 5000


def draw(rng, n, ood):
    X = rng.normal(size=(n, 8))
    if ood:
        X[:, 4:] = X[:, 4:] * 1.35 + 0.45
        d = 0.65 * X[:, 4] - 0.55 * X[:, 5] + 0.35 * X[:, 6] + 0.20 * X[:, 7] + rng.normal(0, 0.50, n)
    else:
        d = 0.90 * X[:, 4] - 0.70 * X[:, 5] + 0.45 * X[:, 6] + 0.25 * X[:, 7] + rng.normal(0, 0.35, n)
    eta = 1.8 * X[:, 0] - 1.4 * X[:, 1] + 1.1 * X[:, 2] - 0.9 * X[:, 3]
    d_z = (d - d.mean()) / (d.std() + 1e-12)
    m = np.abs(eta)
    u_z = -(m - m.mean()) / (m.std() + 1e-12)
    return X, d, eta, d_z, u_z


def flip_rate(b0, w_conf, u_z, d_z):
    return float(np.mean(1.0 / (1.0 + np.exp(-(b0 + w_conf * u_z + W_LATENT * d_z)))))


def build(rng, n, w_conf, b0, ood=False):
    X, d, eta, d_z, u_z = draw(rng, n, ood)
    y_fast = (eta >= 0).astype(int)
    p_flip = 1.0 / (1.0 + np.exp(-(b0 + w_conf * u_z + W_LATENT * d_z)))
    y = np.where(rng.binomial(1, p_flip), 1 - y_fast, y_fast)
    return {"X": X, "y": y, "d": d, "y_fast": y_fast}


def outputs(fast, pack, mu=None, sd=None):
    p = fast.predict_proba(pack["X"][:, :4])[:, 1]
    pred = (p >= 0.5).astype(int)
    m = np.abs(np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9)))
    if mu is None:
        mu, sd = float(m.mean()), float(m.std() + 1e-12)
    return pred, -(m - mu) / sd, (pred != pack["y"]).astype(int), mu, sd


def cue(level, d, cfeat, d_mean, d_std, rng):
    if level == "L0":
        return rng.normal(size=len(d))
    if level == "L1":
        return cfeat.copy()
    a = LEVELS[level]
    return (a * (d - d_mean) / d_std + rng.normal(size=len(d))) / np.sqrt(a * a + 1.0)


def top(score, budget):
    return np.argsort(-score, kind="mergesort")[:int(round(len(score) * budget))]


def boot_ci(diff, rng, B=BOOT, chunk=500):
    n, means, done = len(diff), np.empty(B), 0
    while done < B:
        m = min(chunk, B - done)
        means[done:done + m] = rng.multinomial(n, np.full(n, 1.0 / n), size=m).dot(diff) / n
        done += m
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(diff)), float(lo), float(hi)


# reference flip rate from the w_conf = 0 cell
ref_rng = np.random.default_rng(np.random.SeedSequence([SEED, 0]).spawn(1)[0])
_, _, _, d_z0, u_z0 = draw(ref_rng, 200000, False)
TARGET = flip_rate(B0_REF, 0.0, u_z0, d_z0)

rows = []
for wi, w_conf in enumerate(W_CONF):
    b0 = B0_REF if w_conf == 0.0 else brentq(lambda b: flip_rate(b, w_conf, u_z0, d_z0) - TARGET, -5.0, 5.0)
    w_ss = np.random.SeedSequence([SEED, wi])
    r_tr, r_v, r_t, r_o, r_cue, _, r_boot = [np.random.default_rng(s) for s in w_ss.spawn(7)]
    tr, va, te, oo = (build(r_tr, N["train"], w_conf, b0), build(r_v, N["val"], w_conf, b0),
                      build(r_t, N["test"], w_conf, b0), build(r_o, N["ood"], w_conf, b0, ood=True))
    fast = LogisticRegression(max_iter=2000, random_state=SEED).fit(tr["X"][:, :4], tr["y_fast"])
    pred_tr, c_tr, e_tr, mu, sd = outputs(fast, tr)
    pred_t, c_t, e_t, _, _ = outputs(fast, te, mu, sd)
    pred_o, c_o, e_o, _, _ = outputs(fast, oo, mu, sd)
    d_mean, d_std = float(tr["d"].mean()), float(tr["d"].std() + 1e-12)
    g0 = LogisticRegression(max_iter=2000, random_state=SEED).fit(c_tr.reshape(-1, 1), e_tr)
    s0_t, s0_o = g0.predict_proba(c_t.reshape(-1, 1))[:, 1], g0.predict_proba(c_o.reshape(-1, 1))[:, 1]
    for level in LEVELS:
        q_tr = cue(level, tr["d"], c_tr, d_mean, d_std, r_cue)
        q_t = cue(level, te["d"], c_t, d_mean, d_std, r_cue)
        q_o = cue(level, oo["d"], c_o, d_mean, d_std, r_cue)
        g1 = LogisticRegression(max_iter=2000, random_state=SEED).fit(np.c_[c_tr, q_tr], e_tr)
        for split, pred, pack, s0, s1 in (("ID", pred_t, te, s0_t, g1.predict_proba(np.c_[c_t, q_t])[:, 1]),
                                          ("OOD", pred_o, oo, s0_o, g1.predict_proba(np.c_[c_o, q_o])[:, 1])):
            base = (pred == pack["y"]).astype(float)
            a0, a1 = base.copy(), base.copy()
            a0[top(s0, BUDGET)] = 1.0
            a1[top(s1, BUDGET)] = 1.0
            obs, lo, hi = boot_ci(a1 - a0, r_boot)
            rows.append({"split": split, "w_conf": w_conf, "b0": b0, "level": level,
                         "fast_error_rate": float((pred != pack["y"]).mean()),
                         "confidence_accuracy": float(a0.mean()), "intuition_accuracy": float(a1.mean()),
                         "delta_pp": obs * 100, "ci95_low_pp": lo * 100, "ci95_high_pp": hi * 100,
                         "significant": bool(lo > 0 or hi < 0)})

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/matched_error_rate_primary.csv", index=False)
print(f"target flip rate = {TARGET:.4f}\n")
print(out[out.split == "ID"].pivot(index="w_conf", columns="level", values="delta_pp").round(2).to_string())
print("\nfast error rate per cell (ID):")
print(out[out.split == "ID"].pivot(index="w_conf", columns="level", values="fast_error_rate").round(4).iloc[:, :1].to_string())
print("\nOOD:")
print(out[out.split == "OOD"].pivot(index="w_conf", columns="level", values="delta_pp").round(2).to_string())
