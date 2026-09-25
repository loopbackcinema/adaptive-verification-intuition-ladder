"""
Confirmatory hypothesis tests H1-H4, exactly as specified in PREREGISTRATION.md.
Run after build_v9_pipeline.py --aggregate. H5 (real data) is tested separately
in run_real_data_study.py.

Usage: python analyze_hypotheses.py <v9_out_dir>
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

OUT = sys.argv[1]
routing = pd.read_csv(f"{OUT}/routing_gain.csv")
diag = pd.read_csv(f"{OUT}/incremental_information_diagnostic.csv")

PRIMARY_BUDGET = 0.20
ALPHA = 0.05
N_PERM = 5000
N_BOOT = 5000
rng = np.random.default_rng(20261001)

results = {}


def holm(pvals):
    """Holm step-down correction. Returns dict name->adjusted p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running_max = max(running_max, val)
        adj[name] = running_max
    return adj


def perm_test_spearman(x, y, n_perm=N_PERM, seed=0):
    r_obs, _ = spearmanr(x, y)
    rr = np.random.default_rng(seed)
    n = len(x)
    count = 0
    for _ in range(n_perm):
        yp = rr.permutation(y)
        r, _ = spearmanr(x, yp)
        if abs(r) >= abs(r_obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return r_obs, p


def seed_bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    rr = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rr.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


# ============================================================ H1: predictive law transfers
# Development: linear generator, seeds 1-10. Held-out (a): linear, seeds 11-20.
# Held-out (b): nonlinear generator, all seeds.
dev = routing[(routing.generator == "linear") & (routing.seed <= 10) &
              (routing.budget == PRIMARY_BUDGET) & (routing.learner == "logreg")]
hold_a = routing[(routing.generator == "linear") & (routing.seed > 10) &
                  (routing.budget == PRIMARY_BUDGET) & (routing.learner == "logreg")]
hold_b = routing[(routing.generator == "nonlinear") &
                  (routing.budget == PRIMARY_BUDGET) & (routing.learner == "logreg")]

diag_dev = diag[(diag.generator == "linear") & (diag.seed <= 10)]
diag_a = diag[(diag.generator == "linear") & (diag.seed > 10)]
diag_b = diag[diag.generator == "nonlinear"]

key_cols = ["seed", "w_conf", "level"]
dev_m = dev.merge(diag_dev, on=key_cols)
a_m = hold_a.merge(diag_a, on=key_cols)
b_m = hold_b.merge(diag_b, on=key_cols)

iso_h1 = IsotonicRegression(out_of_bounds="clip").fit(dev_m.delta_auc, dev_m.gain_vs_B1_pp)
pred_a = iso_h1.predict(a_m.delta_auc)
pred_b = iso_h1.predict(b_m.delta_auc)

rho_a, _ = spearmanr(pred_a, a_m.gain_vs_B1_pp)
rho_b, _ = spearmanr(pred_b, b_m.gain_vs_B1_pp)
mae_a = float(np.mean(np.abs(pred_a - a_m.gain_vs_B1_pp)))

h1_support = bool(rho_a >= 0.80 and rho_b >= 0.70 and mae_a <= 0.75)
results["H1"] = {
    "development_n": len(dev_m), "holdout_a_n": len(a_m), "holdout_b_n": len(b_m),
    "spearman_holdout_a_linear_seeds11_20": float(rho_a),
    "spearman_holdout_b_nonlinear": float(rho_b),
    "mae_holdout_a_pp": mae_a,
    "support": h1_support,
}

# ============================================================ H2: value survives strong baseline
h2_cell = routing[(routing.generator == "linear") & (routing.level == "a200") &
                   (routing.w_conf == 1.0) & (routing.budget == PRIMARY_BUDGET) &
                   (routing.learner == "logreg")]
mean_g, lo_g, hi_g = seed_bootstrap_ci(h2_cell.gain_vs_B3_pp.to_numpy(), seed=1)
h2_support = bool(lo_g > 0)
results["H2"] = {"n_seeds": len(h2_cell), "mean_gain_vs_B3_pp": mean_g,
                  "ci95_low": lo_g, "ci95_high": hi_g, "support": h2_support}

# ============================================================ H3: gain declines with w_conf
h3_pvals, h3_rhos = {}, {}
for generator in ["linear", "nonlinear"]:
    for level in ["a100", "a150", "a200", "a300"]:
        cell = routing[(routing.generator == generator) & (routing.level == level) &
                        (routing.budget == PRIMARY_BUDGET) & (routing.learner == "logreg")]
        seed_means = cell.groupby("seed").apply(
            lambda g: spearmanr(g.w_conf, g.gain_vs_B1_pp)[0], include_groups=False)
        r_all, p_all = perm_test_spearman(cell.w_conf.to_numpy(), cell.gain_vs_B1_pp.to_numpy(),
                                           seed=hash((generator, level)) % (2**31))
        h3_pvals[f"{generator}_{level}"] = p_all
        h3_rhos[f"{generator}_{level}"] = r_all

h3_adj = holm(h3_pvals)
h3_all_support = all(r <= -0.80 for r in h3_rhos.values())
results["H3"] = {"rhos": h3_rhos, "holm_adjusted_p": h3_adj, "support": bool(h3_all_support)}

# ============================================================ H4: cue-degradation shift
# Not run in this synthetic pass (requires the shift module); flagged not-yet-tested.
results["H4"] = {"support": None, "note": "cue-degradation shift not yet implemented in this pass"}

print(pd.Series({k: v.get("support") for k, v in results.items()}))
import json
json.dump(results, open(f"{OUT}/hypothesis_tests.json", "w"), indent=2, default=str)
print("\nFull results written to hypothesis_tests.json")
for k, v in results.items():
    print(f"\n{k}:")
    for kk, vv in v.items():
        if kk not in ("rhos", "holm_adjusted_p"):
            print(f"  {kk}: {vv}")
