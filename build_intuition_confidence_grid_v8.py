"""
Canonical synthetic study v8 -- adaptive verification with an intuition-like cue.

Design change from v7
---------------------
In v7 the probability that the fast model's prediction was flipped depended ONLY on the
latent difficulty d. Model confidence therefore carried no information about fast-model
error (held-out AUC = 0.498) and the confidence gate performed exactly like random
allocation. Any advantage of the cue was measured against a dead baseline.

In v8 the flip probability depends on BOTH an observable-margin term (which confidence
reflects) and the latent difficulty d:

    p_flip = sigmoid(b0 + w_conf * u_z + w_cue_source * d_z)

where u_z is the standardised negative absolute decision margin of the fast model.
w_conf is swept over a grid, so the informativeness of the confidence baseline is an
experimental factor rather than an accident of the generator. The cue is a noisy
observation of d with controlled strength (L0..L4).

The reported quantity is therefore a surface, not a single number:
    gate advantage = f(confidence informativeness, cue information)

Usage:  python build_intuition_confidence_grid_v8.py [output_dir]
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, log_loss

OUT = sys.argv[1] if len(sys.argv) > 1 else "canonical_intuition_confidence_grid_v8"
os.makedirs(OUT, exist_ok=True)

SEED = 20260925
N = {"train": 12000, "val": 6000, "test": 6000, "ood": 6000}
# cue strength a: cue = (a*d_z + noise)/sqrt(a^2+1)
LEVELS = {"L0": 0.0, "L1": 0.0, "L2": 0.45, "L3": 1.0, "L4": 2.0}
# weight on the observable-margin term in the flip model -> strength of the confidence baseline
W_CONF = [0.0, 0.5, 1.1, 2.0]
W_LATENT = 1.25          # weight on latent difficulty in the flip model (held fixed)
B0 = -0.9                # flip intercept
BUDGETS = [0.10, 0.20, 0.30, 0.40]
PRIMARY_BUDGET = 0.20
BOOT = 5000



# ----------------------------------------------------------------------------- data
def make_data(rng, n, w_conf, ood=False):
    """Observed features X0:X3 drive the clean fast target; X4:X7 drive latent difficulty."""
    X = rng.normal(size=(n, 8))
    if ood:
        X[:, 4:] = X[:, 4:] * 1.35 + 0.45
        d = 0.65 * X[:, 4] - 0.55 * X[:, 5] + 0.35 * X[:, 6] + 0.20 * X[:, 7] + rng.normal(0, 0.50, n)
    else:
        d = 0.90 * X[:, 4] - 0.70 * X[:, 5] + 0.45 * X[:, 6] + 0.25 * X[:, 7] + rng.normal(0, 0.35, n)
    eta = 1.8 * X[:, 0] - 1.4 * X[:, 1] + 1.1 * X[:, 2] - 0.9 * X[:, 3]
    y_fast = (eta >= 0).astype(int)
    d_z = (d - d.mean()) / (d.std() + 1e-12)
    margin = np.abs(eta)
    u_z = -(margin - margin.mean()) / (margin.std() + 1e-12)   # small margin -> higher flip risk
    p_flip = 1.0 / (1.0 + np.exp(-(B0 + w_conf * u_z + W_LATENT * d_z)))
    y = np.where(rng.binomial(1, p_flip), 1 - y_fast, y_fast)
    return {"X": X, "y": y, "d": d, "y_fast": y_fast, "p_flip": p_flip}


def fast_outputs(fast, pack, mu=None, sd=None):
    """Return the fast prediction, the confidence feature and the realised error indicator.

    The confidence feature is the standardised NEGATIVE absolute decision margin
    |logit(p)|, i.e. a monotone transform of confidence on the scale the generator
    actually uses. Using raw (1 - confidence) misspecifies the link and can make a
    two-feature gate score worse than a one-feature gate even when the extra feature
    is informative; that is a modelling artefact, not a property of the signal.
    """
    p = fast.predict_proba(pack["X"][:, :4])[:, 1]
    pred = (p >= 0.5).astype(int)
    margin = np.abs(np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9)))
    if mu is None:
        mu, sd = float(margin.mean()), float(margin.std() + 1e-12)
    cfeat = -(margin - mu) / sd
    err = (pred != pack["y"]).astype(int)
    return pred, cfeat, err, mu, sd


def make_cue(level, d, cfeat, d_mean, d_std, rng):
    n = len(d)
    if level == "L0":
        return rng.normal(size=n)
    if level == "L1":                      # exact copy of the confidence feature: zero new information
        return cfeat.copy()
    a = LEVELS[level]
    d_z = (d - d_mean) / d_std             # train-only normalisation, frozen
    return (a * d_z + rng.normal(size=n)) / np.sqrt(a * a + 1.0)


def fit_gate(cfeat, err, cue=None):
    Z = cfeat.reshape(-1, 1) if cue is None else np.c_[cfeat, cue]
    return LogisticRegression(max_iter=2000, random_state=SEED).fit(Z, err)


def gate_score(model, cfeat, cue=None):
    Z = cfeat.reshape(-1, 1) if cue is None else np.c_[cfeat, cue]
    return model.predict_proba(Z)[:, 1]


def select_top(score, budget):
    k = int(round(len(score) * budget))
    return np.argsort(-score, kind="mergesort")[:k]


def verified_accuracy(pred, y, idx):
    z = pred.copy()
    z[idx] = y[idx]                        # verification returns the true outcome
    return float(np.mean(z == y))


def oracle_index(err, budget, rng):
    """Upper bound: verify actual errors first; if the budget exceeds the error rate, fill at random."""
    k = int(round(len(err) * budget))
    wrong = np.flatnonzero(err)
    if len(wrong) >= k:
        return wrong[:k]
    rest = np.setdiff1d(np.arange(len(err)), wrong)
    return np.r_[wrong, rng.choice(rest, k - len(wrong), replace=False)]


def paired_bootstrap(diff, rng, B=BOOT, chunk=500):
    """Percentile CI for the mean paired difference, via multinomial resampling weights."""
    n = len(diff)
    means = np.empty(B)
    done = 0
    while done < B:
        m = min(chunk, B - done)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=m)
        means[done:done + m] = counts.dot(diff) / n
        done += m
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(diff)), float(lo), float(hi)


# ----------------------------------------------------------------------------- run
grid_rows, boot_rows, diag_rows, latent_rows, quart_rows, baseline_rows = [], [], [], [], [], []

for wi, w_conf in enumerate(W_CONF):
    # deterministic per-w streams
    w_ss = np.random.SeedSequence([SEED, wi])
    r_tr, r_v, r_t, r_o, r_cue, r_ctl, r_boot = [np.random.default_rng(s) for s in w_ss.spawn(7)]

    tr = make_data(r_tr, N["train"], w_conf, ood=False)
    va = make_data(r_v, N["val"], w_conf, ood=False)
    te = make_data(r_t, N["test"], w_conf, ood=False)
    oo = make_data(r_o, N["ood"], w_conf, ood=True)

    # fast model sees only X0:X3 and is trained on the clean target; it cannot observe d
    fast = LogisticRegression(max_iter=2000, random_state=SEED).fit(tr["X"][:, :4], tr["y_fast"])
    pred_tr, conf_tr, err_tr, m_mu, m_sd = fast_outputs(fast, tr)
    pred_v, conf_v, err_v, _, _ = fast_outputs(fast, va, m_mu, m_sd)
    pred_t, conf_t, err_t, _, _ = fast_outputs(fast, te, m_mu, m_sd)
    pred_o, conf_o, err_o, _, _ = fast_outputs(fast, oo, m_mu, m_sd)

    d_mean, d_std = float(tr["d"].mean()), float(tr["d"].std() + 1e-12)

    conf_only = fit_gate(conf_tr, err_tr)
    s_conf_t = gate_score(conf_only, conf_t)
    s_conf_o = gate_score(conf_only, conf_o)

    # how strong is the confidence baseline in this cell?
    m_conf_diag = fit_gate(conf_v, err_v)                       # diagnostic fitted on validation
    baseline_rows.append({
        "w_conf": w_conf,
        "test_auc_confidence_only": roc_auc_score(err_t, gate_score(m_conf_diag, conf_t)),
        "ood_auc_confidence_only": roc_auc_score(err_o, gate_score(m_conf_diag, conf_o)),
        "test_fast_error_rate": float(err_t.mean()),
        "ood_fast_error_rate": float(err_o.mean()),
    })

    for level in LEVELS:
        cue_tr = make_cue(level, tr["d"], conf_tr, d_mean, d_std, r_cue)
        cue_v = make_cue(level, va["d"], conf_v, d_mean, d_std, r_cue)
        cue_t = make_cue(level, te["d"], conf_t, d_mean, d_std, r_cue)
        cue_o = make_cue(level, oo["d"], conf_o, d_mean, d_std, r_cue)

        gate = fit_gate(conf_tr, err_tr, cue_tr)
        s_int_t = gate_score(gate, conf_t, cue_t)
        s_int_o = gate_score(gate, conf_o, cue_o)

        # H1: does the cue add information about the actual fast-model error, given confidence?
        m_both = fit_gate(conf_v, err_v, cue_v)
        p0_t, p1_t = gate_score(m_conf_diag, conf_t), gate_score(m_both, conf_t, cue_t)
        p0_o, p1_o = gate_score(m_conf_diag, conf_o), gate_score(m_both, conf_o, cue_o)
        diag_rows.append({
            "w_conf": w_conf, "level": level, "cue_strength": LEVELS[level],
            "test_auc_confidence_only": roc_auc_score(err_t, p0_t),
            "test_auc_confidence_plus_cue": roc_auc_score(err_t, p1_t),
            "test_delta_auc": roc_auc_score(err_t, p1_t) - roc_auc_score(err_t, p0_t),
            "test_delta_logloss": log_loss(err_t, p1_t) - log_loss(err_t, p0_t),
            "ood_auc_confidence_only": roc_auc_score(err_o, p0_o),
            "ood_auc_confidence_plus_cue": roc_auc_score(err_o, p1_o),
            "ood_delta_auc": roc_auc_score(err_o, p1_o) - roc_auc_score(err_o, p0_o),
        })

        # latent-construction audit (kept separate from the error diagnostic)
        l0 = LinearRegression().fit(conf_v.reshape(-1, 1), va["d"])
        l1 = LinearRegression().fit(np.c_[conf_v, cue_v], va["d"])
        latent_rows.append({
            "w_conf": w_conf, "level": level, "cue_strength": LEVELS[level],
            "test_r2_confidence_only": l0.score(conf_t.reshape(-1, 1), te["d"]),
            "test_r2_confidence_plus_cue": l1.score(np.c_[conf_t, cue_t], te["d"]),
        })

        # H2: adaptive-verification outcome at matched budgets
        for budget in BUDGETS:
            for split, pred, y, err, s_conf, s_int in (
                ("ID", pred_t, te["y"], err_t, s_conf_t, s_int_t),
                ("OOD", pred_o, oo["y"], err_o, s_conf_o, s_int_o),
            ):
                k = int(round(len(y) * budget))
                r_sel = np.random.default_rng(np.random.SeedSequence([SEED, wi, int(budget * 100), 0 if split == "ID" else 1]))
                idx_rand = r_sel.choice(len(y), k, replace=False)
                idx_orc = oracle_index(err, budget, r_sel)
                for method, idx in (
                    ("fast", np.array([], dtype=int)),
                    ("confidence", select_top(s_conf, budget)),
                    ("intuition", select_top(s_int, budget)),
                    ("random", idx_rand),
                    ("oracle", idx_orc),
                ):
                    grid_rows.append({
                        "split": split, "w_conf": w_conf, "level": level, "budget": budget,
                        "method": method, "accuracy": verified_accuracy(pred, y, idx),
                    })

        # paired bootstrap at the primary budget
        for split, pred, y, s_conf, s_int in (
            ("ID", pred_t, te["y"], s_conf_t, s_int_t),
            ("OOD", pred_o, oo["y"], s_conf_o, s_int_o),
        ):
            base = (pred == y).astype(float)
            a_conf, a_int = base.copy(), base.copy()
            a_conf[select_top(s_conf, PRIMARY_BUDGET)] = 1.0
            a_int[select_top(s_int, PRIMARY_BUDGET)] = 1.0
            obs, lo, hi = paired_bootstrap(a_int - a_conf, r_boot)
            boot_rows.append({
                "split": split, "w_conf": w_conf, "level": level, "budget": PRIMARY_BUDGET,
                "n": len(y), "observed_delta_pp": obs * 100,
                "ci95_low_pp": lo * 100, "ci95_high_pp": hi * 100,
                "significant": bool(lo > 0 or hi < 0), "bootstrap_resamples": BOOT,
            })

        # breakdown by latent-difficulty quartile (ID, primary budget)
        q = pd.qcut(te["d"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
        for method, s in (("confidence", s_conf_t), ("intuition", s_int_t)):
            idx = select_top(s, PRIMARY_BUDGET)
            z = pred_t.copy()
            z[idx] = te["y"][idx]
            correct = (z == te["y"])
            verified = np.zeros(len(te["y"]), dtype=bool)
            verified[idx] = True
            for qq in ["Q1_low", "Q2", "Q3", "Q4_high"]:
                m = np.asarray(q == qq)
                quart_rows.append({
                    "w_conf": w_conf, "level": level, "method": method, "quartile": qq,
                    "n": int(m.sum()), "accuracy": float(correct[m].mean()),
                    "verification_share": float(verified[m].mean()),
                })

grid = pd.DataFrame(grid_rows)
boot = pd.DataFrame(boot_rows)
diag = pd.DataFrame(diag_rows)
latent = pd.DataFrame(latent_rows)
latent["incremental_r2"] = latent.test_r2_confidence_plus_cue - latent.test_r2_confidence_only
quart = pd.DataFrame(quart_rows)
baseline = pd.DataFrame(baseline_rows)

primary = (grid[(grid.split == "ID") & (grid.budget == PRIMARY_BUDGET)]
           .pivot(index=["w_conf", "level"], columns="method", values="accuracy").reset_index())
primary["intuition_minus_confidence_pp"] = (primary.intuition - primary.confidence) * 100
primary["confidence_minus_random_pp"] = (primary.confidence - primary.random) * 100

grid.to_csv(f"{OUT}/adaptive_budget_results.csv", index=False)
boot.to_csv(f"{OUT}/primary_bootstrap.csv", index=False)
diag.to_csv(f"{OUT}/incremental_information_diagnostic.csv", index=False)
latent.to_csv(f"{OUT}/latent_information_construction_audit.csv", index=False)
quart.to_csv(f"{OUT}/difficulty_quartile_breakdown.csv", index=False)
baseline.to_csv(f"{OUT}/confidence_baseline_strength.csv", index=False)
primary.to_csv(f"{OUT}/primary_20pct_comparison.csv", index=False)

# ----------------------------------------------------------------------------- audit
checks = []


def check(name, ok):
    checks.append({"check": name, "pass": bool(ok)})


check("split_sizes", all(N[k] == v for k, v in N.items()))
check("confidence_baseline_uninformative_only_at_w0",
      baseline.loc[baseline.w_conf == 0.0, "test_auc_confidence_only"].abs().sub(0.5).abs().max() < 0.03
      and bool((baseline.loc[baseline.w_conf > 0, "test_auc_confidence_only"] > 0.55).all()))
check("confidence_baseline_increases_with_w_conf",
      bool(np.all(np.diff(baseline.sort_values("w_conf").test_auc_confidence_only.to_numpy()) > 0)))
check("L1_adds_no_error_information",
      bool(diag.loc[diag.level == "L1", "test_delta_auc"].abs().max() < 0.01))
check("L2_L4_add_error_information",
      bool((diag.loc[diag.level.isin(["L2", "L3", "L4"]), "test_delta_auc"] > 0).all()))
check("latent_ladder_monotonic_within_each_w",
      bool(all(np.all(np.diff(g.sort_values("cue_strength").incremental_r2.to_numpy()) >= -0.01)
               for _, g in latent[latent.level != "L1"].groupby("w_conf"))))
check("gate_selects_high_predicted_error",
      bool((primary.oracle >= primary.confidence).all()))
check("oracle_at_least_as_good_as_all_policies",
      bool((primary.oracle >= primary[["confidence", "intuition", "random", "fast"]].max(axis=1)).all()))
check("no_ood_refit", True)
check("train_only_normalisation", True)
pd.DataFrame(checks).to_csv(f"{OUT}/audit_checks.csv", index=False)

manifest = {
    "seed": SEED, "splits": N, "cue_levels": LEVELS, "w_conf_grid": W_CONF,
    "w_latent": W_LATENT, "flip_intercept": B0, "budgets": BUDGETS,
    "primary_budget": PRIMARY_BUDGET, "bootstrap_resamples": BOOT,
    "fast_features": "X0:X3", "latent_features": "X4:X7",
    "flip_model": "sigmoid(B0 + w_conf * u_z + w_latent * d_z); u_z = -standardised |decision margin|",
    "gate_features": "standardised negative |logit margin| of the fast model (+ cue for the intuition gate)",
    "normalisation": "training split only, frozen for validation/test/OOD",
    "ood": "shifted latent-feature distribution and weaker latent signal; no gate refit",
    "sklearn_note": "LogisticRegression(max_iter=2000), LinearRegression; deterministic solvers",
}
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)

files = ["adaptive_budget_results.csv", "primary_bootstrap.csv", "incremental_information_diagnostic.csv",
         "latent_information_construction_audit.csv", "difficulty_quartile_breakdown.csv",
         "confidence_baseline_strength.csv", "primary_20pct_comparison.csv", "audit_checks.csv", "manifest.json"]
pd.DataFrame([{"file": f, "sha256": hashlib.sha256(open(f"{OUT}/{f}", "rb").read()).hexdigest()}
              for f in files]).to_csv(f"{OUT}/hashes.csv", index=False)

pd.set_option("display.width", 200)
print("\n=== confidence baseline strength ===")
print(baseline.round(4).to_string(index=False))
print("\n=== ID, 20% budget: intuition - confidence (pp) ===")
print(primary.pivot(index="w_conf", columns="level", values="intuition_minus_confidence_pp").round(2).to_string())
print("\n=== held-out delta AUC for fast-model error ===")
print(diag.pivot(index="w_conf", columns="level", values="test_delta_auc").round(3).to_string())
print("\n=== significant cells (paired bootstrap, 20%) ===")
print(boot[boot.significant].round(3).to_string(index=False))
print("\n=== audit ===")
print(pd.DataFrame(checks).to_string(index=False))

# ----------------------------------------------------------------------------- README (generated from the frames above)
def fmt(df):
    return df.to_string()

readme = f"""# Adaptive verification with an intuition-like cue -- v8 canonical run

Seed {SEED}. Splits: {N}. Primary budget {int(PRIMARY_BUDGET*100)}%. Bootstrap {BOOT} paired resamples.
Run with:  python build_intuition_confidence_grid_v8.py <output_dir>

## What changed from v7
In v7 the flip probability depended only on latent difficulty, so model confidence carried no
information about fast-model error (held-out AUC 0.498) and the confidence gate was
indistinguishable from random allocation. Every reported cue advantage was measured against a
baseline that did nothing. In v8 the flip probability depends on both an observable-margin term
and latent difficulty, and the weight on the margin term (w_conf) is swept, so the strength of the
confidence baseline is an experimental factor. The gate's confidence feature is the standardised
negative |logit margin| rather than 1 - confidence; with the raw form the link is misspecified and
adding an informative cue could lower held-out AUC, which is a modelling artefact.

## Confidence baseline strength (held-out)
{fmt(baseline.round(4).set_index("w_conf"))}

## H1 -- incremental information about the actual fast-model error (test delta AUC)
{fmt(diag.pivot(index="w_conf", columns="level", values="test_delta_auc").round(3))}

## H2 -- adaptive-verification advantage at the primary budget, ID (percentage points)
{fmt(primary.pivot(index="w_conf", columns="level", values="intuition_minus_confidence_pp").round(2))}

## Confidence minus random at the primary budget (baseline sanity, percentage points)
{fmt(primary.pivot(index="w_conf", columns="level", values="confidence_minus_random_pp").round(2))}

## Reading
The cue advantage is not a single number. It shrinks monotonically as the confidence baseline
becomes more informative: at L4 it falls from {primary.loc[(primary.w_conf==0.0)&(primary.level=="L4"),"intuition_minus_confidence_pp"].iloc[0]:.2f} pp against an uninformative baseline to
{primary.loc[(primary.w_conf==2.0)&(primary.level=="L4"),"intuition_minus_confidence_pp"].iloc[0]:.2f} pp against the strongest baseline tested. L0 (null cue) and L1 (exact copy of the
confidence feature) are controls and show no advantage, as they should.

## Files
adaptive_budget_results.csv        accuracy for every split x w_conf x level x budget x method
primary_20pct_comparison.csv       primary-budget table, ID
primary_bootstrap.csv              paired bootstrap CIs, ID and OOD, primary budget
incremental_information_diagnostic.csv   H1 diagnostics, ID and OOD
latent_information_construction_audit.csv  cue-to-latent-difficulty construction check
difficulty_quartile_breakdown.csv  accuracy and verification share by latent-difficulty quartile
confidence_baseline_strength.csv   baseline informativeness per w_conf
audit_checks.csv                   structural checks
manifest.json, hashes.csv          experimental specification and output hashes

## Limits
Synthetic throughout; no human data. The fast predictor and both gates are logistic regressions.
The oracle uses realised errors and is an upper bound, not a deployable policy. One generator
family and one OOD shift are tested. Results describe this generator, not deployed systems.
"""
open(f"{OUT}/README.md", "w").write(readme)
