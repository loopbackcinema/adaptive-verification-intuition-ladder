# Preregistration — v9 study

**Title (working):** Predicting the Routing Value of Auxiliary Signals Before Deployment: Conditional Information, Confidence Strength, and Budgeted Verification

**Author:** Velayutham S (ORCID 0009-0005-6518-9291), Independent Researcher, Chennai, India

**Repository:** https://github.com/loopbackcinema/adaptive-verification-intuition-ladder

**Status at time of writing:** No v9 code has been written or run. This document fixes the questions, hypotheses, design, outcomes, analyses and decision rules before any v9 result exists. Its commit time and the Zenodo archive of the release that contains it serve as the timestamp.

---

## 0. What is already known (pilot disclosure)

The v8 study (release v8.0.1, DOI 10.5281/zenodo.22955744) is treated as an **exploratory pilot**. Its results are known to the author and are not re-tested here as confirmatory findings. In v8, with a linear generator and logistic gates:

- the routing gain of a cue over a confidence-only gate at a 20% budget fell from 5.83 to 2.33 percentage points (pp) for the strongest cue as the confidence baseline became more informative;
- the weak cue showed no detected benefit against the strongest baseline tested;
- a matched-error-rate sensitivity analysis reproduced the same pattern.

v9 asks new questions that v8 did not answer. Where a v9 hypothesis overlaps with a v8 observation, this is stated.

---

## 1. Research questions

- **RQ1 (predictive law).** Can a cheap diagnostic computed before deployment — the gain in held-out AUC for predicting fast-model error when a cue is added to a baseline (ΔAUC) — predict the expensive quantity practitioners care about, the matched-budget routing gain?
- **RQ2 (strong baselines).** Does the cue retain routing value against a learned error predictor that uses every input the fast model sees, rather than confidence alone?
- **RQ3 (misspecification).** Do the conclusions hold when the data-generating process is nonlinear and when the gate is a nonlinear learner?
- **RQ4 (shift).** How does routing gain behave under covariate shift, prevalence shift, and degradation of the cue itself?
- **RQ5 (practical value).** How much verification budget does a useful cue save at matched accuracy?
- **RQ6 (real data).** Does the ΔAUC → routing-gain relation hold on public tabular datasets with real feature groups instead of a synthetic latent variable?

---

## 2. Confirmatory hypotheses

Five hypotheses are confirmatory. All other analyses are descriptive.

**H1 — the predictive law transfers.** A monotone mapping from validation-set ΔAUC to test-set routing gain at the 20% budget, fitted on development cells, predicts routing gain on held-out cells.

- Development cells: linear generator, seeds 1–10. Held-out cells: (a) linear generator, seeds 11–20; (b) nonlinear generator, all seeds.
- Mapping: isotonic regression of routing gain on ΔAUC (baseline B3, Section 4.3), fitted on development cells only.
- **Support:** Spearman ρ between predicted and observed gain ≥ 0.80 on held-out set (a) **and** ≥ 0.70 on held-out set (b), and mean absolute error ≤ 0.75 pp on (a).
- **Failure:** either correlation below its threshold, or MAE above 0.75 pp on (a).

**H2 — value survives a strong baseline.** Against baseline B3, the strongest cue (a = 2.0) yields positive routing gain at the 20% budget at w_conf = 1.0 under the linear generator with the logistic gate.

- **Support:** seed-level 95% interval for mean gain lies above zero.
- **Failure:** the interval includes or lies below zero.

**H3 — gain declines with baseline strength.** For cue strengths a ∈ {1.0, 1.5, 2.0, 3.0}, routing gain against B1 decreases as w_conf increases across its 11 levels. This extends a v8 observation to a denser grid, more seeds, and the nonlinear generator.

- **Support:** Spearman ρ between w_conf and mean gain ≤ −0.80 for each of the four cue strengths, under both generators.
- **Failure:** any of the eight correlations above −0.80.

**H4 — cue degradation erodes and can reverse the gain.** Under the cue-degradation shift (Section 4.5), routing gain against B3 at the 20% budget for a = 2.0, w_conf = 1.0, declines monotonically with degradation severity, and at the most severe level is not positive.

- **Support:** Spearman ρ between severity and mean gain ≤ −0.80 **and** the 95% interval at the most severe level includes or lies below zero.
- **Failure:** either condition not met.

**H5 — the law holds on real data.** On the real-data datasets (Section 6), ΔAUC and routing gain are positively related across datasets × cue-group conditions.

- **Support:** Spearman ρ ≥ 0.60 across all real-data conditions pooled.
- **Failure:** ρ below 0.60.

**Multiplicity.** The five confirmatory tests are corrected with Holm's method at family-wise α = 0.05 where a p-value is involved (seed-level permutation test for correlations; seed-level bootstrap for H2 and H4). Threshold criteria above must also be met.

**Commitment.** Every hypothesis is reported as supported or not supported, whatever the outcome. A failed hypothesis is reported in the abstract if it bears on the main claim.

---

## 3. A point fixed in advance about calibration

Any strictly monotone recalibration of the fast model's confidence (temperature scaling, Platt scaling, isotonic calibration) preserves the ordering of cases, so a top-k gate built on calibrated confidence selects the same cases as one built on raw confidence. Calibrated-confidence baselines are therefore expected to give identical routing outcomes to B1. They are included (B2) to verify this empirically and to answer the reviewer-level objection directly; the substantive strong baseline is B3.

---

## 4. Synthetic design

### 4.1 Factors

| Factor | Levels |
|---|---|
| Generator | linear (v8 form), nonlinear |
| w_conf (margin weight in flip model) | 0.0, 0.2, 0.4, …, 2.0 (11 levels) |
| Cue strength a | 0 (null), redundant copy of B1 feature, 0.25, 0.45, 0.7, 1.0, 1.5, 2.0, 3.0 |
| Gate learner | logistic regression, gradient boosting, multilayer perceptron |
| Baseline | B1, B2, B3 (Section 4.3) |
| Budget | 5, 10, 15, 20, 25, 30, 40% (primary: 20%) |
| Seeds | 20, from `numpy.random.SeedSequence(20261001)` spawned per seed |

Split sizes per seed: training 12,000; validation 6,000; test 6,000; each shifted test set 6,000.

### 4.2 Generators

Features X ~ N(0, I₈). Fast target from X0–X3: η = 1.8·X0 − 1.4·X1 + 1.1·X2 − 0.9·X3, y_fast = 1[η ≥ 0]. Latent difficulty from X4–X7: d = 0.90·X4 − 0.70·X5 + 0.45·X6 + 0.25·X7 + ε, ε ~ N(0, 0.35²). u_z is the standardised negative |η|; d_z is standardised d.

- **Linear:** p_flip = σ(B0 + w_conf·u_z + 1.25·d_z).
- **Nonlinear:** p_flip = σ(B0 + w_conf·u_z + 1.25·(d_z + 0.5·d_z² − 0.5) + 0.6·u_z·d_z + 0.8·1[d_z > 1]).

B0 is solved per (generator, w_conf) so that the expected flip rate equals the linear generator's rate at w_conf = 0 (the v8 matched-error-rate procedure), holding task difficulty fixed across the grid.

Cue: q = (a·d_z + e) / √(a² + 1), e ~ N(0, 1), with d normalised by training-split statistics.

### 4.3 Baselines

- **B1 — confidence gate:** logistic regression on the standardised negative |logit| margin of the fast model (v8 feature).
- **B2 — calibrated-confidence gates:** B1's feature after temperature scaling, and after isotonic calibration, each fitted on the validation split.
- **B3 — learned error predictor:** gradient-boosted classifier predicting fast-model error from X0–X3 plus the B1 feature; the strongest baseline that uses no information the fast model lacks.

The cue gate for each baseline uses that baseline's inputs plus the cue, with the same learner.

### 4.4 Gate learners (fixed hyperparameters, no tuning on test data)

- Logistic regression: L2, C = 1.0, max_iter = 2000.
- Gradient boosting: `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)`.
- Multilayer perceptron: `MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True, max_iter=500, random_state=seed)`.

Gates are fitted on the training split. ΔAUC diagnostics are fitted on the validation split and evaluated on the test split. No model is refitted on any shifted set.

### 4.5 Shifts

- **Covariate shift:** X4–X7 scaled by (1 + s) and shifted by s/2, s ∈ {0.25, 0.5, 1.0}.
- **Prevalence shift:** flip intercept shifted by δ ∈ {−0.5, +0.5, +1.0}.
- **Cue degradation:** cue noise standard deviation multiplied by k ∈ {1, 1.5, 2, 3, 5}; separately, a fraction f ∈ {0.1, 0.25, 0.5} of cue values replaced by pure noise.

### 4.6 Outcomes

- **Routing gain:** accuracy of the cue gate minus accuracy of the corresponding baseline gate at the same budget, in pp. Verified cases take the true label in the synthetic study.
- **ΔAUC:** validation-fitted, test-evaluated AUC for predicting fast-model error, baseline + cue minus baseline.
- **Budget savings:** the smallest budget b* at which the cue gate matches the baseline's accuracy at 20%, by linear interpolation on the budget grid; savings = (0.20 − b*) / 0.20.
- Controls reported in every table: fast-only, random allocation, oracle.

### 4.7 Statistical analysis

Seed is the unit of replication. Cell means are reported with 95% intervals from 5,000 bootstrap resamples of seeds. Within-seed paired case-level bootstraps are reported descriptively. Correlation tests use 5,000 seed-level permutations.

---

## 5. Audit checks (pipeline fails if any fails)

1. Split sizes as specified; no case shared across splits.
2. Redundant-copy cue gives ΔAUC within ±0.005 of zero and routing gain within ±0.1 pp for every cell.
3. B2 routing outcomes identical to B1 for every cell (Section 3).
4. Oracle accuracy ≥ every practical gate for every cell.
5. Gate selections target high predicted error (mean score of selected cases ≥ overall mean).
6. Normalisation statistics computed from the training split only.
7. No refitting on shifted sets.
8. Two independent full runs produce bit-identical research outputs (SHA-256).
9. Realised flip rate within ±1 pp of target across the w_conf grid.

---

## 6. Real-data study (RQ6, H5)

Run by the author locally with a script provided in the repository; datasets are retrieved with `sklearn.datasets.fetch_openml`.

- **Datasets, in priority order:** adult (OpenML 1590), bank-marketing (1461), electricity (151), MagicTelescope (1120). Replacements if one cannot be retrieved: phoneme (1489), credit-g (31). Identifiers are to be confirmed against OpenML before the first run; any change is logged.
- **Feature groups:** for each dataset and each of 10 seeds, features are randomly split into a fast group (50%) and a held-out group (50%).
- **Fast model:** logistic regression on the fast group. **Verifier:** gradient boosting on all features (not an oracle; realistic verification).
- **Cue:** cross-fitted predicted probability of fast-model error from a logistic model on the held-out group only, trained on the training split.
- **Baselines:** B1 and B3 as in Section 4.3, using the fast group.
- **Outcomes:** ΔAUC and routing gain at 20% as in Section 4.6.

---

## 7. What would change the paper's claims

- **H1 fails:** the paper does not claim a predictive law; it reports ΔAUC as necessary but not sufficient, with the held-out evidence.
- **H2 fails:** the paper states that, in this setting, the cue adds no detectable routing value once a learned error predictor on the fast model's own inputs is available, and narrows its practical claims accordingly.
- **H3 fails:** the v8 pattern is reported as not robust to the denser grid or to the nonlinear generator.
- **H4 fails:** the paper does not claim that degraded cues become harmful.
- **H5 fails:** all conclusions are restricted to the synthetic setting and this is stated in the title or abstract.

---

## 8. Deviations

Any departure from this plan — a bug fix, a dataset replacement, a change forced by a library — is recorded in `DEVIATIONS.md` with the date, the reason, and whether it was made before or after seeing any v9 result. Results affected by a post-hoc deviation are labelled exploratory.
