# Adaptive verification with an auxiliary routing signal

**Status:** Submitted to *Neurocomputing* (Elsevier) as a regular article, September 26, 2026.

This repository holds two linked studies. The submitted paper, *Predicting the Routing Value of
Auxiliary Signals Before Deployment: A Preregistered Study of Conditional Information, Baseline
Strength, and Budgeted Verification* (see `paper/`), is the preregistered v9 study: it tests
whether a cheap, held-out diagnostic (delta-AUC) predicts the expensive routing gain from adding
an auxiliary signal to an adaptive-verification system, before that signal is deployed. The
canonical v8 package below was an earlier, exploratory pilot that motivated v9's design; it is
disclosed as exploratory and is not used as confirmatory evidence in the submitted paper.
confidence has been used. Two factors are crossed: how much independent information the cue carries
about actual fast-model error (levels L0-L4) and how informative the confidence baseline itself is
(`w_conf` = 0.0, 0.5, 1.1, 2.0). Everything is synthetic; nothing here models human intuition.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python build_intuition_confidence_grid_v8.py my_run        # ~2 minutes
python sensitivity_matched_error_rate.py my_sensitivity
python make_figure.py my_run figures/figure1.png
```

Compare `my_run/hashes.csv` with `results/hashes.csv`: the nine research outputs are bit-for-bit
identical across independent reruns on the same environment. `results/rerun_hash_comparison.csv`
records that check for the archived run. Environment used for the archived outputs: Python 3.12.3
with the pinned versions in `requirements.txt`.

## Headline result

Intuition minus confidence accuracy at the 20% verification budget, in-distribution (percentage
points; L0 is a null cue, L1 an exact copy of the confidence feature):

| w_conf | conf. AUC | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| 0.0 | 0.4951 | 0.02 | 0.00 | 2.65 | 4.40 | 5.83 |
| 0.5 | 0.6017 | -0.02 | 0.00 | 1.20 | 3.43 | 4.35 |
| 1.1 | 0.7037 | 0.02 | 0.00 | 0.88 | 1.98 | 3.27 |
| 2.0 | 0.7966 | -0.03 | 0.00 | 0.47 | 1.38 | 2.33 |

The cue's marginal value falls as the confidence baseline becomes more informative, and stays
positive for the strong cue across the tested range (L4, 95% paired-bootstrap intervals exclude
zero in all four ID and all four OOD cells). A matched-error-rate sensitivity run holds fast-model
error near 33.7% across `w_conf` and reproduces the same surface, so the decline is not an artefact
of changing task difficulty.

## Files

| Path | Contents |
|---|---|
| `build_intuition_confidence_grid_v8.py` | Canonical generator, gates, budgets, bootstrap, audit checks, hashes |
| `sensitivity_matched_error_rate.py` | Matched-error-rate sensitivity run |
| `make_figure.py` | Rebuilds Figure 1 from `results/primary_bootstrap.csv` |
| `results/primary_20pct_comparison.csv` | Primary-budget accuracies and differences |
| `results/primary_bootstrap.csv` | 5,000-resample paired bootstrap, ID and OOD |
| `results/adaptive_budget_results.csv` | Every split x w_conf x level x budget x method |
| `results/incremental_information_diagnostic.csv` | Held-out change in AUC for fast-model error |
| `results/latent_information_construction_audit.csv` | Cue-to-latent-difficulty construction check |
| `results/difficulty_quartile_breakdown.csv` | Accuracy and verification share by difficulty quartile |
| `results/confidence_baseline_strength.csv` | Baseline informativeness per `w_conf` |
| `results/audit_checks.csv` | Ten structural checks |
| `results/manifest.json`, `results/hashes.csv`, `results/rerun_hash_comparison.csv` | Specification and integrity records |
| `results/matched_error_rate_primary.csv` | Sensitivity-run primary comparison |
| `figures/figure1.png` | Effect surface with bootstrap intervals |

## Design notes worth knowing before reusing this code

- The gate's confidence feature is the standardised negative `|logit(p)|` margin, not `1 - confidence`.
  The raw form misspecifies the link and can make a two-feature gate score *worse* on held-out data
  than a one-feature gate even when the extra feature is informative.
- The confidence baseline must itself be informative, otherwise any cue advantage is measured
  against a baseline that performs like random allocation. `w_conf` makes that an experimental factor.
- Normalisation statistics come from the training split only and are frozen for validation, test and
  OOD. No gate is refit on OOD data, so the OOD numbers test transfer, not adaptation.
- The oracle verifies realised errors and is an upper bound, not a deployable policy.

## Limits

Synthetic data only; logistic-regression fast model and gates; one generator family and one
distribution shift; verification modelled as perfect correction; confidence baselines tested only up
to AUC ~0.80, with no extrapolation beyond that.

## License and citation

Code is MIT-licensed (`LICENSE`). Output CSVs are released under CC BY 4.0. See `CITATION.cff`.
