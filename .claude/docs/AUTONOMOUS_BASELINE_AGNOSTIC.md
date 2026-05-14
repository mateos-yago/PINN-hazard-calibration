# Autonomous Baseline-Hazard-Agnostic Architecture Search

This document is the protocol for the post-Stage-4 campaign. The previous campaign
(Stage 1-4, see `AUTONOMOUS_TRAINING.md` and `EXPERIMENT_RESULTS_AND_BEST_MODEL.md`)
produced a Stage 4 winner that depends on `BaselineReferenceLoss`, an oracle leak
unavailable for real survival data. This campaign removes that dependency and
searches for a single architecture that passes thresholds across multiple baseline
hazard families and multiple covariate counts.

---

## Goal

A single (model architecture + training recipe) that, with **`baseline_ref` weight
set to zero**, meets all three acceptance thresholds on every baseline hazard family
in the test panel:

- β RMSE < 0.10
- Hazard integrated relative MSE < 0.05
- C-index > 0.75

Test panel of baselines (same for every architecture):

| Key | Class | Params | Shape |
|---|---|---|---|
| `exp` | ExponentialBaseline | `lam=0.5` | constant |
| `weibull` | WeibullBaseline | `k=1.5, lam=0.5` | power-monotone increasing |
| `gompertz` | GompertzBaseline | `a=0.3, b=0.2` | exponential-monotone increasing |
| `piecewise` | PiecewiseConstantBaseline | `cutpoints=[2.0], rates=[0.2,0.6]` | step up |

Phases:

- **Phase A — p=1**, β = `[1.5]`. Find a small architecture that passes on all 4 baselines.
- **Phase B — p=4**, β = `[1.0, -0.5, 0.3, -0.2]`. Scale the Phase A winner.

All shared simulation defaults: `n_samples=1000`, `censoring_rate=0.3`, `seed=42`.

**Why β=[1.5] at p=1, not the historical [1.0]?** Pre-flight oracle C-index
check (`xᵀβ_true` as risk score on each simulator dataset): β=[1.0] at p=1
seed=42 gives oracle C ≈ 0.72 across all four baselines, **below** the 0.75
threshold. β=[1.5] lifts oracle C to ≈ 0.786 across all four baselines, giving
the threshold a 0.03 margin. The β value lives in `experiments/sweep.py:BETA_BY_P`.

---

## File / naming conventions

| Artifact | Path |
|---|---|
| Architecture YAML (model + training only, no `simulation` block) | `experiments/configs/architectures/{arch_id}.yaml` |
| Leaf config (auto-generated per baseline) | `experiments/configs/{arch_id}_{baseline}.yaml` |
| Leaf experiment results | `experiments/results/{arch_id}_{baseline}/` |
| Sweep aggregate per architecture | `experiments/sweep_results/{arch_id}/{summary.csv,summary.json,summary.md,plot_matrix.png}` |
| Architecture log | `experiments/architecture_log.md` (append-only, one entry per `arch_id`) |
| Phase summary docs | `.claude/docs/PHASE_A_RESULTS.md`, `PHASE_B_RESULTS.md`, `BASELINE_AGNOSTIC_FINAL.md` |

Architecture-id format: `phaseA_v{N}` or `phaseB_v{N}` where `{N}` increments
monotonically per phase.

Leaf experiment names are produced automatically by the sweep runner:
`{arch_id}_{baseline}`.

---

## How to run a sweep

```bash
# Phase A bootstrap:
python -m experiments.sweep \
  --architecture experiments/configs/architectures/phaseA_v1.yaml \
  --baselines exp,weibull,gompertz,piecewise \
  --p 1

# Phase B (after Phase A passes):
python -m experiments.sweep \
  --architecture experiments/configs/architectures/phaseB_v1.yaml \
  --baselines exp,weibull,gompertz,piecewise \
  --p 4
```

Useful flags: `--epochs <N>` overrides the architecture YAML's `n_epochs` (use a
shorter run when probing a new hyperparameter; promote to full epochs once the
direction looks right).

The sweep:

1. Loads the architecture YAML (no `simulation` block).
2. For each baseline key, merges in the preset `simulation` block from
   `experiments/sweep.py:BASELINE_PRESETS` plus the β preset for the given `--p`.
3. Writes the leaf config to `experiments/configs/{arch_id}_{baseline}.yaml`.
4. Calls `run_from_config()` (in-process) — produces the usual per-experiment
   artifacts under `experiments/results/{arch_id}_{baseline}/`.
5. Aggregates metrics into the sweep summary table + 2×2 hazard plot grid.

---

## Session-start protocol (autonomous resumption)

When resuming this campaign in a new session:

1. Read `experiments/architecture_log.md`. Find the last `phaseA_v{N}` or
   `phaseB_v{N}` entry. The `Next planned change:` line tells you what to do.
2. If the last entry shows an accepted architecture for Phase A and the next phase
   is B, jump to Phase B with the same recipe + `--p 4`.
3. If the last entry shows a failure, the diagnostic table below tells you which
   lever to try.
4. Create `experiments/configs/architectures/{next_arch_id}.yaml` reflecting the
   change. Keep the diff minimal — change one lever per version when possible.
5. Run the sweep.
6. Append one architecture-level entry to `experiments/architecture_log.md`. See
   the template below.
7. Commit and push (per project convention).
8. Decide next change OR mark the phase complete.

When a phase completes (all 4 baselines pass under a single architecture), write
`.claude/docs/PHASE_{A,B}_RESULTS.md` with the winning architecture, the metrics
table, and the rationale chain that led to it.

---

## Architecture-log entry template

```markdown
## phaseA_v{N}  [YYYY-MM-DD]

**Diff vs phaseA_v{N-1}:**
- {one line per changed hyperparameter / loss component}

**Rationale:** {why this change was made — point to the specific failure in the
previous entry it addresses}

**Sweep results (p=1):**

| Baseline | β RMSE | Hazard IRMSE | C-index | All pass |
|---|---|---|---|---|
| exp | ... | ... | ... | ✓/✗ |
| weibull | ... | ... | ... | ✓/✗ |
| gompertz | ... | ... | ... | ✓/✗ |
| piecewise | ... | ... | ... | ✓/✗ |

**Failure mode (if any):** {which baseline failed, which metric, what the plot
shows — e.g. "Gompertz tail underfit, γ̂ saturates after t≈3"}

**Next planned change:** {next lever from the diagnostic table, or
"ADVANCE TO PHASE B"}
```

---

## Diagnostic table: failure mode → lever

Levers are tried in order. **Try without before adding** — only introduce a new
regularizer if it both has a plausible mechanism for the observed failure AND
demonstrably improves a metric vs the matched control.

| Failure mode | Likely cause | First lever | If still failing |
|---|---|---|---|
| All baselines fail β with the same sign error | β / γ collapse without PL anchor | Confirm `beta_initialization: cox_pl` is on and `lr_beta` ≤ 1e-4. Raise `pl` weight to 0.5. | Lower `lr_beta` to 5e-5 |
| Weibull / Gompertz tail underfit (IRMSE high, plot saturates) | γ-network can't represent log-α curvature past observed range | Add `sqrt_t` to `time_features` | Widen γ-network to `[96, 96, 96]` |
| PiecewiseConstant fails IRMSE | step too sharp for smooth MLP | Widen γ-network; add 1 layer | Conditional lever: `BaselineSmoothnessLoss` *only if* γ̂ shows oscillations rather than over-smoothing |
| Hazard plot shows γ̂ trending down where data has events | early-training Λ-surrogate violates `Λ' ≥ 0` | Conditional lever: `CumulativeHazardMonotonicityLoss` on `Λ` (shape-agnostic; OK to test). Verify with ablation. | Switch to event-weighted collocation (concentrate ODE residual where data lives) |
| β passes on some baselines, fails on others, same architecture | identifiability — γ absorbs β's effect | Loss curriculum: Phase-1 epochs `(L_MLE + L_PL)` only, Phase-2 turns on `(L_ODE, L_IC)` | Increase `pl` weight; keep `lr_beta` low |
| C-index < oracle C-index − 0.005 | discrimination loss, not seed-limited | Inspect β estimate — if direction is correct but magnitude is off, raise `pl` weight | Switch to ODE-by-construction parameterization (last resort) |
| C-index ≥ oracle − 0.005 but threshold not met | data seed limitation, not the architecture | Note in log; do not change architecture. Document with the oracle C-index figure. | (no architecture change) |

---

## Conditional lever ablation rule

When introducing a new loss component or major training-loop change (e.g.
`CumulativeHazardMonotonicityLoss`, `BaselineSmoothnessLoss`, event-weighted
collocation, ODE-by-construction):

1. Implement the lever.
2. Run a sweep with the lever on (same seed, same other hyperparameters).
3. Run a sweep with the lever off (matched control).
4. Compare per-baseline metrics. The lever is **retained** only if it strictly
   improves at least one metric on at least one baseline without regressing any
   other metric below threshold.
5. If retained, document the ablation in the architecture log under both
   versions. If rejected, document and drop — do not include it in subsequent
   versions.

This rule is the safeguard against "more regularizers always look fine because
some of them work sometimes."

---

## Acceptance criteria

- **Per leaf run** (unchanged): β RMSE < 0.10, Hazard IRMSE < 0.05, C-index > 0.75.
- **Per architecture, per phase**: all 4 baselines pass simultaneously.
- **C-index seed relaxation**: if oracle C-index for the dataset seed is below
  0.75, the C-index threshold relaxes to `oracle_c_index - 0.005`. The oracle
  C-index is computed from the linear predictor `xᵀβ_true` on the standardized
  data; this is the maximum any model could achieve.
- **Final campaign**: one architecture that passes Phase A and Phase B.

---

## Reusing existing infrastructure

The following modules are imported as-is; the campaign does not duplicate them:

- `src.simulation.cox_simulator` — all 4 baseline families.
- `src.data.pipeline` — `SurvivalDataPipeline`, `SurvivalDataset`.
- `src.models.pinn` — `HazardPINN.from_config`.
- `src.training.loss` — `CompositeLoss` and 7 components.
- `src.training.trainer` — `Trainer`, `ExperimentConfig`.
- `src.evaluation.metrics` — `EvaluationReport`.
- `experiments.run_experiment.run_from_config` — single-config runner used
  internally by the sweep.

New code introduced by this campaign:

- `experiments/sweep.py` — multi-baseline runner + aggregation.
- `experiments/configs/architectures/` — architecture-only YAMLs.

Future additions (only when a conditional lever triggers):

- `src/training/loss.py` — `CumulativeHazardMonotonicityLoss`,
  `BaselineSmoothnessLoss` (when needed).
- `src/training/trainer.py` — `loss_schedule`, `event_weighted` collocation
  (when needed).
- `src/models/pinn.py` — `parameterization='quadrature'` (last resort).
