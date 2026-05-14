# The Quadrature Parameterization — ODE-by-Construction for the Cox PINN

_Why `phaseA_v2` and beyond compute the cumulative hazard by trapezoidal
integration of `γ(t)` instead of with a free `Λ_φ(t,x)` network, and what that
choice means for the "physics-informed" interpretation of the model._

---

## 1. The model and the ODE it satisfies

The Cox proportional hazards model writes the hazard as

```
λ(t | x) = α(t) · exp(xᵀβ)
```

where `α(t) ≥ 0` is the baseline hazard. With `γ(t) := log α(t)` and
`Λ(t,x) := ∫₀ᵗ λ(s | x) ds` the cumulative hazard, the equivalent ODE form is

```
∂Λ/∂t  =  exp(γ(t) + xᵀβ)     (ODE)
Λ(0,x) =  0                    (IC)
```

The Cox model is therefore completely characterized by the pair `(γ, β)`: once
they are fixed, `Λ` is determined by integration of the ODE. The hazard
expression and Λ's relationship to `γ` are the only physics in this problem.

## 2. The original PINN setup (what `phaseA_v1` and Stages 1–4 did)

A canonical PINN approximates `Λ` directly with a neural network `Λ_φ(t,x)`
and enforces the ODE and IC as **soft penalties** on collocation points:

```
L_ODE(φ,θ,β) = mean over (t_j, x_j) of  ( ∂Λ_φ/∂t  −  exp(γ_θ + xᵀβ) )²
L_IC (φ)     = mean over x_i           of  ( Λ_φ(0, x_i) )²
```

This is what the original implementation (`Hazard_Function_Estimation_with_PINN.pdf`)
describes and what the stage-1–4 campaign used:

- `Λ_φ` = `SurrogateNetwork` — an MLP `(t, x) → ℝ≥0` (with Softplus output).
- `γ_θ` = `CoefficientNetwork` — an MLP `t → ℝ`.
- `β` = a learnable parameter vector.
- Composite loss = `mle + pl + ode + ic + (optional priors)`.

`L_ODE` ties `Λ_φ`'s derivative to `γ_θ`. `L_IC` pins `Λ_φ` to 0 at `t=0`.
Together they encode the physics softly.

## 3. Why the canonical PINN failed without an oracle

Stage 4 `v14` (the previous campaign's winner) achieved `IRMSE = 0.0045` *with*
`baseline_ref` active — a loss that supervises `γ(t)` directly against the
simulator's true `α(t)`. That oracle isn't available on real data.

`phaseA_v1` is the Stage 4 v14 recipe with `baseline_ref` and `monotonic`
disabled — the strict pure-PINN form (`L_MLE + L_PL + L_ODE + L_IC` only). It
**catastrophically fails** on every baseline (IRMSE 0.62–1.03; see
`experiments/sweep_results/phaseA_v1/`).

The failure mechanism is identifiability:

- `L_MLE` at observed events `(Y_i, x_i)` only sees the value `Λ_φ(Y_i, x_i)`
  and `γ_θ(Y_i)`. It pulls `γ_θ` up at event times and `Λ_φ` down everywhere.
- `L_ODE` is the *only* term that ties `Λ_φ` to `γ_θ`. At a finite weight on
  finitely many collocation points, it admits a residual.
- The Λ-surrogate has more MLP capacity than `γ_θ` and absorbs most of the MLE
  signal directly — it learns to fit `Λ(Y_i, x_i)` at event points without
  driving `L_ODE` to zero.
- The unsupervised γ-network is left only loosely coupled to the data via
  `L_ODE`, so it drifts to whatever shape minimizes the residual locally. The
  result was the same hump-and-decay γ̂ shape on all four baselines,
  regardless of the true `α(t)`.

`baseline_ref` masks this in Stage 4 v14 because it adds a direct supervision
signal for `γ_θ`. Without that signal, the two-network PINN has a structural
identifiability problem on this data.

## 4. The reformulation: integrate the ODE in closed form

The ODE for `Λ` is *separable* and has an analytic solution given `(γ, β)`:

```
Λ(t,x) = exp(xᵀβ) · ∫₀ᵗ exp(γ(s)) ds .
```

This is exact. There is no PDE/PDE-residual to penalize: once `γ` is fixed,
`Λ` is determined by a 1-D integral. The integral has no closed form for an
arbitrary `γ`, but it can be evaluated to machine precision by quadrature.

Define

```
G(t) := ∫₀ᵗ exp(γ(s)) ds
```

Then

```
Λ(t,x) = exp(xᵀβ) · G(t)
```

`G(t)` depends only on `γ` and on `t` — not on `x`. So we can precompute `G`
on a fixed grid once per forward pass, and look up / interpolate to any
required `t`. Specifically with the trapezoidal rule on `G+1` grid points
`0 = s_0 < s_1 < ⋯ < s_G = 1`:

```
G_cum[k] = Σ_{j=1..k}  ½ ( exp(γ(s_{j-1})) + exp(γ(s_j)) ) · (s_j − s_{j-1})
G_cum[0] = 0
```

For an arbitrary query `t ∈ [0,1]`, locate `s_k ≤ t < s_{k+1}` and linearly
interpolate:

```
G_hat(t) = (1 − τ) · G_cum[k]  +  τ · G_cum[k+1]
τ        = (t − s_k) / (s_{k+1} − s_k)
```

This gives `Λ̂(t, x) = exp(xᵀβ) · G_hat(t)` end-to-end. The whole pipeline is
differentiable w.r.t. `θ` (the γ-net parameters) and `β`.

## 5. The implementation

The code is `HazardPINN._lambda_quadrature` in
`src/models/pinn.py`. With `quadrature_grid = 200` (so `G = 200` grid points
on `[0,1]`), the structure of the routine is:

```python
def _lambda_quadrature(self, t, x):
    n = self.quadrature_grid  # 200

    # 1. Evaluate γ on a fixed uniform grid t_grid ∈ [0,1].
    t_grid = torch.linspace(0.0, 1.0, n, ...).unsqueeze(1)  # [G,1]
    gamma_grid = self.coefficient(t_grid).squeeze(-1)        # [G]
    exp_gamma = torch.exp(gamma_grid)                        # [G]

    # 2. Trapezoidal cumulative integral. dt = 1/(n-1).
    dt = 1.0 / (n - 1)
    segments = (exp_gamma[:-1] + exp_gamma[1:]) * 0.5 * dt   # [G-1]
    G_cum = torch.cat([torch.zeros(1, ...), torch.cumsum(segments, dim=0)])  # [G]

    # 3. Linear interpolation at the query times.
    t_clamped = t.squeeze(-1).clamp(0.0, 1.0)                # [B]
    idx_float = t_clamped * (n - 1)
    idx_lo    = idx_float.floor().long().clamp(0, n - 2)
    frac      = idx_float - idx_lo.to(dtype)
    G_at_t    = G_cum[idx_lo] * (1.0 - frac) + G_cum[idx_lo + 1] * frac  # [B]

    # 4. Multiply by the proportional-hazards factor.
    xb = x @ self.beta                                       # [B]
    return (G_at_t * torch.exp(xb)).unsqueeze(-1)            # [B, 1]
```

Notes:
- The `clamp(0.0, 1.0)` plus the `clamp(0, n-2)` on `idx_lo` give **constant
  extrapolation** past the training time range, which combined with the
  separate clamp inside `CoefficientNetwork.input_clamp_*` removes the runaway
  γ-tail behavior we saw in early variants.
- The integral is exact for any piecewise-linear approximation of `exp(γ)`.
  Discretization error is `O((dt)² · max|d²/ds²(exp(γ))|)` per segment, which
  for our 200-point grid on `[0,1]` is on the order of `10⁻⁵` for typical
  γ-shapes encountered in the campaign.
- The Λ-surrogate's parameters still exist (they're constructed in
  `from_config`) but `_lambda_quadrature` ignores them when
  `parameterization == "quadrature"`. They get no gradient signal.

`HazardPINN.compute_Lambda_derivative` returns `exp(γ + xᵀβ)` directly when
`parameterization == "quadrature"`, because that's the exact derivative of
the analytical form. No autograd through the integrator is needed for the
ODE residual.

## 6. Consequences for the loss function

This is the key point your question landed on. With the quadrature
parameterization the two physics losses become trivial:

### 6.1 Initial condition is exact

`G_cum[0] = 0` by definition, and at `t = 0` the interpolation picks
`idx_lo = 0`, `frac = 0`, so `G_hat(0) = G_cum[0] = 0`. Therefore

```
Λ̂(0, x) = exp(xᵀβ) · 0 = 0
```

for every `x` and every value of `β`. `L_IC` is identically zero.

### 6.2 ODE residual is zero to discretization error

The ODE residual at a query `t` is

```
∂Λ̂/∂t  −  exp(γ̂(t) + xᵀβ)
```

For the quadrature model:

- The "right-hand side" `exp(γ̂(t) + xᵀβ)` uses `γ̂(t) = coefficient(t)`, i.e.
  the γ-net evaluated at the *exact* `t` you queried.
- The "left-hand side" `∂Λ̂/∂t` differentiates the trapezoidal interpolant.
  On the segment `[s_k, s_{k+1}]`:

  ```
  G_hat(t) = G_cum[k]
           + (t − s_k)/dt · ½(exp(γ(s_k)) + exp(γ(s_{k+1}))) · dt
  ∂G_hat/∂t(t) = ½ ( exp(γ(s_k)) + exp(γ(s_{k+1})) )
  ```

  so

  ```
  ∂Λ̂/∂t (t)  =  exp(xᵀβ) · ½ ( exp(γ(s_k)) + exp(γ(s_{k+1})) ) .
  ```

- The RHS at `t` is `exp(γ̂(t) + xᵀβ)`. The mismatch between the two is

  ```
  exp(xᵀβ) · [ ½ (exp(γ(s_k)) + exp(γ(s_{k+1})))  −  exp(γ(t)) ]
  ```

  which is `O((dt)²)` by the trapezoidal-rule midpoint error. With
  `dt = 1/199 ≈ 5.0 × 10⁻³` and γ shapes in our experiments having
  `|d²(exp γ)/dt²| ≲ 10`, the residual is bounded by ~`10⁻⁵`.

In other words `L_ODE` is *already* satisfied to ~5 decimal digits before any
optimization. Setting `ode: 1.0` in the loss weights would compute that
residual but provide no useful gradient signal — only wasted compute. That's
why every architecture from `phaseA_v2` onward sets `ode: 0.0, ic: 0.0`.

### 6.3 What's still doing work in the loss

With ODE and IC vacuous, the active terms are:

| Term | What it constrains | Why it's needed |
|---|---|---|
| `L_MLE` | both `γ` and `β` through the survival likelihood | the data-fitting objective |
| `L_PL` | β alone (it doesn't see `γ`) | provides curvature for β; γ disappears in the partial likelihood |
| `L_monotonic` | sign of `∂γ/∂t` | prevents γ from descending after a transient spike |
| `L_smoothness` | `(∂²γ/∂t²)²` via finite differences | removes the MLE-spike pathology on a free continuous γ |

The Cox PH model under quadrature is now a *standard maximum-likelihood
estimator* for a neural-baseline Cox model, augmented with two
shape-regularizers on `γ` and one anchor (`L_PL`) on `β`.

## 7. So is it still a PINN?

A defensible answer is "structurally physics-constrained, not
physics-informed". The taxonomy that's been useful in the field:

| Approach | Λ representation | ODE/IC enforced |
|---|---|---|
| PINN | free NN | as soft penalty (`L_ODE`, `L_IC`) on collocation points |
| Neural ODE | analytic integral of a learned drift | exactly (by the integrator) |
| Quadrature parameterization (ours) | analytic integral of a learned γ | exactly (by the integrator) |
| Cox PH with neural baseline | `exp(xᵀβ) · ∫ exp(γ_θ)` | exactly |

Our quadrature parameterization is closest to the Neural ODE / "Cox PH with
neural baseline" position. The drift function `exp(γ(t) + xᵀβ)` is learned;
the integration is closed-form (trapezoidal). There is no soft physics
penalty because there is no physics residual to penalize — the model
*structurally* satisfies the ODE and IC.

Reasons one might still call it a PINN-flavored model:

- It is rooted in the same ODE statement of the Cox model.
- It learns one neural component (`γ_θ`) whose role is exactly the physics
  quantity `log α(t)`.
- The cumulative-hazard surrogate `Λ_φ` is still part of the codebase and
  the architecture switch is a single string-valued config field
  (`parameterization: "surrogate"` vs `"quadrature"`).

But strictly: under `parameterization: quadrature`, no ODE residual is being
optimized. The "PI" in PINN no longer carries weight on the loss.

## 8. Trade-offs

**Pros of the quadrature parameterization:**
- Removes the structural identifiability problem of the two-network PINN
  without needing an oracle baseline. β recovery in the post-Stage-4
  campaign is uniformly 10× better than `v14` because γ no longer competes
  with Λ for likelihood capacity.
- ODE and IC are exact; you can never blame the optimizer for failing to
  satisfy them.
- Simpler computation graph — one differentiable evaluation of `γ` on a grid,
  no double-backprop for `∂Λ/∂t`.

**Cons:**
- Loses the "PINN" framing (your critique).
- Doesn't generalize trivially to PDEs / multi-dimensional state where the
  integral has no closed form. PINNs as originally proposed are most useful
  in *exactly* those settings.
- Still leaves the harder question of how to recover `α(t)` itself
  unsupervised — the post-Stage-4 campaign showed that data-only losses on a
  free `γ(t)` have a tail bias regardless of how well the ODE is satisfied,
  because the bias is in the *likelihood landscape*, not in the physics
  constraint. (See `experiments/sweep_results/phaseB_v3_stress/`.)

## 9. When to revert to a true PINN

Cases where re-introducing a free `Λ_φ(t,x)` and turning `ode`/`ic` back on
would be appropriate:

1. **Working with PDEs or higher-dimensional ODEs** where the integral has no
   closed form. The Cox PH ODE is the simplest possible case; for time-varying
   covariates `x(t)`, frailty models, or competing-risks systems, quadrature
   over `γ` alone is insufficient.
2. **Forcing the surrogate to learn time–covariate interactions that violate
   proportional hazards.** Quadrature bakes in proportional hazards
   (`Λ(t,x) = exp(xᵀβ) · G(t)`) — there is no interaction between `t` and `x`
   beyond the multiplicative one. If the truth violates PH, the canonical
   PINN form has the flexibility to represent it; quadrature does not.
3. **Sanity-check or comparison runs.** Restoring the canonical PINN (with
   some non-oracle identifiability fix) and comparing it against quadrature
   would isolate exactly how much of the post-Stage-4 improvement came from
   "ODE-by-construction" vs other levers (smoothness, monotonic, β-clamp,
   etc.).

Procedurally, restoring the canonical PINN form requires only:

- Setting `parameterization: surrogate` in the architecture YAML.
- Setting `ode: 1.0, ic: 1.0` in `loss_weights`.
- Adding a non-oracle constraint that breaks the `(γ, Λ_φ)` identifiability
  exposed in `phaseA_v1` — candidates include:
  (a) factorizing the surrogate as `Λ_φ(t, x) = exp(xᵀβ) · Λ_φ(t)` so that
      `Λ_φ` cannot represent non-proportional interactions, or
  (b) increasing `L_ODE` weight aggressively (≥ 100×) so the ODE residual
      really does drive γ→Λ_φ consistency.

Neither has been tried in the post-Stage-4 campaign.

## 10. Summary

Under `parameterization: quadrature`:

- `Λ(t, x)` is computed exactly as `exp(xᵀβ) · ∫₀ᵗ exp(γ(s)) ds`, evaluated by
  trapezoidal cumulative integration on a 200-point grid.
- The ODE `∂Λ/∂t = exp(γ + xᵀβ)` and the IC `Λ(0, x) = 0` hold to
  discretization error `O(10⁻⁵)`, so `L_ODE` and `L_IC` provide no useful
  gradient signal and are set to zero in every Phase-A/B config from `v2`
  onward.
- This is no longer a soft-constrained PINN; it is a structurally-physical
  neural Cox model — close in spirit to a Neural ODE with known integrator.
- We adopted it because the canonical two-network PINN, without an oracle
  `baseline_ref`, suffers a structural identifiability problem (`phaseA_v1`).
- The remaining open problems — tail bias of γ̂, non-monotone-hazard
  failures, the bathtub stress test — are properties of the **likelihood
  landscape**, not of the physics constraint. They are not fixed by
  reactivating `L_ODE` or `L_IC`.
