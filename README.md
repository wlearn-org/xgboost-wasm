# XGBoost In JavaScript: Port Evaluation (Feb 15, 2026)

## Why this exists
You asked whether the current JavaScript XGBoost ecosystem is outdated and what we gain by building our own up-to-date port.

## Quick answer
Most JS XGBoost packages are wrappers around older builds or narrow inference bindings. A first-party StatSim port based on current upstream XGBoost gives us feature parity, model compatibility guarantees, reproducibility controls, and direct integration with `fit`/`jsee`/MCP workflows.

## Current ecosystem snapshot

| Package | npm snapshot | Maintenance signal | Notes |
|---|---|---|---|
| `ml-xgboost` | `1.1.2` (published ~9 years ago) | Very stale | Historically useful, but effectively legacy. |
| `xgboost` | `1.1.0` (published ~8 years ago) | Very stale | Legacy Node addon (`nuanio/xgboost-node`) focused on prediction. |
| `@bonniernews/xgboost` | `2.0.0` (published ~3 years ago) | Stale fork | Forked Node binding line, not upstream-official JS runtime. |
| `@fractal-solutions/xgboost-js` | `1.0.0` (published ~9 months ago) | Mixed | Pure-JS approach, but not tied to upstream `dmlc/xgboost` release cadence. |
| `xgboost_node` | `0.4.2` (published ~3 months ago) | More active | Newer native binding line, but Linux-first and still ecosystem-fragmented. |

Takeaway: there are active packages, but the ecosystem is fragmented and most popular historical packages are clearly outdated.

## Upstream reference point (what “current” means)
Official XGBoost latest release is `v3.2.0` (Feb 10, 2026). Recent upstream releases include algorithm and systems-level updates (for example: linear-model L2 regularization support in 3.2, categorical recoder + multi-target quantile regression in 3.1).

If we rely on older JS wrappers, we miss these improvements and risk model-format drift over time.

## Evidence from our current codebase
Our existing wrapper in `fit` is aligned with an older API surface:
- `fit/src/xgboost-wrapper.js:22` uses `objective: 'reg:linear'`.
- `fit/src/xgboost-wrapper.js:28` uses `silent`, which has long been superseded by `verbosity` in modern XGBoost parameter docs.
- `fit/src/xgboost.js:1` hardcodes a remote wasm asset URL (`https://statsim.com/assets/xgboost.wasm`) without explicit upstream version pinning in this module.

This confirms your concern: our current integration is legacy-style.

## What a StatSim port of latest XGBoost gives us

1. **Version parity and trust**
Pin to upstream `v3.2.x`, track CVEs/bugfixes, and keep a clear compatibility matrix.

2. **Model compatibility guarantees**
Round-trip tests against Python/R/CLI artifacts (JSON/UBJSON) reduce “it loads here but not there” failures.

3. **Reproducibility + auditability**
Deterministic build inputs (toolchain, flags, commit hash), attestation-ready artifacts, and regression fixtures fit StatSim’s trust model.

4. **Portable privacy-preserving execution**
Browser/Node WASM runtime means local inference/training without data upload, aligned with platform principles.

5. **Unified developer + agent surface**
A stable JS API can be exposed through `fit`, `jsee`, and MCP tools so agents call real gradient boosting computation, not approximations.

6. **Performance control**
We can tune for web workloads (WASM SIMD/threads, memory budget, worker model) instead of inheriting opaque defaults.

## Suggested implementation scope (pragmatic)

1. **Phase A (2-3 weeks): inference-first runtime**
Build/ship pinned upstream `v3.2.x` WASM + model load/predict API + parity tests vs Python baseline.

2. **Phase B (3-5 weeks): training support**
Add training APIs and objective/metric coverage for the most used tasks (regression, binary/multiclass classification).

3. **Phase C: StatSim integration**
Wrap as `@statsim/xgb` and integrate into `fit` + MCP tool endpoints with schema-tested IO.

## Sources
- npm package pages:
  - https://www.npmjs.com/package/ml-xgboost
  - https://www.npmjs.com/package/xgboost
  - https://www.npmjs.com/package/@bonniernews/xgboost
  - https://www.npmjs.com/package/@fractal-solutions/xgboost-js
  - https://www.npmjs.com/package/xgboost_node
- package index metadata:
  - https://libraries.io/npm/xgboost_node
- XGBoost official releases:
  - https://github.com/dmlc/xgboost/releases
  - https://xgboost.readthedocs.io/en/stable/changes/v3.2.0.html
  - https://xgboost.readthedocs.io/en/stable/changes/v3.1.0.html
- XGBoost parameters doc:
  - https://xgboost.readthedocs.io/en/stable/parameter.html
