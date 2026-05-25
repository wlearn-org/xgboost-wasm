# Changelog

## 0.2.0
- Depend on the CommonJS `@wlearn/core` release
- Add package homepage and GitHub issue metadata

- Wrap XGBModel with `createModelClass` from `@wlearn/core` for unified task detection
- Add `task` parameter: `'classification'` or `'regression'`, auto-detected from labels if omitted
- When both `task` and `objective` are set, `objective` takes precedence

## 0.1.0 (unreleased)

- Initial release
- XGBoost v3.2.0 compiled to WASM via Emscripten
- Unified sklearn-style API: `create()`, `fit()`, `predict()`, `score()`, `save()`, `dispose()`
- Classification (binary, multiclass), regression, ranking, random forest mode
- Low-level API: `DMatrix`, `Booster` for direct XGBoost C API access
- Buffer-based model I/O (UBJ format, no filesystem dependency)
- Accepts both typed matrices and number[][] with configurable coercion
- `predictProba()` for probability estimates
- `getParams()`/`setParams()` for AutoML integration
- `defaultSearchSpace()` for hyperparameter search
- `FinalizationRegistry` safety net for leak detection
- Apache-2.0 license (same as upstream XGBoost)
