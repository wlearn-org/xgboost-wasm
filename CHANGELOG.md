# Changelog

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
