# Changelog

## 0.1.0 (unreleased)

- Initial release
- XGBoost v3.2.0 compiled to WASM via Emscripten
- Training and inference in browsers and Node.js
- ESM package with `loadXGB()`, `DMatrix`, `Booster`, `train()`, `predict()`
- Buffer-based model I/O (no filesystem dependency)
- `dispose()` with FinalizationRegistry safety net
- Apache-2.0 license (same as upstream XGBoost)
