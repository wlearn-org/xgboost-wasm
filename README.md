# @statsim/xgb

XGBoost v3.2.0 compiled to WebAssembly. Training and inference in browsers and Node.js.

Based on [XGBoost v3.2.0](https://github.com/dmlc/xgboost) (Apache-2.0).

## Install

```bash
npm install @statsim/xgb
```

## Quick start

```js
import { loadXGB, DMatrix, Booster } from '@statsim/xgb'

await loadXGB()

// Create training data
const dtrain = new DMatrix([[1, 2], [3, 4], [5, 6], [7, 8]])
dtrain.setLabel([3, 7, 11, 15])

// Train
const booster = new Booster({
  objective: 'reg:squarederror',
  max_depth: 3,
  eta: 0.3,
  verbosity: 0
}, [dtrain])

for (let i = 0; i < 50; i++) {
  booster.update(dtrain, i)
}

// Predict
const preds = booster.predict(dtrain) // Float32Array

// Save model
const model = booster.saveModel() // Uint8Array (UBJSON)

// Clean up — required, WASM memory is not garbage collected
booster.dispose()
dtrain.dispose()
```

## Convenience API

```js
import { loadXGB, train, predict, DMatrix } from '@statsim/xgb'

await loadXGB()

// Train in one call
const dtrain = new DMatrix([[1, 2], [3, 4], [5, 6], [7, 8]])
dtrain.setLabel([3, 7, 11, 15])

const booster = await train({
  objective: 'reg:squarederror',
  max_depth: 3,
  verbosity: 0
}, dtrain, 50)

const model = booster.saveModel()
booster.dispose()
dtrain.dispose()

// Load model and predict
const preds = await predict(model, [[2, 3], [6, 7]])
// preds: Float32Array
```

## API

### `loadXGB(options?)`

Initialize the WASM module. Must be called before any other API. Returns the raw WASM module.

Options:
- `wasmUrl` — override the `.wasm` file URL (for bundlers that mangle paths)

### `DMatrix(data, options?)`

Create a data matrix.

- `data` — `number[][]` (2D array) or `Float32Array`
- `options.nrow`, `options.ncol` — required when `data` is `Float32Array`
- `options.missing` — missing value indicator (default: `NaN`)
- `options.label` — set labels at construction time

Properties and methods:
- `.rows` — number of rows
- `.cols` — number of columns
- `.setLabel(labels)` — set target labels (`number[]` or `Float32Array`)
- `.setWeight(weights)` — set sample weights
- `.dispose()` — free WASM memory (required, idempotent)

### `Booster(params, cache?)`

Create a booster for training.

- `params` — XGBoost parameters object (see [XGBoost docs](https://xgboost.readthedocs.io/en/stable/parameter.html))
- `cache` — array of `DMatrix` objects for cache hint

Methods:
- `.setParam(name, value)` — set a single parameter
- `.update(dtrain, iteration)` — run one training round
- `.predict(dtest, options?)` — predict, returns `Float32Array`
  - `options.ntreeLimit` — limit number of trees (0 = all)
  - `options.type` — prediction type (0 = normal, 1 = margin, etc.)
- `.saveModel(format?)` — save model, returns `Uint8Array`
  - `format` — `'ubj'` (default) or `'json'`
- `.dispose()` — free WASM memory (required, idempotent)

### `Booster.loadModel(buffer)`

Load a model from a `Uint8Array` (UBJ or JSON format). Returns a `Booster`.

### `train(params, dtrain, numRound?)`

Convenience: create a booster, train for `numRound` iterations, return the booster.

### `predict(modelBuffer, data)`

Convenience: load a model, predict on data (2D array), return `Float32Array`. Disposes internal objects automatically.

## Supported objectives

Tested and verified:
- `reg:squarederror` — regression
- `binary:logistic` — binary classification (probabilities)
- `multi:softprob` — multiclass classification (probabilities)
- `count:poisson` — Poisson regression (counts)
- `survival:cox` — Cox proportional hazards

All XGBoost objectives should work — these are tested in CI.

## Resource management

WASM heap memory is not garbage collected. You **must** call `.dispose()` on every `DMatrix` and `Booster` when done. Double-dispose is safe (idempotent). A `FinalizationRegistry` safety net warns in development if you forget, but do not rely on it.

```js
const dm = new DMatrix(data)
try {
  // use dm...
} finally {
  dm.dispose()
}
```

## Cross-runtime compatibility

Models saved in Python XGBoost 3.2.0 load and predict identically in `@statsim/xgb` (verified with relative tolerance < 1e-4). Models saved from JS can be loaded in Python and vice versa.

## Build from source

Requires [Emscripten](https://emscripten.org/) (emsdk) activated.

```bash
# Clone XGBoost v3.2.0
git clone --depth 1 --branch v3.2.0 --recurse-submodules \
  https://github.com/dmlc/xgboost reference/xgboost-upstream

# Build WASM
bash scripts/build-wasm.sh

# Run tests
node test/test.js
```

## License

Apache-2.0 (same as upstream XGBoost)
