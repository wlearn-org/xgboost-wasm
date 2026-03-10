# @wlearn/xgboost

XGBoost v3.2.0 compiled to WebAssembly. Gradient-boosted trees, random forests, classification, regression, and ranking in browsers and Node.js.

Based on [XGBoost v3.2.0](https://github.com/dmlc/xgboost) (Apache-2.0). Zero dependencies. CommonJS.

## Install

```bash
npm install @wlearn/xgboost
```

## Quick start

```js
const { XGBModel } = require('@wlearn/xgboost')

const model = await XGBModel.create({
  objective: 'binary:logistic',
  max_depth: 3,
  eta: 0.3,
  numRound: 50
})

// Train -- accepts number[][] or { data: Float64Array, rows, cols }
model.fit(
  [[1, 2], [3, 4], [5, 6], [7, 8]],
  [0, 0, 1, 1]
)

// Predict
const preds = model.predict([[2, 3], [6, 7]])  // Float64Array

// Probabilities
const probs = model.predictProba([[2, 3], [6, 7]])  // Float64Array (nrow * nclass)

// Score
const accuracy = model.score([[2, 3], [6, 7]], [0, 1])

// Save / load
const buf = model.save()  // Uint8Array (WLRN bundle)
const model2 = await XGBModel.load(buf)

// Clean up -- required, WASM memory is not garbage collected
model.dispose()
model2.dispose()
```

## API

### `XGBModel.create(params?)`

Async factory. Loads WASM module, returns a ready-to-use model.

Parameters:
- `task` -- `'classification'` or `'regression'`. Auto-detected from labels if omitted. Sets a default objective when no explicit `objective` is provided. If both `task` and `objective` are set, `objective` takes precedence.
- `objective` -- XGBoost objective string (default: `'reg:squarederror'`)
- `max_depth` -- maximum tree depth (default: `6`)
- `eta` -- learning rate (default: `0.3`)
- `numRound` -- number of boosting rounds (default: `100`)
- `num_class` -- number of classes for multiclass objectives
- `subsample` -- row subsampling ratio (default: `1.0`)
- `colsample_bytree` -- column subsampling ratio (default: `1.0`)
- `lambda` -- L2 regularization (default: `1.0`)
- `alpha` -- L1 regularization (default: `0.0`)
- `num_parallel_tree` -- trees per round, >1 for random forest mode (default: `1`)
- `verbosity` -- 0 = silent, 1 = warning, 2 = info (default: `0`)
- `coerce` -- input coercion: `'auto'` | `'warn'` | `'error'` (default: `'auto'`)

### `model.fit(X, y, opts?)`

Train on data. Returns `this`.
- `X` -- `number[][]` or `{ data: Float64Array, rows, cols }`
- `y` -- `number[]` or `Float64Array`
- `opts.sampleWeight` -- per-sample weights (`number[]` or `Float64Array`)

### `model.predict(X)`

Returns `Float64Array` of predicted labels (classification) or values (regression).

### `model.predictProba(X)`

Returns `Float64Array` of shape `nrow * nclass` (row-major probabilities). Available for `binary:logistic` and `multi:softprob` objectives.

### `model.score(X, y)`

Returns accuracy (classification) or R-squared (regression).

### `model.save()` / `XGBModel.load(buffer)`

Save to / load from `Uint8Array` (WLRN bundle with UBJ model blob).

### `model.dispose()`

Free WASM memory. Required. Idempotent.

### `model.getParams()` / `model.setParams(p)`

Get/set hyperparameters. Enables AutoML grid search and cloning.

### `XGBModel.defaultSearchSpace()`

Returns default hyperparameter search space for AutoML.

## Supported objectives

Tested and verified:
- `reg:squarederror` -- regression
- `binary:logistic` -- binary classification (probabilities)
- `multi:softprob` -- multiclass classification (probabilities)
- `multi:softmax` -- multiclass classification (class labels)
- `count:poisson` -- Poisson regression (counts)
- `survival:cox` -- Cox proportional hazards

All XGBoost objectives should work -- these are tested in CI.

## Random forest mode

Set `num_parallel_tree > 1` with subsampling for random forest behavior:

```js
const rf = await XGBModel.create({
  objective: 'binary:logistic',
  numRound: 1,
  num_parallel_tree: 100,
  subsample: 0.8,
  colsample_bynode: 0.8
})
```

## Low-level API

For direct access to XGBoost's C API, use the lower-level `DMatrix` and `Booster` classes:

```js
const { loadXGB, DMatrix, Booster } = require('@wlearn/xgboost')

await loadXGB()

const dtrain = new DMatrix([[1, 2], [3, 4], [5, 6], [7, 8]])
dtrain.setLabel([3, 7, 11, 15])

const booster = new Booster({
  objective: 'reg:squarederror',
  max_depth: 3,
  verbosity: 0
}, [dtrain])

for (let i = 0; i < 50; i++) {
  booster.update(dtrain, i)
}

const preds = booster.predict(dtrain)  // Float32Array
const model = booster.saveModel()       // Uint8Array (UBJ)

booster.dispose()
dtrain.dispose()
```

### `DMatrix(data, options?)`

- `data` -- `number[][]` or `Float32Array`
- `options.nrow`, `options.ncol` -- required when `data` is `Float32Array`
- `options.missing` -- missing value indicator (default: `NaN`)
- `options.label` -- set labels at construction time
- `.setLabel(labels)` -- set target labels
- `.setWeight(weights)` -- set sample weights
- `.dispose()` -- free WASM memory

### `Booster(params, cache?)`

- `.setParam(name, value)` -- set a single parameter
- `.update(dtrain, iteration)` -- run one training round
- `.predict(dtest, options?)` -- predict, returns `Float32Array`
- `.saveModel(format?)` -- `'ubj'` (default) or `'json'`, returns `Uint8Array`
- `.dispose()` -- free WASM memory

### `Booster.loadModel(buffer)`

Load from `Uint8Array`. Returns a `Booster`.

## Resource management

WASM heap memory is not garbage collected. Call `.dispose()` on every `DMatrix`, `Booster`, and `XGBModel` when done. A `FinalizationRegistry` safety net warns if you forget, but do not rely on it.

## Cross-runtime compatibility

Models saved in Python XGBoost 3.2.0 load and predict identically in this package (verified with tolerance < 1e-4). WLRN bundles round-trip between JS and Python.

## Build from source

Requires [Emscripten](https://emscripten.org/) (emsdk) activated.

```bash
git clone --recurse-submodules https://github.com/wlearn-org/xgboost-wasm
cd xgboost-wasm
bash scripts/build-wasm.sh
node test/test.js
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## License

Apache-2.0 (same as upstream XGBoost)
