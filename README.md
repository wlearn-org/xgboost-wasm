# @wlearn/xgboost

XGBoost v3.4.1 compiled to WebAssembly. Gradient-boosted trees, random forests, classification, and regression in browsers and Node.js.

Part of [wlearn](https://wlearn.org) ([GitHub](https://github.com/wlearn-org), [all packages](https://github.com/wlearn-org/wlearn#repository-structure)). Based on [XGBoost v3.4.1](https://github.com/dmlc/xgboost) (Apache-2.0). CommonJS.

## Install

```bash
npm install @wlearn/xgboost
```

## Quick start

```js
const { readFileSync, writeFileSync } = require('fs')
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
const preds = model.predict([[2, 3], [6, 7]])  // Int32Array class labels

// Probabilities
const probs = model.predictProba([[2, 3], [6, 7]])  // Float64Array (nrow * nclass)

// Score
const accuracy = model.score([[2, 3], [6, 7]], [0, 1])

// Save / load
writeFileSync('xgboost.wlrn', model.save())
const model2 = await XGBModel.load(readFileSync('xgboost.wlrn'))
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
- `num_class` -- number of classes for multiclass objectives; inferred from the
  fitted labels when omitted and required to match them when supplied
- `subsample` -- row subsampling ratio (default: `1.0`)
- `colsample_bytree` -- column subsampling ratio (default: `1.0`)
- `lambda` -- L2 regularization (default: `1.0`)
- `alpha` -- L1 regularization (default: `0.0`)
- `num_parallel_tree` -- trees per round, >1 for random forest mode (default: `1`)
- `verbosity` -- 0 = silent, 1 = warning, 2 = info (default: `0`)
- `coerce` -- input coercion: `'auto'` | `'warn'` | `'error'` (default: `'auto'`)

### `model.fit(X, y)`

Train on data. Returns `this`.
- `X` -- `number[][]` or `{ data: Float64Array, rows, cols }`
- `y` -- `number[]` or `Float64Array`. Classification labels must be int32
  values; they need not be contiguous. The wrapper maps the sorted public labels
  to XGBoost's internal `0..K-1` indices and maps predictions back.

### `model.predict(X)`

Returns classifier labels as `Int32Array` and regression values as
`Float64Array`.

### `model.predictProba(X)`

Returns `Float64Array` of shape `nrow * nclass` (row-major probabilities).
Columns follow the sorted order in `model.classes`. Available for
`binary:logistic` and `multi:softprob` objectives.

### `model.score(X, y)`

Returns accuracy (classification) or R-squared (regression).

### `model.save()` / `XGBModel.load(buffer)`

Save to / load from `Uint8Array` (WLRN bundle with UBJ model blob).

### `model.dispose()`

Release WASM memory immediately. Use in long-running apps, workers, cross-validation, and AutoML loops. Idempotent.

### `model.getParams()` / `model.setParams(p)`

Get/set hyperparameters. Enables AutoML grid search and cloning.

### `XGBModel.defaultSearchSpace()`

Returns default hyperparameter search space for AutoML.

## Objective coverage

The high-level `XGBModel` test suite exercises:
- `reg:squarederror` -- regression
- `binary:logistic` -- binary classification (probabilities)
- `multi:softprob` -- multiclass classification (probabilities)
- `multi:softmax` -- multiclass classification (class labels)

The low-level `Booster` smoke suite also exercises `count:poisson` and
`survival:cox`. The unified estimator intentionally rejects ranking and
survival objectives because it does not yet define ranking groups or survival
metrics; use `Booster` directly for those tasks. Other upstream objectives are
not claimed as high-level `XGBModel` contracts until their prediction and score
semantics are tested.

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
- `.dispose()` -- release WASM memory

### `Booster(params, cache?)`

- `.setParam(name, value)` -- set a single parameter
- `.update(dtrain, iteration)` -- run one training round
- `.predict(dtest, options?)` -- predict, returns `Float32Array`
- `.saveModel(format?)` -- `'ubj'` (default) or `'json'`, returns `Uint8Array`
- `.dispose()` -- release WASM memory

### `Booster.loadModel(buffer)`

Load from `Uint8Array`. Returns a `Booster`.

## Classifier migration from 0.2

Version 0.3 returns classifier labels as `Int32Array` and preserves arbitrary
finite int32 public labels through an internal ordinal encoding. Version 0.2
returned common classifier labels in a floating-point typed array and did not
reliably support noncontiguous public labels. Regression predictions remain
floating point.

## Resource management

Use `.dispose()` when creating and discarding many `DMatrix`, `Booster`, or `XGBModel` objects so WASM memory is released promptly.

## Cross-runtime compatibility

The native XGBoost 3.4.1 fixtures cover classification and squared-error, expectile, quantile and absolute-error regression (prediction tolerance < 1e-5). Older 3.2 WLRN model bytes are preserved through load/save until a successful refit. WLRN bundles round-trip between JS and Python.

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
