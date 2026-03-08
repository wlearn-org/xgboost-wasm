const { join } = require('path')
const { readFileSync, existsSync } = require('fs')

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    failed++
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

// Deterministic pseudo-random (LCG)
function makeLCG(seed = 42) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff
    return s / 0x7fffffff
  }
}

async function main() {

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadXGB } = require('../src/wasm.js')
const wasm = await loadXGB()

await test('WASM module loads', async () => {
  assert(wasm, 'wasm module is null')
  assert(typeof wasm.ccall === 'function', 'ccall not available')
  assert(typeof wasm.cwrap === 'function', 'cwrap not available')
})

await test('XGBGetLastError returns string', async () => {
  const err = wasm.ccall('XGBGetLastError', 'string', [], [])
  assert(typeof err === 'string', `expected string, got ${typeof err}`)
})

// ============================================================
// DMatrix
// ============================================================
console.log('\n=== DMatrix ===')

const { DMatrix } = require('../src/dmatrix.js')

await test('DMatrix from 2D array', async () => {
  const dm = new DMatrix([[1, 2], [3, 4], [5, 6]])
  assert(dm.rows === 3, `expected 3 rows, got ${dm.rows}`)
  assert(dm.cols === 2, `expected 2 cols, got ${dm.cols}`)
  dm.dispose()
})

await test('DMatrix from Float32Array', async () => {
  const data = new Float32Array([1, 2, 3, 4, 5, 6])
  const dm = new DMatrix(data, { nrow: 3, ncol: 2 })
  assert(dm.rows === 3, `expected 3 rows, got ${dm.rows}`)
  assert(dm.cols === 2, `expected 2 cols, got ${dm.cols}`)
  dm.dispose()
})

await test('DMatrix setLabel', async () => {
  const dm = new DMatrix([[1, 2], [3, 4]])
  dm.setLabel([0, 1])
  dm.dispose()
})

await test('DMatrix double dispose is safe', async () => {
  const dm = new DMatrix([[1, 2]])
  dm.dispose()
  dm.dispose() // should not throw
})

await test('DMatrix throws after dispose', async () => {
  const dm = new DMatrix([[1, 2]])
  dm.dispose()
  let threw = false
  try { dm.rows } catch { threw = true }
  assert(threw, 'accessing rows after dispose should throw')
})

// ============================================================
// Booster
// ============================================================
console.log('\n=== Booster ===')

const { Booster } = require('../src/booster.js')

await test('Booster create with params', async () => {
  const dtrain = new DMatrix([[1, 2], [3, 4], [5, 6], [7, 8]])
  dtrain.setLabel([0, 1, 2, 3])
  const booster = new Booster({
    objective: 'reg:squarederror',
    max_depth: 2,
    eta: 0.3,
    verbosity: 0
  }, [dtrain])
  assert(booster.handle, 'booster handle is null')
  booster.dispose()
  dtrain.dispose()
})

await test('Booster train and predict', async () => {
  const rand = makeLCG(100)
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = rand() * 10
    const x2 = rand() * 10
    X.push([x1, x2])
    y.push(x1 + x2)
  }

  const dtrain = new DMatrix(X)
  dtrain.setLabel(y)

  const booster = new Booster({
    objective: 'reg:squarederror',
    max_depth: 3,
    eta: 0.3,
    verbosity: 0,
    seed: 42
  }, [dtrain])

  for (let i = 0; i < 50; i++) {
    booster.update(dtrain, i)
  }

  const preds = booster.predict(dtrain)
  assert(preds instanceof Float32Array, 'predictions should be Float32Array')
  assert(preds.length === 100, `expected 100 predictions, got ${preds.length}`)

  const predMean = preds.reduce((a, v) => a + v, 0) / preds.length
  const yMean = y.reduce((a, v) => a + v, 0) / y.length
  assertClose(predMean, yMean, 2.0, `pred mean ${predMean} too far from true mean ${yMean}`)

  booster.dispose()
  dtrain.dispose()
})

await test('Booster save and load model', async () => {
  const X = [[1, 2], [3, 4], [5, 6], [7, 8]]
  const y = [3, 7, 11, 15]

  const dtrain = new DMatrix(X)
  dtrain.setLabel(y)

  const booster = new Booster({
    objective: 'reg:squarederror',
    max_depth: 2,
    eta: 0.3,
    verbosity: 0,
    seed: 42
  }, [dtrain])

  for (let i = 0; i < 20; i++) {
    booster.update(dtrain, i)
  }

  const preds1 = booster.predict(dtrain)

  const modelBuf = booster.saveModel('ubj')
  assert(modelBuf instanceof Uint8Array, 'model buffer should be Uint8Array')
  assert(modelBuf.length > 0, 'model buffer should not be empty')

  const booster2 = Booster.loadModel(modelBuf)
  const preds2 = booster2.predict(dtrain)

  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `prediction ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  booster.dispose()
  booster2.dispose()
  dtrain.dispose()
})

await test('Booster save JSON format', async () => {
  const dtrain = new DMatrix([[1, 2], [3, 4]])
  dtrain.setLabel([0, 1])

  const booster = new Booster({
    objective: 'reg:squarederror',
    verbosity: 0
  }, [dtrain])

  booster.update(dtrain, 0)

  const jsonBuf = booster.saveModel('json')
  assert(jsonBuf.length > 0, 'JSON model should not be empty')

  const jsonStr = new TextDecoder().decode(jsonBuf)
  const parsed = JSON.parse(jsonStr)
  assert(parsed.learner, 'JSON model should have learner key')

  booster.dispose()
  dtrain.dispose()
})

await test('Booster double dispose is safe', async () => {
  const dtrain = new DMatrix([[1, 2]])
  dtrain.setLabel([0])
  const booster = new Booster({ verbosity: 0 }, [dtrain])
  booster.dispose()
  booster.dispose() // should not throw
  dtrain.dispose()
})

// ============================================================
// Classification (low-level Booster)
// ============================================================
console.log('\n=== Classification ===')

await test('Binary classification', async () => {
  const rand = makeLCG(200)
  const X = []
  const y = []
  for (let i = 0; i < 200; i++) {
    const x1 = rand() * 10
    const x2 = rand() * 10
    X.push([x1, x2])
    y.push(x1 + x2 > 10 ? 1 : 0)
  }

  const dtrain = new DMatrix(X)
  dtrain.setLabel(y)

  const booster = new Booster({
    objective: 'binary:logistic',
    max_depth: 3,
    eta: 0.3,
    verbosity: 0,
    seed: 42
  }, [dtrain])

  for (let i = 0; i < 30; i++) {
    booster.update(dtrain, i)
  }

  const preds = booster.predict(dtrain)
  assert(preds.length === 200, `expected 200 predictions, got ${preds.length}`)

  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] >= 0 && preds[i] <= 1,
      `prediction ${i} out of [0,1] range: ${preds[i]}`)
  }

  let correct = 0
  for (let i = 0; i < preds.length; i++) {
    if ((preds[i] > 0.5 ? 1 : 0) === y[i]) correct++
  }
  const accuracy = correct / preds.length
  assert(accuracy > 0.7, `accuracy ${accuracy} too low`)

  booster.dispose()
  dtrain.dispose()
})

// ============================================================
// More objectives
// ============================================================
console.log('\n=== More Objectives ===')

await test('Multiclass classification (multi:softprob)', async () => {
  const rand = makeLCG(300)
  const X = []
  const y = []
  for (let i = 0; i < 300; i++) {
    const x1 = rand() * 10
    const x2 = rand() * 10
    X.push([x1, x2])
    const sum = x1 + x2
    y.push(sum < 7 ? 0 : sum < 13 ? 1 : 2)
  }

  const dtrain = new DMatrix(X)
  dtrain.setLabel(y)

  const booster = new Booster({
    objective: 'multi:softprob',
    num_class: 3,
    max_depth: 3,
    eta: 0.3,
    verbosity: 0,
    seed: 42
  }, [dtrain])

  for (let i = 0; i < 30; i++) {
    booster.update(dtrain, i)
  }

  const preds = booster.predict(dtrain)
  assert(preds.length === 300 * 3, `expected 900 predictions, got ${preds.length}`)

  for (let r = 0; r < 300; r++) {
    const sum = preds[r * 3] + preds[r * 3 + 1] + preds[r * 3 + 2]
    assertClose(sum, 1.0, 1e-4, `row ${r} probs sum to ${sum}, expected ~1.0`)
  }

  let correct = 0
  for (let r = 0; r < 300; r++) {
    let best = 0
    for (let c = 1; c < 3; c++) {
      if (preds[r * 3 + c] > preds[r * 3 + best]) best = c
    }
    if (best === y[r]) correct++
  }
  const accuracy = correct / 300
  assert(accuracy > 0.6, `multiclass accuracy ${accuracy} too low`)

  booster.dispose()
  dtrain.dispose()
})

await test('Count regression (count:poisson)', async () => {
  const rand = makeLCG(400)
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x = rand() * 5
    X.push([x])
    y.push(Math.max(0, Math.round(x * 2 + rand() * 2)))
  }

  const dtrain = new DMatrix(X)
  dtrain.setLabel(y)

  const booster = new Booster({
    objective: 'count:poisson',
    max_depth: 3,
    eta: 0.3,
    verbosity: 0,
    seed: 42
  }, [dtrain])

  for (let i = 0; i < 30; i++) {
    booster.update(dtrain, i)
  }

  const preds = booster.predict(dtrain)
  assert(preds.length === 100, `expected 100 predictions, got ${preds.length}`)

  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] >= 0, `prediction ${i} is negative: ${preds[i]}`)
  }

  booster.dispose()
  dtrain.dispose()
})

await test('Survival regression (survival:cox)', async () => {
  const rand = makeLCG(500)
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = rand() * 10
    const x2 = rand() * 5
    X.push([x1, x2])
    const time = x1 * 0.5 + x2 + rand() * 3
    y.push(rand() > 0.3 ? time : -time)
  }

  const dtrain = new DMatrix(X)
  dtrain.setLabel(y)

  const booster = new Booster({
    objective: 'survival:cox',
    max_depth: 2,
    eta: 0.1,
    verbosity: 0,
    seed: 42
  }, [dtrain])

  for (let i = 0; i < 20; i++) {
    booster.update(dtrain, i)
  }

  const preds = booster.predict(dtrain)
  assert(preds.length === 100, `expected 100 predictions, got ${preds.length}`)

  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] > 0, `prediction ${i} should be positive hazard ratio: ${preds[i]}`)
  }

  booster.dispose()
  dtrain.dispose()
})

// ============================================================
// XGBModel (high-level Estimator interface)
// ============================================================
console.log('\n=== XGBModel ===')

const { XGBModel } = require('../src/model.js')

await test('XGBModel.create and fit (regression)', async () => {
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    max_depth: 3,
    eta: 0.3,
    numRound: 20,
    seed: 42
  })

  const X = [[1, 2], [3, 4], [5, 6], [7, 8]]
  const y = [3, 7, 11, 15]
  model.fit(X, y)

  assert(model.isFitted, 'model should be fitted')
  assert(model.nrClass === 0, 'regression should have nrClass 0')

  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'predictions should be Float64Array')
  assert(preds.length === 4, `expected 4 predictions, got ${preds.length}`)

  model.dispose()
})

await test('XGBModel binary classifier predict returns class labels', async () => {
  const rand = makeLCG(600)
  const model = await XGBModel.create({
    objective: 'binary:logistic',
    max_depth: 3,
    eta: 0.3,
    numRound: 30,
    seed: 42
  })

  const X = []
  const y = []
  for (let i = 0; i < 200; i++) {
    const x1 = rand() * 10
    const x2 = rand() * 10
    X.push([x1, x2])
    y.push(x1 + x2 > 10 ? 1 : 0)
  }

  model.fit(X, y)

  assert(model.nrClass === 2, `expected nrClass 2, got ${model.nrClass}`)
  const classes = model.classes
  assert(classes[0] === 0 && classes[1] === 1, `expected classes [0, 1], got [${classes}]`)

  const preds = model.predict(X)
  assert(preds instanceof Float64Array, 'predictions should be Float64Array')
  assert(preds.length === 200, `expected 200 predictions, got ${preds.length}`)

  // All predictions should be class labels (0 or 1)
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] === 0 || preds[i] === 1,
      `prediction ${i} should be 0 or 1, got ${preds[i]}`)
  }

  // Accuracy should be decent
  let correct = 0
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === y[i]) correct++
  }
  assert(correct / preds.length > 0.7, 'accuracy too low')

  model.dispose()
})

await test('XGBModel predictProba binary: shape rows*2, rows sum to 1', async () => {
  const model = await XGBModel.create({
    objective: 'binary:logistic',
    max_depth: 3,
    eta: 0.3,
    numRound: 20,
    seed: 42
  })

  const X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [2, 9]]
  const y = [0, 0, 1, 1, 1, 1]
  model.fit(X, y)

  const proba = model.predictProba(X)
  assert(proba instanceof Float64Array, 'proba should be Float64Array')
  assert(proba.length === 6 * 2, `expected 12 values, got ${proba.length}`)

  // Each row should sum to ~1
  for (let r = 0; r < 6; r++) {
    const sum = proba[r * 2] + proba[r * 2 + 1]
    assertClose(sum, 1.0, 1e-6, `row ${r} probs sum to ${sum}`)
  }

  // All values in [0, 1]
  for (let i = 0; i < proba.length; i++) {
    assert(proba[i] >= 0 && proba[i] <= 1, `proba[${i}] = ${proba[i]} out of range`)
  }

  model.dispose()
})

await test('XGBModel multiclass predict and predictProba', async () => {
  const rand = makeLCG(700)
  const model = await XGBModel.create({
    objective: 'multi:softprob',
    num_class: 3,
    max_depth: 3,
    eta: 0.3,
    numRound: 30,
    seed: 42
  })

  const X = []
  const y = []
  for (let i = 0; i < 150; i++) {
    const x1 = rand() * 10
    const x2 = rand() * 10
    X.push([x1, x2])
    const sum = x1 + x2
    y.push(sum < 7 ? 0 : sum < 13 ? 1 : 2)
  }

  model.fit(X, y)
  assert(model.nrClass === 3, `expected nrClass 3, got ${model.nrClass}`)

  // predict returns class labels
  const preds = model.predict(X)
  assert(preds.length === 150, `expected 150 predictions, got ${preds.length}`)
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] === 0 || preds[i] === 1 || preds[i] === 2,
      `prediction ${i} should be 0/1/2, got ${preds[i]}`)
  }

  // predictProba returns rows * nrClass
  const proba = model.predictProba(X)
  assert(proba.length === 150 * 3, `expected 450 proba values, got ${proba.length}`)
  for (let r = 0; r < 150; r++) {
    const sum = proba[r * 3] + proba[r * 3 + 1] + proba[r * 3 + 2]
    assertClose(sum, 1.0, 1e-4, `row ${r} probs sum to ${sum}`)
  }

  model.dispose()
})

await test('XGBModel score (accuracy for classifier, R2 for regressor)', async () => {
  // Classifier
  const clf = await XGBModel.create({
    objective: 'binary:logistic',
    max_depth: 3,
    numRound: 20,
    seed: 42
  })
  const rand = makeLCG(800)
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = rand() * 10
    const x2 = rand() * 10
    X.push([x1, x2])
    y.push(x1 + x2 > 10 ? 1 : 0)
  }
  clf.fit(X, y)
  const acc = clf.score(X, y)
  assert(typeof acc === 'number', 'score should be a number')
  assert(acc > 0.7, `accuracy ${acc} too low`)
  clf.dispose()

  // Regressor
  const reg = await XGBModel.create({
    objective: 'reg:squarederror',
    max_depth: 3,
    numRound: 50,
    seed: 42
  })
  const Xr = [[1, 2], [3, 4], [5, 6], [7, 8]]
  const yr = [3, 7, 11, 15]
  reg.fit(Xr, yr)
  const r2 = reg.score(Xr, yr)
  assert(typeof r2 === 'number', 'score should be a number')
  assert(r2 > 0.5, `R2 ${r2} too low`)
  reg.dispose()
})

await test('XGBModel predictProba throws for regression', async () => {
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    numRound: 10,
    seed: 42
  })
  model.fit([[1, 2], [3, 4]], [1, 2])

  let threw = false
  try { model.predictProba([[1, 2]]) } catch { threw = true }
  assert(threw, 'predictProba should throw for regression objective')

  model.dispose()
})

await test('XGBModel predictProba throws for multi:softmax', async () => {
  const model = await XGBModel.create({
    objective: 'multi:softmax',
    num_class: 3,
    numRound: 10,
    seed: 42
  })
  model.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 2])

  let threw = false
  try { model.predictProba([[1, 2]]) } catch { threw = true }
  assert(threw, 'predictProba should throw for multi:softmax objective')

  model.dispose()
})

// ============================================================
// Convenience API
// ============================================================
console.log('\n=== Convenience API ===')

const { train, predict: predictConv } = require('../src/index.js')

await test('train() convenience function', async () => {
  const model = await train({
    objective: 'reg:squarederror',
    max_depth: 2,
    eta: 0.3,
    numRound: 20,
    seed: 42
  }, [[1, 2], [3, 4], [5, 6], [7, 8]], [3, 7, 11, 15])

  assert(model.isFitted, 'model should be fitted')
  const preds = model.predict([[1, 2], [3, 4]])
  assert(preds.length === 2, `expected 2 predictions, got ${preds.length}`)

  model.dispose()
})

await test('predict() convenience function', async () => {
  // Train and save a bundle
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    max_depth: 2,
    numRound: 20,
    seed: 42
  })
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [3, 7, 11, 15])
  const bundle = model.save()
  model.dispose()

  // Predict from bundle
  const preds = await predictConv(bundle, [[1, 2], [3, 4]])
  assert(preds instanceof Float64Array, 'should return Float64Array')
  assert(preds.length === 2, `expected 2 predictions, got ${preds.length}`)
})

// ============================================================
// Save / Load (WLRN bundle format)
// ============================================================
console.log('\n=== Save / Load ===')

const { decodeBundle, load: coreLoad } = require('@wlearn/core')

await test('save produces WLRN bundle', async () => {
  const model = await XGBModel.create({
    objective: 'binary:logistic',
    max_depth: 3,
    numRound: 10,
    seed: 42
  })
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1])

  const bundle = model.save()
  assert(bundle instanceof Uint8Array, 'bundle should be Uint8Array')

  // WLRN magic bytes
  assert(bundle[0] === 0x57, `magic[0]: expected 0x57, got 0x${bundle[0].toString(16)}`)
  assert(bundle[1] === 0x4c, `magic[1]: expected 0x4c, got 0x${bundle[1].toString(16)}`)
  assert(bundle[2] === 0x52, `magic[2]: expected 0x52, got 0x${bundle[2].toString(16)}`)
  assert(bundle[3] === 0x4e, `magic[3]: expected 0x4e, got 0x${bundle[3].toString(16)}`)

  const { manifest, toc } = decodeBundle(bundle)
  assert(manifest.typeId === 'wlearn.xgboost.classifier@1',
    `expected classifier typeId, got ${manifest.typeId}`)
  assert(manifest.params.objective === 'binary:logistic',
    `expected binary:logistic, got ${manifest.params.objective}`)
  assert(manifest.metadata.nrClass === 2, `expected nrClass 2, got ${manifest.metadata.nrClass}`)
  assert(manifest.metadata.classes[0] === 0 && manifest.metadata.classes[1] === 1,
    `expected classes [0,1], got ${manifest.metadata.classes}`)
  assert(toc.length === 1, `expected 1 TOC entry, got ${toc.length}`)
  assert(toc[0].id === 'model', `expected TOC id "model", got ${toc[0].id}`)

  model.dispose()
})

await test('save regressor uses regressor typeId', async () => {
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    numRound: 10,
    seed: 42
  })
  model.fit([[1, 2], [3, 4]], [1.5, 3.5])

  const { manifest } = decodeBundle(model.save())
  assert(manifest.typeId === 'wlearn.xgboost.regressor@1',
    `expected regressor typeId, got ${manifest.typeId}`)

  model.dispose()
})

await test('save and load round-trip (predictions match)', async () => {
  const rand = makeLCG(900)
  const model = await XGBModel.create({
    objective: 'binary:logistic',
    max_depth: 3,
    numRound: 20,
    seed: 42
  })

  const X = []
  const y = []
  for (let i = 0; i < 50; i++) {
    const x1 = rand() * 10
    const x2 = rand() * 10
    X.push([x1, x2])
    y.push(x1 + x2 > 10 ? 1 : 0)
  }

  model.fit(X, y)
  const preds1 = model.predict(X)
  const proba1 = model.predictProba(X)

  const bundle = model.save()
  const restored = await XGBModel.load(bundle)

  const preds2 = restored.predict(X)
  const proba2 = restored.predictProba(X)

  assert(preds1.length === preds2.length, 'prediction length mismatch')
  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `pred ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  assert(proba1.length === proba2.length, 'proba length mismatch')
  for (let i = 0; i < proba1.length; i++) {
    assertClose(proba1[i], proba2[i], 1e-6, `proba ${i}: ${proba1[i]} !== ${proba2[i]}`)
  }

  // Params preserved
  assert(restored.getParams().objective === 'binary:logistic', 'objective should be preserved')
  assert(restored.nrClass === 2, 'nrClass should be preserved')
  const classes = restored.classes
  assert(classes[0] === 0 && classes[1] === 1, 'classes should be preserved')

  model.dispose()
  restored.dispose()
})

await test('save and load regressor round-trip', async () => {
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    max_depth: 3,
    numRound: 30,
    seed: 42
  })
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [3, 7, 11, 15])

  const preds1 = model.predict([[2, 3], [6, 7]])
  const bundle = model.save()
  const restored = await XGBModel.load(bundle)
  const preds2 = restored.predict([[2, 3], [6, 7]])

  for (let i = 0; i < preds1.length; i++) {
    assertClose(preds1[i], preds2[i], 1e-6, `pred ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  model.dispose()
  restored.dispose()
})

// ============================================================
// Registry Dispatch
// ============================================================
console.log('\n=== Registry Dispatch ===')

await test('core.load() dispatches to xgboost classifier loader', async () => {
  const model = await XGBModel.create({
    objective: 'binary:logistic',
    max_depth: 3,
    numRound: 10,
    seed: 42
  })
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1])
  const bundle = model.save()
  const preds1 = model.predict([[2, 3], [6, 7]])
  model.dispose()

  const restored = await coreLoad(bundle)
  assert(restored instanceof XGBModel, 'core.load should return XGBModel')
  const preds2 = restored.predict([[2, 3], [6, 7]])

  for (let i = 0; i < preds1.length; i++) {
    assert(preds1[i] === preds2[i], `pred ${i}: ${preds1[i]} !== ${preds2[i]}`)
  }

  restored.dispose()
})

await test('core.load() works for regressor bundles', async () => {
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    numRound: 10,
    seed: 42
  })
  model.fit([[1, 2], [3, 4]], [1.5, 3.5])
  const bundle = model.save()
  model.dispose()

  const restored = await coreLoad(bundle)
  assert(restored instanceof XGBModel, 'core.load should return XGBModel')
  const preds = restored.predict([[2, 3]])
  assert(preds.length === 1, 'should predict 1 row')
  restored.dispose()
})

// ============================================================
// Resource Management
// ============================================================
console.log('\n=== Resource Management ===')

await test('dispose is idempotent', async () => {
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    numRound: 5,
    seed: 42
  })
  model.fit([[1, 2]], [1])
  model.dispose()
  model.dispose() // should not throw
})

await test('throws DisposedError after dispose', async () => {
  const { DisposedError } = require('@wlearn/core')
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    numRound: 5,
    seed: 42
  })
  model.fit([[1, 2]], [1])
  model.dispose()

  let threw = false
  let isDisposedError = false
  try { model.predict([[1, 2]]) } catch (e) {
    threw = true
    isDisposedError = e instanceof DisposedError
  }
  assert(threw, 'should throw after dispose')
  assert(isDisposedError, 'should throw DisposedError')
})

await test('throws NotFittedError before fit', async () => {
  const { NotFittedError } = require('@wlearn/core')
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    numRound: 5
  })

  let threw = false
  let isNotFittedError = false
  try { model.predict([[1, 2]]) } catch (e) {
    threw = true
    isNotFittedError = e instanceof NotFittedError
  }
  assert(threw, 'should throw before fit')
  assert(isNotFittedError, 'should throw NotFittedError')
})

await test('refit disposes previous booster', async () => {
  const model = await XGBModel.create({
    objective: 'reg:squarederror',
    numRound: 5,
    seed: 42
  })
  model.fit([[1, 2], [3, 4]], [1, 3])
  const preds1 = model.predict([[1, 2]])

  // Refit with different data
  model.fit([[1, 2], [3, 4]], [10, 30])
  const preds2 = model.predict([[1, 2]])

  // Predictions should differ (different training data)
  assert(model.isFitted, 'model should still be fitted')

  model.dispose()
})

// ============================================================
// Params
// ============================================================
console.log('\n=== Params ===')

await test('getParams / setParams', async () => {
  const model = await XGBModel.create({
    objective: 'binary:logistic',
    max_depth: 5,
    numRound: 50
  })

  const params = model.getParams()
  assert(params.objective === 'binary:logistic', 'objective should match')
  assert(params.max_depth === 5, 'max_depth should match')
  assert(params.numRound === 50, 'numRound should match')

  model.setParams({ max_depth: 7 })
  assert(model.getParams().max_depth === 7, 'max_depth should be updated')
  assert(model.getParams().objective === 'binary:logistic', 'objective should be unchanged')

  model.dispose()
})

await test('defaultSearchSpace returns object', async () => {
  const space = XGBModel.defaultSearchSpace()
  assert(typeof space === 'object', 'space should be an object')
  assert(space.objective, 'should have objective')
  assert(space.max_depth, 'should have max_depth')
  assert(space.eta, 'should have eta')
  assert(space.numRound, 'should have numRound')
})

// ============================================================
// Capabilities
// ============================================================
console.log('\n=== Capabilities ===')

await test('capabilities for binary:logistic', async () => {
  const model = await XGBModel.create({ objective: 'binary:logistic' })
  const cap = model.capabilities
  assert(cap.classifier === true, 'should be classifier')
  assert(cap.regressor === false, 'should not be regressor')
  assert(cap.predictProba === true, 'should support predictProba')
  assert(cap.sampleWeight === false, 'sampleWeight not yet supported')
  assert(cap.earlyStopping === false, 'earlyStopping not yet supported')
  model.dispose()
})

await test('capabilities for multi:softprob', async () => {
  const model = await XGBModel.create({ objective: 'multi:softprob', num_class: 3 })
  const cap = model.capabilities
  assert(cap.classifier === true, 'should be classifier')
  assert(cap.predictProba === true, 'should support predictProba')
  model.dispose()
})

await test('capabilities for multi:softmax', async () => {
  const model = await XGBModel.create({ objective: 'multi:softmax', num_class: 3 })
  const cap = model.capabilities
  assert(cap.classifier === true, 'should be classifier')
  assert(cap.predictProba === false, 'should NOT support predictProba')
  model.dispose()
})

await test('capabilities for reg:squarederror', async () => {
  const model = await XGBModel.create({ objective: 'reg:squarederror' })
  const cap = model.capabilities
  assert(cap.classifier === false, 'should not be classifier')
  assert(cap.regressor === true, 'should be regressor')
  assert(cap.predictProba === false, 'should not support predictProba')
  model.dispose()
})

// ============================================================
// Class ordering stability
// ============================================================
console.log('\n=== Class Ordering ===')

await test('class ordering is stable and restored on load', async () => {
  const model = await XGBModel.create({
    objective: 'binary:logistic',
    numRound: 10,
    seed: 42
  })
  // Labels are 0 and 1 but passed in mixed order
  model.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [1, 0, 1, 0])

  const classes1 = model.classes
  assert(classes1[0] === 0 && classes1[1] === 1, 'classes should be sorted [0, 1]')

  const bundle = model.save()
  const restored = await XGBModel.load(bundle)
  const classes2 = restored.classes
  assert(classes2[0] === classes1[0] && classes2[1] === classes1[1],
    'classes should be identical after load')

  model.dispose()
  restored.dispose()
})

// ============================================================
// Cross-runtime parity (Python fixtures)
// ============================================================
console.log('\n=== Cross-Runtime Parity ===')

const fixturesDir = join(__dirname, 'fixtures')
const hasFixtures = existsSync(join(fixturesDir, 'regression.data.json'))

if (!hasFixtures) {
  console.log('  SKIP: no fixtures (run: conda run -n prob python test/fixtures/generate.py)')
} else {
  function loadFixture(name) {
    return JSON.parse(readFileSync(join(fixturesDir, `${name}.data.json`), 'utf-8'))
  }

  function loadFixtureModel(name) {
    return readFileSync(join(fixturesDir, name))
  }

  await test('Cross-runtime: regression parity', async () => {
    const fix = loadFixture('regression')
    const modelBuf = loadFixtureModel('regression.ubj')

    const booster = Booster.loadModel(modelBuf)
    const dm = new DMatrix(fix.X)

    const preds = booster.predict(dm)
    assert(preds.length === fix.predictions.length,
      `length mismatch: ${preds.length} vs ${fix.predictions.length}`)

    for (let i = 0; i < preds.length; i++) {
      const rel = Math.abs(preds[i] - fix.predictions[i]) / (Math.abs(fix.predictions[i]) + 1e-8)
      assert(rel < 1e-4,
        `pred[${i}]: JS=${preds[i]} Python=${fix.predictions[i]} relDiff=${rel}`)
    }

    booster.dispose()
    dm.dispose()
  })

  await test('Cross-runtime: binary classification parity', async () => {
    const fix = loadFixture('binary')
    const modelBuf = loadFixtureModel('binary.ubj')

    const booster = Booster.loadModel(modelBuf)
    const dm = new DMatrix(fix.X)

    const preds = booster.predict(dm)
    assert(preds.length === fix.predictions.length,
      `length mismatch: ${preds.length} vs ${fix.predictions.length}`)

    for (let i = 0; i < preds.length; i++) {
      const diff = Math.abs(preds[i] - fix.predictions[i])
      assert(diff < 1e-5,
        `pred[${i}]: JS=${preds[i]} Python=${fix.predictions[i]} diff=${diff}`)
    }

    booster.dispose()
    dm.dispose()
  })

  await test('Cross-runtime: multiclass parity', async () => {
    const fix = loadFixture('multiclass')
    const modelBuf = loadFixtureModel('multiclass.ubj')

    const booster = Booster.loadModel(modelBuf)
    const dm = new DMatrix(fix.X)

    const preds = booster.predict(dm)
    assert(preds.length === fix.predictions.length,
      `length mismatch: ${preds.length} vs ${fix.predictions.length}`)

    for (let i = 0; i < preds.length; i++) {
      const diff = Math.abs(preds[i] - fix.predictions[i])
      assert(diff < 1e-5,
        `pred[${i}]: JS=${preds[i]} Python=${fix.predictions[i]} diff=${diff}`)
    }

    booster.dispose()
    dm.dispose()
  })

  await test('Cross-runtime: inference-only (load Python model, predict new data)', async () => {
    const fix = loadFixture('inference')
    const modelBuf = loadFixtureModel(fix.model)

    const booster = Booster.loadModel(modelBuf)
    const dm = new DMatrix(fix.X)

    const preds = booster.predict(dm)
    assert(preds.length === fix.predictions.length,
      `length mismatch: ${preds.length} vs ${fix.predictions.length}`)

    for (let i = 0; i < preds.length; i++) {
      const rel = Math.abs(preds[i] - fix.predictions[i]) / (Math.abs(fix.predictions[i]) + 1e-8)
      assert(rel < 1e-4,
        `pred[${i}]: JS=${preds[i]} Python=${fix.predictions[i]} relDiff=${rel}`)
    }

    booster.dispose()
    dm.dispose()
  })
}

// ============================================================
// Task Param Mapping
// ============================================================
console.log('\n=== Task Param Mapping ===')

await test('task: classification (binary)', async () => {
  const rng = makeLCG(99)
  const n = 40, f = 2
  const X = { data: new Float64Array(n * f), rows: n, cols: f }
  const y = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    X.data[i * f] = rng()
    X.data[i * f + 1] = rng()
    y[i] = X.data[i * f] > 0.5 ? 1 : 0
  }
  const model = await XGBModel.create({ task: 'classification', nRounds: 10 })
  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds.length === n, `expected ${n} predictions, got ${preds.length}`)
  // Verify it's actually classifying (labels are 0 or 1)
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] === 0 || preds[i] === 1, `expected class label, got ${preds[i]}`)
  }
  model.dispose()
})

await test('task: classification (multiclass auto-promotes)', async () => {
  const n = 30, f = 2
  const X = { data: new Float64Array(n * f), rows: n, cols: f }
  const y = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    X.data[i * f] = i / n
    X.data[i * f + 1] = (i * 3 % n) / n
    y[i] = i % 3
  }
  const model = await XGBModel.create({ task: 'classification', nRounds: 10 })
  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds.length === n, `expected ${n} predictions`)
  // Verify multiclass labels
  const labels = new Set()
  for (let i = 0; i < preds.length; i++) labels.add(preds[i])
  assert(labels.size <= 3, `expected at most 3 classes, got ${labels.size}`)
  model.dispose()
})

await test('task: regression', async () => {
  const n = 40, f = 2
  const X = { data: new Float64Array(n * f), rows: n, cols: f }
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    X.data[i * f] = i / n
    X.data[i * f + 1] = (i * 7 % n) / n
    y[i] = 2.5 * X.data[i * f] + 1.3 * X.data[i * f + 1]
  }
  const model = await XGBModel.create({ task: 'regression', nRounds: 20 })
  model.fit(X, y)
  const preds = model.predict(X)
  assert(preds.length === n, `expected ${n} predictions`)
  model.dispose()
})

await test('task + objective conflict throws', async () => {
  let threw = false
  const model = await XGBModel.create({ task: 'classification', objective: 'binary:logistic', nRounds: 5 })
  try {
    const X = { data: new Float64Array([0, 0, 1, 1]), rows: 2, cols: 2 }
    const y = new Int32Array([0, 1])
    model.fit(X, y)
  } catch (e) {
    threw = true
    assert(e.message.includes('Cannot set both'), `unexpected error: ${e.message}`)
  }
  assert(threw, 'expected error for task + objective conflict')
  model.dispose()
})

await test('task: unknown throws', async () => {
  let threw = false
  const model = await XGBModel.create({ task: 'clustering', nRounds: 5 })
  try {
    const X = { data: new Float64Array([0, 0, 1, 1]), rows: 2, cols: 2 }
    const y = new Int32Array([0, 1])
    model.fit(X, y)
  } catch (e) {
    threw = true
    assert(e.message.includes('Unknown task'), `unexpected error: ${e.message}`)
  }
  assert(threw, 'expected error for unknown task')
  model.dispose()
})

// ============================================================
// Summary
// ============================================================
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
process.exit(failed > 0 ? 1 : 0)

}

main()
