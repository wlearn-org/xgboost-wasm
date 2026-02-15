import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

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
  if (diff > tol) throw new Error(msg || `expected ${a} ≈ ${b} (diff=${diff}, tol=${tol})`)
}

// ============================================================
// WASM loading
// ============================================================
console.log('\n=== WASM Loading ===')

const { loadXGB } = await import('../src/wasm.js')
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

const { DMatrix } = await import('../src/dmatrix.js')

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

const { Booster } = await import('../src/booster.js')

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
  // Simple regression: y ≈ x1 + x2
  const X = []
  const y = []
  for (let i = 0; i < 100; i++) {
    const x1 = Math.random() * 10
    const x2 = Math.random() * 10
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

  // Check predictions are reasonable (not all zeros, roughly in range)
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

  // Save model
  const modelBuf = booster.saveModel('ubj')
  assert(modelBuf instanceof Uint8Array, 'model buffer should be Uint8Array')
  assert(modelBuf.length > 0, 'model buffer should not be empty')

  // Load model
  const booster2 = Booster.loadModel(modelBuf)
  const preds2 = booster2.predict(dtrain)

  // Same-runtime round-trip: exact match
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

  // Verify it's valid JSON
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
// Convenience API
// ============================================================
console.log('\n=== Convenience API ===')

const { train, predict } = await import('../src/index.js')

await test('train() convenience function', async () => {
  const dtrain = new DMatrix([[1, 2], [3, 4], [5, 6], [7, 8]])
  dtrain.setLabel([3, 7, 11, 15])

  const booster = await train({
    objective: 'reg:squarederror',
    max_depth: 2,
    eta: 0.3,
    verbosity: 0,
    seed: 42
  }, dtrain, 20)

  const preds = booster.predict(dtrain)
  assert(preds.length === 4, `expected 4 predictions, got ${preds.length}`)

  booster.dispose()
  dtrain.dispose()
})

await test('predict() convenience function', async () => {
  // Train a model first
  const dtrain = new DMatrix([[1, 2], [3, 4], [5, 6], [7, 8]])
  dtrain.setLabel([3, 7, 11, 15])
  const booster = await train({
    objective: 'reg:squarederror',
    max_depth: 2,
    verbosity: 0,
    seed: 42
  }, dtrain, 20)
  const modelBuf = booster.saveModel()
  booster.dispose()
  dtrain.dispose()

  // Predict with convenience function
  const preds = await predict(modelBuf, [[1, 2], [3, 4]])
  assert(preds.length === 2, `expected 2 predictions, got ${preds.length}`)
})

// ============================================================
// Classification
// ============================================================
console.log('\n=== Classification ===')

await test('Binary classification', async () => {
  const X = []
  const y = []
  for (let i = 0; i < 200; i++) {
    const x1 = Math.random() * 10
    const x2 = Math.random() * 10
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

  // All predictions should be probabilities [0, 1]
  for (let i = 0; i < preds.length; i++) {
    assert(preds[i] >= 0 && preds[i] <= 1,
      `prediction ${i} out of [0,1] range: ${preds[i]}`)
  }

  // Compute accuracy — should be decent
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
// Summary
// ============================================================
console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`)
process.exit(failed > 0 ? 1 : 0)
