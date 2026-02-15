export { loadXGB, getXGB } from './wasm.js'
export { DMatrix } from './dmatrix.js'
export { Booster } from './booster.js'

// Convenience: train in one call
export async function train(params, dtrain, numRound = 100) {
  const { loadXGB } = await import('./wasm.js')
  await loadXGB()

  const { Booster } = await import('./booster.js')
  const booster = new Booster(params, [dtrain])
  for (let i = 0; i < numRound; i++) {
    booster.update(dtrain, i)
  }
  return booster
}

// Convenience: load model and predict, auto-disposes intermediates
export async function predict(modelBuffer, data) {
  const { loadXGB } = await import('./wasm.js')
  await loadXGB()

  const { Booster } = await import('./booster.js')
  const { DMatrix } = await import('./dmatrix.js')

  const booster = Booster.loadModel(modelBuffer)
  const dm = new DMatrix(data)
  const result = booster.predict(dm)
  dm.dispose()
  booster.dispose()
  return result
}
