export { loadXGB, getXGB } from './wasm.js'
export { DMatrix } from './dmatrix.js'
export { Booster } from './booster.js'
export { XGBModel } from './model.js'

// Convenience: create, fit, return fitted model
export async function train(params, X, y) {
  const model = await (await import('./model.js')).XGBModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
export async function predict(bundleBytes, X) {
  const model = await (await import('./model.js')).XGBModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}
