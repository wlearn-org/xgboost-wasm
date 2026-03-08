const { loadXGB, getXGB } = require('./wasm.js')
const { DMatrix } = require('./dmatrix.js')
const { Booster } = require('./booster.js')
const { XGBModel } = require('./model.js')

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await XGBModel.create(params)
  model.fit(X, y)
  return model
}

// Convenience: load WLRN bundle and predict, auto-disposes model
async function predict(bundleBytes, X) {
  const model = await XGBModel.load(bundleBytes)
  const result = model.predict(X)
  model.dispose()
  return result
}

module.exports = { loadXGB, getXGB, DMatrix, Booster, XGBModel, train, predict }
