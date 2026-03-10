const { loadXGB, getXGB } = require('./wasm.js')
const { DMatrix } = require('./dmatrix.js')
const { Booster } = require('./booster.js')
const { XGBModel: XGBModelImpl } = require('./model.js')
const { createModelClass } = require('@wlearn/core')

const XGBModel = createModelClass(XGBModelImpl, XGBModelImpl, { name: 'XGBModel', load: loadXGB })

// Convenience: create, fit, return fitted model
async function train(params, X, y) {
  const model = await XGBModel.create(params)
  await model.fit(X, y)
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
