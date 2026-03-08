// WASM loader -- loads the XGBoost WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadXGB(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const createXGBoost = require('../wasm/xgboost.cjs')
    wasmModule = await createXGBoost(options)
    return wasmModule
  })()

  return loading
}

function getXGB() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadXGB() first')
  return wasmModule
}

module.exports = { loadXGB, getXGB }
