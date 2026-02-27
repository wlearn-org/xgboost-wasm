// WASM loader -- loads the XGBoost WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadXGB(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const require = createRequire(import.meta.url)
    const createXGBoost = require('../wasm/xgboost.cjs')
    wasmModule = await createXGBoost(options)
    return wasmModule
  })()

  return loading
}

export function getXGB() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadXGB() first')
  return wasmModule
}
