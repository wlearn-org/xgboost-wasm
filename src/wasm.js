// WASM loader — loads the XGBoost WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

export async function loadXGB(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const { default: createXGBoost } = await import('../wasm/xgboost.js')
    wasmModule = await createXGBoost({
      // locateFile resolves .wasm path for bundlers that rewrite import URLs
      locateFile(path) {
        if (options.wasmUrl) return options.wasmUrl
        if (typeof import.meta?.url !== 'undefined') {
          return new URL('../wasm/' + path, import.meta.url).href
        }
        return path
      }
    })
    return wasmModule
  })()

  return loading
}

export function getXGB() {
  if (!wasmModule) throw new Error('WASM not loaded — call loadXGB() first')
  return wasmModule
}
