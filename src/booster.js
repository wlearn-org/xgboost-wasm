const { getXGB } = require('./wasm.js')

// FinalizationRegistry safety net
const registry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@wlearn/xgboost: Booster was not disposed — calling free() automatically. This is a bug in your code.')
      freeFn(ptr[0])
    }
  })
  : null

// Helper: write C string to WASM heap, call fn, free
function withCString(wasm, str, fn) {
  const bytes = new TextEncoder().encode(str + '\0')
  const ptr = wasm._malloc(bytes.length)
  wasm.HEAPU8.set(bytes, ptr)
  try {
    return fn(ptr)
  } finally {
    wasm._free(ptr)
  }
}

function getLastError(wasm) {
  return wasm.ccall('XGBGetLastError', 'string', [], [])
}

// Internal sentinel for loadModel path
const LOAD_SENTINEL = Symbol('load')

class Booster {
  #handle = null
  #freed = false
  #ptrRef = null

  constructor(params = {}, cache = []) {
    // Internal path: loadModel passes a pre-created handle via sentinel
    if (params === LOAD_SENTINEL) {
      this.#handle = cache // cache holds the handle in this path
      this.#freed = false
      this.#ptrRef = [this.#handle]
      if (registry) {
        registry.register(this, {
          ptr: this.#ptrRef,
          freeFn: (h) => getXGB()._XGBoosterFree(h)
        }, this)
      }
      return
    }

    const wasm = getXGB()

    // XGBoosterCreate(DMatrixHandle cache[], bst_ulong len, BoosterHandle *out)
    // bst_ulong = uint64_t, passed as BigInt with WASM_BIGINT=1
    const cacheLen = cache.length
    const cachePtr = wasm._malloc(Math.max(cacheLen, 1) * 4)
    for (let i = 0; i < cacheLen; i++) {
      wasm.setValue(cachePtr + i * 4, cache[i].handle, 'i32')
    }

    const outPtr = wasm._malloc(4)
    const ret = wasm._XGBoosterCreate(cachePtr, BigInt(cacheLen), outPtr)

    if (ret !== 0) {
      wasm._free(cachePtr)
      wasm._free(outPtr)
      throw new Error(`XGBoosterCreate failed: ${getLastError(wasm)}`)
    }

    this.#handle = wasm.getValue(outPtr, 'i32')
    wasm._free(cachePtr)
    wasm._free(outPtr)

    // Register for leak detection
    this.#ptrRef = [this.#handle]
    if (registry) {
      registry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getXGB()._XGBoosterFree(h)
      }, this)
    }

    // Set params
    for (const [key, value] of Object.entries(params)) {
      this.setParam(key, String(value))
    }
  }

  get handle() {
    if (this.#freed) throw new Error('Booster already disposed')
    return this.#handle
  }

  setParam(name, value) {
    const wasm = getXGB()
    const ret = withCString(wasm, name, (namePtr) =>
      withCString(wasm, String(value), (valuePtr) =>
        wasm._XGBoosterSetParam(this.handle, namePtr, valuePtr)
      )
    )

    if (ret !== 0) {
      throw new Error(`XGBoosterSetParam(${name}) failed: ${getLastError(wasm)}`)
    }
  }

  update(dtrain, iteration) {
    const wasm = getXGB()
    const ret = wasm._XGBoosterUpdateOneIter(this.handle, iteration, dtrain.handle)

    if (ret !== 0) {
      throw new Error(`XGBoosterUpdateOneIter failed: ${getLastError(wasm)}`)
    }
  }

  predict(dtest, { ntreeLimit = 0, type = 0 } = {}) {
    const wasm = getXGB()

    // v3.2.0 requires iteration_begin/iteration_end (not iteration_range)
    const config = JSON.stringify({
      type,
      iteration_begin: 0,
      iteration_end: ntreeLimit,
      strict_shape: false,
      training: false
    })

    const outShapePtr = wasm._malloc(4)
    const outDimPtr = wasm._malloc(8)
    const outResultPtr = wasm._malloc(4)

    const ret = withCString(wasm, config, (configPtr) =>
      wasm._XGBoosterPredictFromDMatrix(this.handle, dtest.handle, configPtr,
        outShapePtr, outDimPtr, outResultPtr)
    )

    if (ret !== 0) {
      wasm._free(outShapePtr)
      wasm._free(outDimPtr)
      wasm._free(outResultPtr)
      throw new Error(`XGBoosterPredictFromDMatrix failed: ${getLastError(wasm)}`)
    }

    const shapeArrPtr = wasm.getValue(outShapePtr, 'i32')
    const ndim = wasm.getValue(outDimPtr, 'i32')

    let totalLen = 1
    for (let i = 0; i < ndim; i++) {
      totalLen *= wasm.getValue(shapeArrPtr + i * 8, 'i32')
    }

    const resultPtr = wasm.getValue(outResultPtr, 'i32')
    const result = new Float32Array(totalLen)
    for (let i = 0; i < totalLen; i++) {
      result[i] = wasm.getValue(resultPtr + i * 4, 'float')
    }

    wasm._free(outShapePtr)
    wasm._free(outDimPtr)
    wasm._free(outResultPtr)

    return result
  }

  saveModel(format = 'ubj') {
    const wasm = getXGB()
    const config = JSON.stringify({ format })

    const outLenPtr = wasm._malloc(8)
    const outDptrPtr = wasm._malloc(4)

    const ret = withCString(wasm, config, (configPtr) =>
      wasm._XGBoosterSaveModelToBuffer(this.handle, configPtr, outLenPtr, outDptrPtr)
    )

    if (ret !== 0) {
      wasm._free(outLenPtr)
      wasm._free(outDptrPtr)
      throw new Error(`XGBoosterSaveModelToBuffer failed: ${getLastError(wasm)}`)
    }

    const len = wasm.getValue(outLenPtr, 'i32')
    const dptr = wasm.getValue(outDptrPtr, 'i32')

    const result = new Uint8Array(len)
    result.set(wasm.HEAPU8.subarray(dptr, dptr + len))

    wasm._free(outLenPtr)
    wasm._free(outDptrPtr)

    return result
  }

  static loadModel(buffer) {
    const wasm = getXGB()

    // Create an empty booster
    const outPtr = wasm._malloc(4)
    let ret = wasm._XGBoosterCreate(0, 0n, outPtr)

    if (ret !== 0) {
      wasm._free(outPtr)
      throw new Error(`XGBoosterCreate failed: ${getLastError(wasm)}`)
    }

    const handle = wasm.getValue(outPtr, 'i32')
    wasm._free(outPtr)

    // Copy buffer to WASM heap and load
    const buf = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
    const bufPtr = wasm._malloc(buf.length)
    wasm.HEAPU8.set(buf, bufPtr)

    ret = wasm._XGBoosterLoadModelFromBuffer(handle, bufPtr, BigInt(buf.length))
    wasm._free(bufPtr)

    if (ret !== 0) {
      wasm._XGBoosterFree(handle)
      throw new Error(`XGBoosterLoadModelFromBuffer failed: ${getLastError(wasm)}`)
    }

    // Use sentinel constructor path
    return new Booster(LOAD_SENTINEL, handle)
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    const wasm = getXGB()
    wasm._XGBoosterFree(this.#handle)

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (registry) registry.unregister(this)

    this.#handle = null
  }
}

module.exports = { Booster }
