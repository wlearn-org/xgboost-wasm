import { getXGB } from './wasm.js'

// FinalizationRegistry safety net — warns if dispose() was never called
const registry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ptr, freeFn }) => {
    if (ptr[0]) {
      console.warn('@statsim/xgb: DMatrix was not disposed — calling free() automatically. This is a bug in your code.')
      freeFn(ptr[0])
    }
  })
  : null

// Helper: write a C string to WASM heap, call fn, free
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

export class DMatrix {
  #handle = null
  #freed = false
  #ptrRef = null

  constructor(data, { nrow, ncol, missing = NaN, label } = {}) {
    const wasm = getXGB()

    // Flatten input if 2D array
    let flat, rows, cols
    if (Array.isArray(data) && Array.isArray(data[0])) {
      rows = data.length
      cols = data[0].length
      flat = new Float32Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          flat[i * cols + j] = data[i][j]
        }
      }
    } else if (data instanceof Float32Array) {
      if (!nrow || !ncol) throw new Error('nrow and ncol required for Float32Array input')
      flat = data
      rows = nrow
      cols = ncol
    } else {
      throw new Error('data must be Float32Array or number[][]')
    }

    // Allocate and copy data to WASM heap
    const dataBytes = flat.length * 4
    const dataPtr = wasm._malloc(dataBytes)
    wasm.HEAPF32.set(flat, dataPtr / 4)

    // Allocate output handle pointer (i32)
    const outPtr = wasm._malloc(4)

    // XGDMatrixCreateFromMat(float *data, bst_ulong nrow, bst_ulong ncol, float missing, DMatrixHandle *out)
    // Legalized: (i32 data, i32 nrow_lo, i32 nrow_hi, i32 ncol_lo, i32 ncol_hi, f32 missing, i32 out)
    const ret = wasm._XGDMatrixCreateFromMat(dataPtr, rows, 0, cols, 0, missing, outPtr)
    if (ret !== 0) {
      wasm._free(dataPtr)
      wasm._free(outPtr)
      throw new Error(`XGDMatrixCreateFromMat failed: ${getLastError(wasm)}`)
    }

    this.#handle = wasm.getValue(outPtr, 'i32')
    wasm._free(dataPtr)
    wasm._free(outPtr)

    // Register for leak detection
    this.#ptrRef = [this.#handle]
    if (registry) {
      registry.register(this, {
        ptr: this.#ptrRef,
        freeFn: (h) => getXGB()._XGDMatrixFree(h)
      }, this)
    }

    if (label) this.setLabel(label)
  }

  get handle() {
    if (this.#freed) throw new Error('DMatrix already disposed')
    return this.#handle
  }

  get rows() {
    const wasm = getXGB()
    // XGDMatrixNumRow(DMatrixHandle, bst_ulong *out)
    // out is a pointer to uint64_t — read low 32 bits
    const outPtr = wasm._malloc(8)
    wasm._XGDMatrixNumRow(this.handle, outPtr)
    const val = wasm.getValue(outPtr, 'i32') // low 32 bits is enough
    wasm._free(outPtr)
    return val
  }

  get cols() {
    const wasm = getXGB()
    const outPtr = wasm._malloc(8)
    wasm._XGDMatrixNumCol(this.handle, outPtr)
    const val = wasm.getValue(outPtr, 'i32')
    wasm._free(outPtr)
    return val
  }

  setLabel(labels) {
    this.#setFloatInfo('label', labels)
  }

  setWeight(weights) {
    this.#setFloatInfo('weight', weights)
  }

  #setFloatInfo(field, values) {
    const wasm = getXGB()
    const arr = values instanceof Float32Array ? values : new Float32Array(values)
    const ptr = wasm._malloc(arr.length * 4)
    wasm.HEAPF32.set(arr, ptr / 4)

    // XGDMatrixSetFloatInfo(DMatrixHandle, const char *field, const float *array, bst_ulong len)
    // Legalized: (i32 handle, i32 field_str, i32 array_ptr, i32 len_lo, i32 len_hi)
    const ret = withCString(wasm, field, (fieldPtr) =>
      wasm._XGDMatrixSetFloatInfo(this.handle, fieldPtr, ptr, arr.length, 0)
    )

    wasm._free(ptr)

    if (ret !== 0) {
      throw new Error(`XGDMatrixSetFloatInfo(${field}) failed: ${getLastError(wasm)}`)
    }
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    const wasm = getXGB()
    wasm._XGDMatrixFree(this.#handle)

    if (this.#ptrRef) this.#ptrRef[0] = null
    if (registry) registry.unregister(this)

    this.#handle = null
  }
}
