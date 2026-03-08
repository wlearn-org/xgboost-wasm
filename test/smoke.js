const { loadXGB } = require('../src/wasm.js')
const { DMatrix } = require('../src/dmatrix.js')
const { Booster } = require('../src/booster.js')

async function main() {

const wasm = await loadXGB()

const dm = new DMatrix([[1, 2], [3, 4]])
dm.setLabel([0, 1])

const b = new Booster({ objective: 'reg:squarederror', verbosity: 0 }, [dm])
b.update(dm, 0)
console.log('Training done')

// Correct config format for v3.2.0
const config = JSON.stringify({
  type: 0,
  iteration_begin: 0,
  iteration_end: 0,
  strict_shape: false,
  training: false
})
console.log('Config:', config)

const configBytes = new TextEncoder().encode(config + '\0')
const configPtr = wasm._malloc(configBytes.length)
wasm.HEAPU8.set(configBytes, configPtr)

const outShapePtr = wasm._malloc(4)
const outDimPtr = wasm._malloc(8)
const outResultPtr = wasm._malloc(4)

const ret = wasm._XGBoosterPredictFromDMatrix(
  b.handle, dm.handle, configPtr,
  outShapePtr, outDimPtr, outResultPtr
)
console.log('ret:', ret)

if (ret !== 0) {
  console.log('ERROR:', wasm.ccall('XGBGetLastError', 'string', [], []))
} else {
  const ndim = wasm.getValue(outDimPtr, 'i32')
  const shapePtr = wasm.getValue(outShapePtr, 'i32')
  console.log('ndim:', ndim)

  let total = 1
  for (let i = 0; i < ndim; i++) {
    const d = wasm.getValue(shapePtr + i * 8, 'i32')
    console.log(`shape[${i}]:`, d)
    total *= d
  }

  const resultPtr = wasm.getValue(outResultPtr, 'i32')
  for (let i = 0; i < total; i++) {
    console.log(`pred[${i}]:`, wasm.getValue(resultPtr + i * 4, 'float'))
  }
}

wasm._free(configPtr)
wasm._free(outShapePtr)
wasm._free(outDimPtr)
wasm._free(outResultPtr)
b.dispose()
dm.dispose()
console.log('Done!')

}

main()
