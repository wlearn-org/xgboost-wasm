import { loadXGB, getXGB } from './wasm.js'
import { DMatrix } from './dmatrix.js'
import { Booster } from './booster.js'
import {
  normalizeY,
  encodeBundle, decodeBundle,
  register,
  DisposedError, NotFittedError
} from '@wlearn/core'

// FinalizationRegistry safety net -- warns if dispose() was never called
const leakRegistry = typeof FinalizationRegistry !== 'undefined'
  ? new FinalizationRegistry(({ ref, freeFn }) => {
    if (ref[0]) {
      console.warn('@wlearn/xgboost: XGBModel was not disposed -- calling free() automatically. This is a bug in your code.')
      freeFn(ref[0])
    }
  })
  : null

// --- Objective classification ---

const CLASSIFIER_OBJECTIVES = new Set([
  'binary:logistic', 'binary:logitraw', 'binary:hinge',
  'multi:softmax', 'multi:softprob'
])

const PROBA_OBJECTIVES = new Set([
  'binary:logistic', 'multi:softprob'
])

// XGBoost params that are wlearn-only (not passed to Booster)
const WLEARN_PARAMS = new Set(['numRound', 'coerce', 'task'])

// --- Internal sentinel for load path ---
const LOAD_SENTINEL = Symbol('load')

// --- XGBModel ---

export class XGBModel {
  #booster = null
  #freed = false
  #boosterRef = null
  #params = {}
  #fitted = false
  #nrClass = 0
  #classes = null

  constructor(handle, params, extra) {
    if (handle === LOAD_SENTINEL) {
      // Load path: handle is sentinel, params is Booster, extra is { params, nrClass, classes }
      this.#booster = params
      this.#params = extra.params || {}
      this.#nrClass = extra.nrClass || 0
      this.#classes = extra.classes ? new Int32Array(extra.classes) : null
      this.#fitted = true
      this.#freed = false
      this.#boosterRef = [this.#booster]
      if (leakRegistry) {
        leakRegistry.register(this, {
          ref: this.#boosterRef,
          freeFn: (b) => { try { b.dispose() } catch {} }
        }, this)
      }
    } else {
      // Normal construction from create()
      this.#params = handle || {}
      this.#freed = false
    }
  }

  static async create(params = {}) {
    await loadXGB()
    return new XGBModel(params)
  }

  // --- Estimator interface ---

  fit(X, y) {
    this.#ensureFitted(false)

    // Map task param to objective if needed
    this.#resolveTask(y)

    // Dispose previous booster if refitting
    if (this.#booster) {
      this.#booster.dispose()
      this.#booster = null
      if (this.#boosterRef) this.#boosterRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)

    // For classifiers: validate integer labels and extract classes
    if (this.#isClassifier()) {
      const unique = new Set()
      for (let i = 0; i < yNorm.length; i++) {
        const v = yNorm[i]
        if (v !== Math.floor(v)) {
          throw new Error(`Classifier labels must be integers, got ${v} at index ${i}`)
        }
        unique.add(v)
      }
      const sorted = [...unique].sort((a, b) => a - b)
      this.#classes = new Int32Array(sorted)
      this.#nrClass = sorted.length
    } else {
      this.#classes = null
      this.#nrClass = 0
    }

    // Convert y to Float32 for DMatrix
    const yF32 = yNorm instanceof Float32Array ? yNorm : new Float32Array(yNorm)

    if (yF32.length !== rows) {
      throw new Error(`y length (${yF32.length}) does not match X rows (${rows})`)
    }

    // Create DMatrix
    const dm = new DMatrix(xData, { nrow: rows, ncol: cols })
    dm.setLabel(yF32)

    // Extract wlearn-only params, pass rest to XGBoost
    const numRound = this.#params.numRound || 100
    const xgbParams = {}
    for (const [key, val] of Object.entries(this.#params)) {
      if (!WLEARN_PARAMS.has(key)) xgbParams[key] = val
    }
    // Default verbosity to 0 (silent) unless user set it
    if (!('verbosity' in xgbParams)) xgbParams.verbosity = 0

    // Auto-set num_class for multi-class objectives
    const obj = xgbParams.objective || ''
    if (obj.startsWith('multi:') && !('num_class' in xgbParams) && this.#nrClass > 0) {
      xgbParams.num_class = this.#nrClass
    }

    // Create Booster and train
    const booster = new Booster(xgbParams, [dm])
    for (let i = 0; i < numRound; i++) {
      booster.update(dm, i)
    }

    dm.dispose()

    this.#booster = booster
    this.#fitted = true

    this.#boosterRef = [this.#booster]
    if (leakRegistry) {
      leakRegistry.register(this, {
        ref: this.#boosterRef,
        freeFn: (b) => { try { b.dispose() } catch {} }
      }, this)
    }

    return this
  }

  predict(X) {
    this.#ensureFitted()
    const { data: xData, rows, cols } = this.#normalizeX(X)

    const dm = new DMatrix(xData, { nrow: rows, ncol: cols })
    const rawPreds = this.#booster.predict(dm)
    dm.dispose()

    const obj = this.#params.objective || 'reg:squarederror'

    if (this.#isClassifier()) {
      const result = new Float64Array(rows)

      if (obj === 'binary:logistic') {
        // Raw is P(class=1), threshold at 0.5
        for (let i = 0; i < rows; i++) {
          result[i] = this.#classes[rawPreds[i] > 0.5 ? 1 : 0]
        }
      } else if (obj === 'multi:softprob') {
        // Raw is rows * nrClass probabilities, argmax
        const nc = this.#nrClass
        for (let i = 0; i < rows; i++) {
          let best = 0
          for (let c = 1; c < nc; c++) {
            if (rawPreds[i * nc + c] > rawPreds[i * nc + best]) best = c
          }
          result[i] = this.#classes[best]
        }
      } else if (obj === 'multi:softmax') {
        // Raw is class indices (0-based)
        for (let i = 0; i < rows; i++) {
          const idx = Math.round(rawPreds[i])
          result[i] = this.#classes[idx] ?? rawPreds[i]
        }
      } else {
        // binary:logitraw, binary:hinge -- threshold at 0
        for (let i = 0; i < rows; i++) {
          result[i] = this.#classes[rawPreds[i] > 0 ? 1 : 0]
        }
      }

      return result
    }

    // Regression: return raw values as Float64
    return new Float64Array(rawPreds)
  }

  predictProba(X) {
    this.#ensureFitted()
    const obj = this.#params.objective || 'reg:squarederror'

    if (!PROBA_OBJECTIVES.has(obj)) {
      throw new Error(`predictProba requires binary:logistic or multi:softprob objective, got "${obj}"`)
    }

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const dm = new DMatrix(xData, { nrow: rows, ncol: cols })
    const rawPreds = this.#booster.predict(dm)
    dm.dispose()

    if (obj === 'binary:logistic') {
      // XGBoost returns P(class=1). Expand to rows * 2: [P(class=0), P(class=1)]
      const result = new Float64Array(rows * 2)
      for (let i = 0; i < rows; i++) {
        const p1 = rawPreds[i]
        result[i * 2] = 1 - p1
        result[i * 2 + 1] = p1
      }
      return result
    }

    // multi:softprob -- already rows * nrClass
    return new Float64Array(rawPreds)
  }

  score(X, y) {
    const preds = this.predict(X)
    const yArr = normalizeY(y)

    if (!this.#isClassifier()) {
      // R-squared
      let ssRes = 0, ssTot = 0, yMean = 0
      for (let i = 0; i < yArr.length; i++) yMean += yArr[i]
      yMean /= yArr.length
      for (let i = 0; i < yArr.length; i++) {
        ssRes += (yArr[i] - preds[i]) ** 2
        ssTot += (yArr[i] - yMean) ** 2
      }
      return ssTot === 0 ? 0 : 1 - ssRes / ssTot
    }

    // Accuracy
    let correct = 0
    for (let i = 0; i < preds.length; i++) {
      if (preds[i] === yArr[i]) correct++
    }
    return correct / preds.length
  }

  // --- Model I/O ---

  save() {
    this.#ensureFitted()
    const rawBytes = this.#booster.saveModel('ubj')
    const typeId = this.#isClassifier()
      ? 'wlearn.xgboost.classifier@1'
      : 'wlearn.xgboost.regressor@1'
    return encodeBundle(
      {
        typeId,
        params: this.getParams(),
        metadata: {
          nrClass: this.#nrClass,
          classes: this.#classes ? Array.from(this.#classes) : [],
          objective: this.#params.objective || 'reg:squarederror'
        }
      },
      [{ id: 'model', data: rawBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = decodeBundle(bytes)
    return XGBModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadXGB()

    const entry = toc.find(e => e.id === 'model')
    if (!entry) throw new Error('Bundle missing "model" artifact')
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)

    const booster = Booster.loadModel(raw)

    const meta = manifest.metadata || {}
    return new XGBModel(LOAD_SENTINEL, booster, {
      params: manifest.params || {},
      nrClass: meta.nrClass || 0,
      classes: meta.classes || null
    })
  }

  dispose() {
    if (this.#freed) return
    this.#freed = true

    if (this.#booster) {
      this.#booster.dispose()
    }

    if (this.#boosterRef) this.#boosterRef[0] = null
    if (leakRegistry) leakRegistry.unregister(this)

    this.#booster = null
    this.#fitted = false
  }

  // --- Params ---

  getParams() {
    return { ...this.#params }
  }

  setParams(p) {
    Object.assign(this.#params, p)
    return this
  }

  static defaultSearchSpace() {
    return {
      objective: { type: 'categorical', values: ['binary:logistic', 'reg:squarederror'] },
      max_depth: { type: 'int_uniform', low: 3, high: 10 },
      eta: { type: 'log_uniform', low: 0.01, high: 0.3 },
      numRound: { type: 'int_uniform', low: 50, high: 500 },
      subsample: { type: 'uniform', low: 0.5, high: 1.0 },
      colsample_bytree: { type: 'uniform', low: 0.5, high: 1.0 },
      min_child_weight: { type: 'log_uniform', low: 1, high: 10 },
      lambda: { type: 'log_uniform', low: 1e-3, high: 10 },
      alpha: { type: 'log_uniform', low: 1e-3, high: 10 }
    }
  }

  // --- Inspection ---

  get nrClass() {
    return this.#nrClass
  }

  get classes() {
    return this.#classes ? Int32Array.from(this.#classes) : new Int32Array(0)
  }

  get isFitted() {
    return this.#fitted && !this.#freed
  }

  get capabilities() {
    const obj = this.#params.objective || 'reg:squarederror'
    const isCls = CLASSIFIER_OBJECTIVES.has(obj)
    return {
      classifier: isCls,
      regressor: !isCls,
      predictProba: PROBA_OBJECTIVES.has(obj),
      decisionFunction: false,
      sampleWeight: false,
      csr: false,
      earlyStopping: false
    }
  }

  get probaDim() {
    if (!this.isFitted) return 0
    const obj = this.#params.objective || 'reg:squarederror'
    if (obj === 'binary:logistic') return 2
    if (obj === 'multi:softprob') return this.#nrClass
    return 0
  }

  // --- Private helpers ---

  #normalizeX(X) {
    // Fast path: typed matrix { data, rows, cols }
    if (X && typeof X === 'object' && !Array.isArray(X) && X.data) {
      const { data, rows, cols } = X
      if (data instanceof Float32Array) return { data, rows, cols }
      return { data: new Float32Array(data), rows, cols }
    }

    // Slow path: number[][]
    if (Array.isArray(X) && Array.isArray(X[0])) {
      const rows = X.length
      const cols = X[0].length
      const data = new Float32Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          data[i * cols + j] = X[i][j]
        }
      }
      return { data, rows, cols }
    }

    throw new Error('X must be number[][] or { data: TypedArray, rows, cols }')
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('XGBModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('XGBModel is not fitted. Call fit() first.')
  }

  #resolveTask(y) {
    const task = this.#params.task
    if (!task) return
    if (this.#params.objective) {
      throw new Error("Cannot set both 'task' and 'objective'. Use one or the other.")
    }
    if (task === 'classification') {
      // Count unique values in y to decide binary vs multiclass
      const yNorm = normalizeY(y)
      const unique = new Set()
      for (let i = 0; i < yNorm.length; i++) unique.add(yNorm[i])
      if (unique.size > 2) {
        this.#params.objective = 'multi:softprob'
        this.#params.num_class = unique.size
      } else {
        this.#params.objective = 'binary:logistic'
      }
    } else if (task === 'regression') {
      this.#params.objective = 'reg:squarederror'
    } else {
      throw new Error(`Unknown task: '${task}'. Use 'classification' or 'regression'.`)
    }
  }

  #isClassifier() {
    const obj = this.#params.objective || 'reg:squarederror'
    return CLASSIFIER_OBJECTIVES.has(obj)
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.xgboost.classifier@1', async (m, t, b) => XGBModel._fromBundle(m, t, b))
register('wlearn.xgboost.regressor@1', async (m, t, b) => XGBModel._fromBundle(m, t, b))
