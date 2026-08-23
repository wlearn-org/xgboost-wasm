const { loadXGB, getXGB } = require('./wasm.js')
const { DMatrix } = require('./dmatrix.js')
const { Booster } = require('./booster.js')
const {
  normalizeY,
  encodeBundle, validateBundle,
  register,
  DisposedError, NotFittedError
} = require('@wlearn/core')

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

function validateUnifiedObjective(objective) {
  if (typeof objective !== 'string' || objective.length === 0) {
    throw new Error('objective must be a non-empty XGBoost objective string')
  }
  if (objective.startsWith('rank:') || objective.startsWith('survival:')) {
    throw new Error(
      `The high-level XGBModel does not implement ranking groups or survival ` +
      `metrics for objective "${objective}"; use the low-level Booster API.`
    )
  }
}

// XGBoost params that are wlearn-only (not passed to Booster)
const WLEARN_PARAMS = new Set(['numRound', 'coerce', 'task'])

// --- Internal sentinel for load path ---
const LOAD_SENTINEL = Symbol('load')

// --- XGBModel ---

class XGBModel {
  #booster = null
  #freed = false
  #boosterRef = null
  #params = {}
  #fitted = false
  #nrClass = 0
  #classes = null
  #warned = false

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

    const { data: xData, rows, cols } = this.#normalizeX(X)
    const yNorm = normalizeY(y)
    if (yNorm.length !== rows) {
      throw new Error(`y length (${yNorm.length}) does not match X rows (${rows})`)
    }
    const fitParams = this.#resolveFitParams(yNorm)
    const objective = fitParams.objective || 'reg:squarederror'
    validateUnifiedObjective(objective)

    let yTrain
    let classes = null
    let nrClass = 0

    // XGBoost classifier objectives consume ordinal labels 0..K-1. Preserve
    // public labels separately and train on their sorted ordinal indices.
    if (CLASSIFIER_OBJECTIVES.has(objective)) {
      const unique = new Set()
      for (let i = 0; i < yNorm.length; i++) {
        const v = yNorm[i]
        if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) {
          throw new Error(`Classifier labels must be int32 values, got ${v} at index ${i}`)
        }
        unique.add(v)
      }
      const sorted = [...unique].sort((a, b) => a - b)
      classes = new Int32Array(sorted)
      nrClass = sorted.length
      if (nrClass < 2) {
        throw new Error(`Classification requires at least 2 classes, got ${nrClass}`)
      }
      const classIndex = new Map(sorted.map((label, index) => [label, index]))
      yTrain = new Float32Array(yNorm.length)
      for (let i = 0; i < yNorm.length; i++) {
        yTrain[i] = classIndex.get(yNorm[i])
      }
    } else {
      for (let i = 0; i < yNorm.length; i++) {
        if (!Number.isFinite(yNorm[i])) {
          throw new Error(`Regression labels must be finite, got ${yNorm[i]} at index ${i}`)
        }
      }
      yTrain = yNorm instanceof Float32Array ? yNorm : new Float32Array(yNorm)
    }

    // Extract wlearn-only params, pass rest to XGBoost
    const numRound = fitParams.numRound ?? 100
    if (!Number.isSafeInteger(numRound) || numRound < 1) {
      throw new Error('numRound must be a positive safe integer')
    }
    const xgbParams = {}
    for (const [key, val] of Object.entries(fitParams)) {
      if (!WLEARN_PARAMS.has(key)) xgbParams[key] = val
    }
    // Default verbosity to 0 (silent) unless user set it
    if (!('verbosity' in xgbParams)) xgbParams.verbosity = 0

    // Auto-set num_class for multi-class objectives
    const obj = xgbParams.objective || ''
    if (obj.startsWith('binary:') && nrClass !== 2) {
      throw new Error(
        `Binary objective requires exactly 2 classes, got ${nrClass}`
      )
    }
    if (obj.startsWith('multi:')) {
      if ('num_class' in xgbParams && xgbParams.num_class !== nrClass) {
        throw new Error(
          `num_class (${xgbParams.num_class}) does not match fitted classes (${nrClass})`
        )
      }
      xgbParams.num_class = nrClass
    }

    const dm = new DMatrix(xData, { nrow: rows, ncol: cols })
    let booster = null
    try {
      dm.setLabel(yTrain)
      booster = new Booster(xgbParams, [dm])
      for (let i = 0; i < numRound; i++) {
        booster.update(dm, i)
      }
    } catch (error) {
      if (booster) booster.dispose()
      throw error
    } finally {
      dm.dispose()
    }

    const previousBooster = this.#booster
    if (previousBooster) {
      previousBooster.dispose()
      if (this.#boosterRef) this.#boosterRef[0] = null
      if (leakRegistry) leakRegistry.unregister(this)
    }

    this.#booster = booster
    this.#params = fitParams
    this.#classes = classes
    this.#nrClass = nrClass
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
    let rawPreds
    try {
      rawPreds = this.#booster.predict(dm)
    } finally {
      dm.dispose()
    }

    const obj = this.#params.objective || 'reg:squarederror'

    if (this.#isClassifier()) {
      const result = new Int32Array(rows)

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
          if (idx < 0 || idx >= this.#classes.length) {
            throw new Error(`XGBoost returned invalid class index ${rawPreds[i]} at row ${i}`)
          }
          result[i] = this.#classes[idx]
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
    let rawPreds
    try {
      rawPreds = this.#booster.predict(dm)
    } finally {
      dm.dispose()
    }

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
    if (yArr.length !== preds.length) {
      throw new Error(`y length (${yArr.length}) does not match prediction rows (${preds.length})`)
    }

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
    const identity = this.#booster.modelIdentity()
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
          objective: this.#params.objective || 'reg:squarederror',
          nFeatures: identity.numFeature
        }
      },
      [{ id: 'model', data: rawBytes }]
    )
  }

  static async load(bytes) {
    const { manifest, toc, blobs } = validateBundle(bytes)
    return XGBModel._fromBundle(manifest, toc, blobs)
  }

  static async _fromBundle(manifest, toc, blobs) {
    await loadXGB()

    const classifier = manifest.typeId === 'wlearn.xgboost.classifier@1'
    const regressor = manifest.typeId === 'wlearn.xgboost.regressor@1'
    if (!classifier && !regressor) {
      throw new Error(`XGBModel cannot load bundle type ${JSON.stringify(manifest.typeId)}`)
    }

    const params = manifest.params || {}
    const objective = params.objective || 'reg:squarederror'
    validateUnifiedObjective(objective)
    const meta = manifest.metadata || {}
    if (meta.objective !== objective) {
      throw new Error(`${manifest.typeId} objective metadata does not match params`)
    }
    if (classifier) {
      const classes = meta.classes
      if (!CLASSIFIER_OBJECTIVES.has(objective) ||
          !Number.isInteger(meta.nrClass) || meta.nrClass < 2 ||
          !Array.isArray(classes) || classes.length !== meta.nrClass ||
          !classes.every(value => Number.isInteger(value) &&
            value >= -2147483648 && value <= 2147483647) ||
          classes.some((value, index) => index > 0 && value <= classes[index - 1])) {
        throw new Error(`${manifest.typeId} has invalid classifier metadata`)
      }
    } else if (CLASSIFIER_OBJECTIVES.has(objective) ||
               meta.nrClass !== 0 ||
               !(meta.classes == null ||
                 (Array.isArray(meta.classes) && meta.classes.length === 0))) {
      throw new Error(`${manifest.typeId} has invalid regressor metadata`)
    }

    if (!Array.isArray(toc) || toc.length !== 1 || toc[0].id !== 'model' ||
        toc[0].mediaType !== 'application/octet-stream') {
      throw new Error('XGBoost bundle must contain exactly one model artifact')
    }
    const entry = toc[0]
    const raw = blobs.subarray(entry.offset, entry.offset + entry.length)

    const booster = Booster.loadModel(raw)
    try {
      const identity = booster.modelIdentity()
      if (identity.objective !== objective) {
        throw new Error(`${manifest.typeId} model objective does not match manifest`)
      }
      if (identity.numTarget !== 1) {
        throw new Error(`${manifest.typeId} model must contain exactly one target`)
      }
      if (classifier && objective.startsWith('multi:') &&
          identity.numClass !== meta.nrClass) {
        throw new Error(`${manifest.typeId} model class count does not match manifest`)
      }
      if (meta.nFeatures != null &&
          (!Number.isSafeInteger(meta.nFeatures) || meta.nFeatures < 1 ||
           meta.nFeatures !== identity.numFeature)) {
        throw new Error(`${manifest.typeId} model feature count does not match manifest`)
      }
    } catch (error) {
      booster.dispose()
      throw error
    }

    return new XGBModel(LOAD_SENTINEL, booster, {
      params,
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
    if (Object.prototype.hasOwnProperty.call(p, 'task') &&
        p.task !== this.#params.task &&
        !Object.prototype.hasOwnProperty.call(p, 'objective')) {
      delete this.#params.objective
      delete this.#params.num_class
    }
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
    const coerce = this.#params.coerce || 'auto'
    if (!['auto', 'warn', 'error'].includes(coerce)) {
      throw new Error(`coerce must be "auto", "warn", or "error", got ${JSON.stringify(coerce)}`)
    }
    // Fast path: typed matrix { data, rows, cols }
    if (X && typeof X === 'object' && !Array.isArray(X) && X.data != null) {
      const { data, rows, cols } = X
      if (!Number.isSafeInteger(rows) || !Number.isSafeInteger(cols) ||
          rows < 1 || cols < 1 || !Number.isSafeInteger(rows * cols)) {
        throw new Error(`Invalid matrix dimensions: rows=${rows}, cols=${cols}`)
      }
      if (typeof data.length !== 'number' || data.length !== rows * cols) {
        throw new Error(`data.length (${data?.length}) !== rows * cols (${rows * cols})`)
      }
      if (data instanceof Float32Array) return { data, rows, cols }
      if (coerce === 'error' && !(data instanceof Float64Array)) {
        throw new Error('Input coercion disabled; typed matrix data must be Float32Array or Float64Array')
      }
      try {
        return { data: new Float32Array(data), rows, cols }
      } catch (error) {
        throw new Error(`Matrix data cannot be converted to Float32Array: ${error.message}`)
      }
    }

    // Slow path: number[][]
    if (Array.isArray(X)) {
      if (X.length === 0 || !Array.isArray(X[0]) || X[0].length === 0) {
        throw new Error('X must be a non-empty rectangular number[][]')
      }
      const rows = X.length
      const cols = X[0].length
      if (coerce === 'error') {
        throw new Error(
          'Input coercion disabled; pass { data: Float32Array|Float64Array, rows, cols } instead of number[][]'
        )
      }
      const data = new Float32Array(rows * cols)
      for (let i = 0; i < rows; i++) {
        if (!Array.isArray(X[i]) || X[i].length !== cols) {
          throw new Error(`X must be rectangular; row ${i} has length ${X[i]?.length}, expected ${cols}`)
        }
        for (let j = 0; j < cols; j++) {
          const value = X[i][j]
          if (typeof value !== 'number') {
            throw new Error(`X[${i}][${j}] must be a number`)
          }
          data[i * cols + j] = value
        }
      }
      if (coerce === 'warn' && !this.#warned) {
        this.#warned = true
        console.warn(
          `@wlearn/xgboost: Converted number[][] to Float32Array ` +
          `(copied ${(data.byteLength / 1024).toFixed(1)} KB, shape ${rows}x${cols}).`
        )
      }
      return { data, rows, cols }
    }

    throw new Error('X must be number[][] or { data: TypedArray, rows, cols }')
  }

  #ensureFitted(requireFit = true) {
    if (this.#freed) throw new DisposedError('XGBModel has been disposed.')
    if (requireFit && !this.#fitted) throw new NotFittedError('XGBModel is not fitted. Call fit() first.')
  }

  #resolveFitParams(y) {
    const params = { ...this.#params }
    const task = params.task
    if (!task) return params
    // If objective is already set, it takes precedence over task
    if (params.objective) return params
    if (task === 'classification') {
      // Count unique values in y to decide binary vs multiclass
      const yNorm = normalizeY(y)
      const unique = new Set()
      for (let i = 0; i < yNorm.length; i++) unique.add(yNorm[i])
      if (unique.size > 2) {
        params.objective = 'multi:softprob'
        params.num_class = unique.size
      } else {
        params.objective = 'binary:logistic'
      }
    } else if (task === 'regression') {
      params.objective = 'reg:squarederror'
    } else {
      throw new Error(`Unknown task: '${task}'. Use 'classification' or 'regression'.`)
    }
    return params
  }

  #isClassifier() {
    const obj = this.#params.objective || 'reg:squarederror'
    return CLASSIFIER_OBJECTIVES.has(obj)
  }
}

// --- Register loaders with @wlearn/core ---

register('wlearn.xgboost.classifier@1', async (m, t, b) => XGBModel._fromBundle(m, t, b))
register('wlearn.xgboost.regressor@1', async (m, t, b) => XGBModel._fromBundle(m, t, b))

module.exports = { XGBModel }
