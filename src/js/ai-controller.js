import { CONFIG, HARD_DIFFICULTY_RULES } from './config.js';
import { normalizeState } from './normalization.js';
import { CSS_INPUT_PROPERTIES, modelProfile } from './model-profiles.js';
import { cssNumberReader } from './css-numbers.js';

let controllerSequence = 0;
export function scopeModelCss(css, scopeId) {
  if (!/^nn-\d+$/.test(scopeId)) throw new Error('Invalid neural model scope.');
  return css.replaceAll('.css-neural-model', `[data-neural-sink="${scopeId}"]`);
}

export function validateMetadata(metadata, profileId = 'speed-516') {
  const profile = modelProfile(profileId);
  if (JSON.stringify(metadata.architecture) !== JSON.stringify(profile.architecture) ||
      metadata.modelProfile !== profile.id) throw new Error('The model belongs to a different architecture.');
  if (metadata.trainingDifficulty !== undefined && !['normal', 'hard'].includes(metadata.trainingDifficulty)) throw new Error('Invalid model training difficulty.');
  if (metadata.trainingDifficulty === 'hard' && !Object.entries(HARD_DIFFICULTY_RULES).every(([key, value]) => metadata.difficultyConfig?.[key] === value)) throw new Error('The hard model speed rules do not match this game.');
  if (metadata.version !== profile.version || metadata.parityVerified !== true ||
      !Number.isInteger(metadata.parity?.caseCount) || metadata.parity.caseCount < 20 ||
      metadata.parity?.tolerance !== 1e-4 || !Number.isFinite(metadata.parity?.maximumProbabilityError) ||
      metadata.parity.maximumProbabilityError < 0 || metadata.parity.maximumProbabilityError >= 1e-4) {
    throw new Error('This model has not passed CSS/Python parity validation.');
  }
  if (JSON.stringify(metadata.inputNames) !== JSON.stringify(profile.inputNames) ||
      !Object.entries(profile.normalization).every(([key, value]) => metadata.normalization?.[key] === value) ||
      !Object.entries(CONFIG).every(([key, value]) => metadata.gameConfig?.[key] === value) ||
      metadata.sampleIntervalMs !== CONFIG.decisionIntervalMs || metadata.minimumJumpIntervalMs !== CONFIG.minimumJumpIntervalMs) {
    throw new Error('The model game or normalization settings do not match this game.');
  }
  if (!/^[a-f0-9]{64}$/.test(metadata.cssSha256) || !/^[a-f0-9]{64}$/.test(metadata.modelSha256) ||
      !Number.isFinite(metadata.threshold) || metadata.threshold <= 0 || metadata.threshold >= 1) {
    throw new Error('Invalid model metadata.');
  }
}

export class AiController {
  constructor(sink, modelBase, profileId = 'speed-516') {
    this.profile = modelProfile(profileId);
    this.sink = sink;
    this.scopeId = `nn-${++controllerSequence}`;
    this.sink.setAttribute?.('data-neural-sink', this.scopeId);
    this.modelBase = modelBase ?? this.profile.base;
    this.savedFamily = 'recordings';
    this.isTrial = false;
    this.ready = false;
    this.threshold = 0.5;
    this.error = '';
    this.reset();
  }

  reset() {
    this.lastPrediction = null;
    this.decision = 'READY';
    this.decisionCount = 0;
    this.jumpCount = 0;
    this.waitCount = 0;
  }

  setSavedFamily(family) {
    if (!['recordings', 'rl'].includes(family) || family === 'rl' && !this.profile.rlBase) {
      throw new Error('This model source is not supported by this lab.');
    }
    this.savedFamily = family;
  }

  async loadSaved(difficulty = 'normal', { force = false } = {}) {
    try {
      if (!['normal', 'hard'].includes(difficulty)) throw new Error('Unknown game difficulty.');
      let base = this.savedFamily === 'rl' ? this.profile.rlBase : this.profile.base;
      const hardBase = this.savedFamily === 'rl' ? this.profile.rlHardBase : this.profile.hardBase;
      if (difficulty === 'hard') {
        const response = await fetch(`${hardBase}model.meta.json`, { cache: 'no-store' });
        if (response.ok) base = hardBase;
        else if (response.status !== 404) throw new Error('The saved hard model could not be checked.');
      }
      if (!force && this.ready && !this.isTrial && this.modelBase === base) return true;
      if (!await this.initialize(base)) return false;
      this.isTrial = false;
      return true;
    } catch (error) { this.error = error.message; return false; }
  }

  async initialize(modelBase = this.modelBase) {
    let style;
    try {
      const metadataResponse = await fetch(`${modelBase}model.meta.json`, { cache: 'no-store' });
      if (!metadataResponse.ok) throw new Error('No trained, CSS-verified model has been loaded.');
      const metadata = await metadataResponse.json();
      validateMetadata(metadata, this.profile.id);
      const cssResponse = await fetch(`${modelBase}model.css`, { cache: 'no-store' });
      if (!cssResponse.ok) throw new Error('The model CSS file could not be loaded.');
      const cssText = await cssResponse.text();
      const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(cssText));
      const digest = [...new Uint8Array(hash)].map((value) => value.toString(16).padStart(2, '0')).join('');
      if (digest !== metadata.cssSha256) throw new Error('The CSS file changed; it needs a new parity check.');
      style = document.createElement('style');
      style.textContent = scopeModelCss(cssText, this.scopeId);
      document.head.append(style);
      const probe = cssNumberReader(this.sink)('--nn-probe');
      if (Math.abs(probe - 0.7310585786300049) >= 1e-4) throw new Error('This browser does not support the required CSS calculations.');
      this.style?.remove();
      this.style = style;
      this.modelBase = modelBase;
      this.metadata = metadata;
      this.threshold = metadata.threshold;
      this.ready = true;
      this.error = '';
    } catch (error) {
      style?.remove();
      if (this.ready && this.style) this.error = error.message;
      else this.fail(error);
      return false;
    }
    return this.ready;
  }

  setThreshold(value) {
    if (!Number.isFinite(value) || value <= 0 || value >= 1) throw new Error('The threshold must be between 0 and 1.');
    this.threshold = value;
  }

  evaluate(state) {
    if (!this.ready) throw new Error(this.error || 'The AI model is not ready.');
    const inputs = normalizeState(state, this.profile.normalization);
    CSS_INPUT_PROPERTIES.slice(0, inputs.length).forEach((property, index) => this.sink.style.setProperty(property, String(inputs[index])));
    const readNumber = cssNumberReader(this.sink);
    const probability = readNumber('--jump-probability');
    const hidden = Array.from({ length: this.profile.architecture[1] }, (_, index) => readNumber(`--h${index + 1}`));
    const logit = readNumber('--output-z');
    if (probability < 0 || probability > 1 || hidden.some((value) => value < 0)) throw new Error('Invalid CSS output range.');
    this.lastPrediction = { inputs, probability, hidden, logit };
    this.decisionCount += 1;
    return this.lastPrediction;
  }

  recordDecision(action, requested) {
    if (action === 1) this.jumpCount += 1;
    else this.waitCount += 1;
    this.decision = action === 1 ? 'JUMP' : requested ? 'COOLDOWN' : 'WAIT';
  }

  fail(error) {
    this.ready = false;
    this.error = error.message;
    this.decision = 'ERROR';
  }
}
