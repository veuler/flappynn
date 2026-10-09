import test from 'node:test';
import assert from 'node:assert/strict';
import { CONFIG, INPUT_NAMES, NORMALIZATION, HARD_DIFFICULTY_RULES } from '../src/js/config.js';
import { AiController, validateMetadata } from '../src/js/ai-controller.js';
import { Game } from '../src/js/game.js';
import { TrainingRecorder } from '../src/js/training-recorder.js';
import { SessionController } from '../src/js/session-controller.js';
import { cssNumberReader } from '../src/js/css-numbers.js';

test('CSS reader retains numeric precision lost by string serialization', () => {
  const actual = -73.41345352003815;
  const expected = -73.41339844535528;
  const read = cssNumberReader({ computedStyleMap: () => new Map([['--output-z', { unit: 'number', value: actual }]]) });
  assert.equal(read('--output-z'), actual);
  assert.ok(Math.abs(read('--output-z') - expected) < 1e-4);
  assert.ok(Math.abs(Number('-73.4135') - expected) >= 1e-4);
});

test('CSS reader falls back strictly when numeric Typed OM is unavailable', (context) => {
  let token = ' 5.3e-6 ';
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'getComputedStyle');
  Object.defineProperty(globalThis, 'getComputedStyle', { configurable: true, value: () => ({ getPropertyValue: () => token }) });
  context.after(() => previous ? Object.defineProperty(globalThis, 'getComputedStyle', previous) : delete globalThis.getComputedStyle);
  for (const sink of [{}, { computedStyleMap: () => new Map([['--h1', { toString: () => 'unparsed' }]]) }]) {
    const read = cssNumberReader(sink);
    assert.equal(read('--h1'), 5.3e-6);
    for (const invalid of ['', 'calc(1 + 1)', '3px', 'Infinity', 'NaN']) {
      token = invalid;
      assert.throws(() => read('--h1'), /Invalid CSS output/);
    }
    token = ' 5.3e-6 ';
  }
});

test('CSS reader rejects nonfinite typed numbers and reads fresh input states', () => {
  let value = 1;
  const sink = { computedStyleMap: () => new Map([['--h1', { unit: 'number', value }]]) };
  assert.equal(cssNumberReader(sink)('--h1'), 1);
  value = 2;
  assert.equal(cssNumberReader(sink)('--h1'), 2);
  for (value of [Infinity, NaN]) assert.throws(() => cssNumberReader(sink)('--h1'), /Invalid CSS output/);
});

test('failed candidate loading preserves the working model, its source and readiness', async (context) => {
  const ai = new AiController({});
  let removed = false;
  ai.style = { remove() { removed = true; } };
  ai.ready = true;
  ai.metadata = { modelSha256: 'working model' };
  context.mock.method(globalThis, 'fetch', async () => new Response('{}', { status: 404 }));
  assert.equal(await ai.initialize('./learning/invalid/'), false);
  assert.equal(ai.ready, true);
  assert.equal(ai.modelBase, './models/speed-516/');
  assert.equal(ai.metadata.modelSha256, 'working model');
  assert.equal(removed, false);
  assert.ok(ai.error);
});

test('saved model selection falls back to normal, chooses an existing hard slot and restores normal explicitly', async (context) => {
  const ai = new AiController({});
  ai.ready = true;
  let hardExists = false;
  const sources = [];
  context.mock.method(globalThis, 'fetch', async () => new Response('{}', { status: hardExists ? 200 : 404 }));
  context.mock.method(ai, 'initialize', async function(base) { sources.push(base); this.modelBase = base; this.ready = true; return true; });
  assert.equal(await ai.loadSaved('hard'), true);
  assert.equal(ai.modelBase, './models/speed-516/');
  assert.deepEqual(sources, []);
  hardExists = true;
  assert.equal(await ai.loadSaved('hard'), true);
  assert.equal(ai.modelBase, './models/speed-516/hard/');
  assert.equal(ai.isTrial, false);
  ai.isTrial = true;
  ai.modelBase = './learning/candidate/';
  assert.equal(await ai.loadSaved('normal', { force: true }), true);
  assert.equal(ai.modelBase, './models/speed-516/');
  assert.equal(ai.isTrial, false);
  assert.equal(await ai.loadSaved('unknown'), false);
  assert.equal(ai.modelBase, './models/speed-516/');
});

function setup(probabilities = [0.9]) {
  const game = new Game();
  const recorder = new TrainingRecorder();
  const ai = {
    ready: true, threshold: 0.5, calls: [], decisions: [],
    reset() { this.calls = []; this.decisions = []; },
    evaluate(state) { this.calls.push(state); return { probability: probabilities[(this.calls.length - 1) % probabilities.length] }; },
    recordDecision(action, requested) { this.decisions.push({ action, requested }); },
    fail(error) { this.ready = false; this.error = error.message; },
  };
  return { game, recorder, ai, session: new SessionController(game, recorder, ai) };
}

test('AI consumes supplied CSS decisions at 20 Hz, applies cooldown and records no human data', () => {
  const { game, recorder, ai, session } = setup([0.9, 0.9, 0.1]);
  session.start(1, 'ai');
  for (let i = 0; i < 18; i++) session.step();
  assert.equal(ai.calls.length, 3);
  assert.deepEqual(ai.decisions.map((d) => d.action), [0, 1, 0]);
  assert.ok(Math.abs(ai.calls[0].elapsedSeconds - 0.05) < 1e-8);
  assert.ok(ai.calls[1].birdVelocityY > CONFIG.jumpVelocity);
  assert.equal(recorder.samples.length, 0);
  assert.equal(recorder.sessions.length, 0);
  assert.equal(game.status, 'running');
});

test('manual jump events do not control the bird during an AI run', () => {
  const { game, session } = setup([0]);
  session.start(1, 'ai');
  session.step();
  const velocity = game.bird.velocityY;
  session.requestJump();
  assert.equal(game.bird.velocityY, velocity);
});

test('AI totals count applied actions including cooldown waits and reset with a new run', () => {
  const game = new Game();
  const ai = new AiController({});
  ai.ready = true;
  const probabilities = [.9, .9, .1];
  ai.evaluate = () => ({ probability: probabilities[ai.decisionCount++ % probabilities.length] });
  const session = new SessionController(game, new TrainingRecorder(), ai);
  session.start(1, 'ai');
  for (let tick = 0; tick < 18; tick++) session.step();
  assert.equal(ai.jumpCount, 1);
  assert.equal(ai.waitCount, 2);
  assert.equal(ai.jumpCount + ai.waitCount, ai.decisionCount);
  game.pause();
  session.step();
  assert.equal(ai.decisionCount, 3);
  session.start(2, 'ai');
  assert.equal(ai.jumpCount, 0);
  assert.equal(ai.waitCount, 0);
  assert.equal(ai.decisionCount, 0);
});

test('unavailable AI is rejected before ending or clearing an active training session', () => {
  const { game, recorder, session, ai } = setup();
  session.start(1, 'training');
  const oldState = game.getState();
  ai.ready = false;
  assert.throws(() => session.start(2, 'ai'));
  assert.deepEqual(game.getState(), oldState);
  assert.ok(recorder.activeSession);
});

test('an inference read failure pauses the run and never substitutes a decision', () => {
  const { game, session, ai } = setup();
  ai.evaluate = () => { throw new Error('invalid computed CSS'); };
  session.start(1, 'ai');
  for (let i = 0; i < 6; i++) session.step();
  assert.equal(game.status, 'paused');
  assert.equal(ai.ready, false);
  assert.equal(ai.error, 'invalid computed CSS');
  assert.equal(ai.decisions.length, 0);
});

test('AI metadata requires parity, current game settings and complete digests', () => {
  const metadata = {
    version: 2, modelProfile: 'speed-516', architecture: [5, 16, 1], parityVerified: true, parity: { caseCount: 520, tolerance: 1e-4, maximumProbabilityError: 1e-5 },
    inputNames: [...INPUT_NAMES, 'pipeSpeed'], normalization: { ...NORMALIZATION, version: 2, maxPipeSpeed: 270 }, gameConfig: { ...CONFIG },
    sampleIntervalMs: 50, minimumJumpIntervalMs: 100, cssSha256: 'a'.repeat(64), modelSha256: 'b'.repeat(64), threshold: 0.5,
  };
  assert.doesNotThrow(() => validateMetadata(metadata));
  assert.doesNotThrow(() => validateMetadata({ ...metadata, trainingDifficulty: 'hard', difficultyConfig: { ...HARD_DIFFICULTY_RULES } }));
  assert.throws(() => validateMetadata({ ...metadata, trainingDifficulty: 'hard', difficultyConfig: { ...HARD_DIFFICULTY_RULES, maxSpeedMultiplier: 2 } }));
  for (const mutate of [m => m.parityVerified = false, m => m.normalization.velocityScale = 1,
    m => m.gameConfig.gravity = 1, m => m.cssSha256 = 'missing', m => m.threshold = NaN,
    m => delete m.parity.caseCount, m => m.parity.maximumProbabilityError = -1, m => m.trainingDifficulty = 'unknown']) {
    const bad = structuredClone(metadata);
    mutate(bad);
    assert.throws(() => validateMetadata(bad));
  }
});
