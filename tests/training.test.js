import test from 'node:test';
import assert from 'node:assert/strict';
import { CONFIG, HARD_DIFFICULTY_RULES, NORMALIZATION } from '../src/js/config.js';
import { Game } from '../src/js/game.js';
import { normalizeState } from '../src/js/normalization.js';
import { TrainingRecorder } from '../src/js/training-recorder.js';
import { SessionController } from '../src/js/session-controller.js';

function setup(options) {
  const game = new Game();
  const recorder = new TrainingRecorder(options);
  const session = new SessionController(game, recorder);
  return { game, recorder, session };
}

function steps(session, count) {
  for (let i = 0; i < count; i++) session.step();
}

test('normalization preserves input order, clips limits and rejects missing/nonfinite state', () => {
  const state = { birdY: 320, birdVelocityY: -600, nextPipeDistanceX: 300, nextPipeGapCenterY: 160 };
  assert.deepEqual(normalizeState(state), [0.5, -0.5, 0.5, 0.25]);
  assert.deepEqual(normalizeState({ birdY: 700, birdVelocityY: -3000, nextPipeDistanceX: -5, nextPipeGapCenterY: 800 }), [1, -1, 0, 1]);
  for (const value of [NaN, Infinity, null, undefined]) {
    assert.throws(() => normalizeState({ ...state, birdY: value }));
  }
});

test('training records exactly 20 decision intervals per simulated second, before jumps', () => {
  const { game, recorder, session } = setup();
  session.start(12345, 'training');
  assert.equal(recorder.samples.length, 1);
  assert.equal(recorder.samples[0][1], 0);
  assert.equal(recorder.samples[0][5], 1);
  steps(session, 11);
  assert.equal(recorder.samples.length, 2);
  session.requestJump();
  const oldVelocity = game.bird.velocityY;
  session.step();
  assert.equal(recorder.samples.length, 3);
  assert.equal(recorder.samples[2][5], 1);
  assert.equal(recorder.samples[2][1], (oldVelocity + CONFIG.gravity * CONFIG.fixedStepSeconds) / NORMALIZATION.velocityScale);
  assert.equal(game.bird.velocityY, CONFIG.jumpVelocity);
  for (let i = 12; i < 120; i++) {
    game.bird.y = 320; game.bird.velocityY = 0;
    session.step();
  }
  assert.equal(recorder.samples.length, 21); // t=0 plus 20 boundaries, not 120 physics frames.
  assert.equal(recorder.jumpCount, 2);
});

test('repeated requests merge into one accepted event and cooldown rejection becomes WAIT', () => {
  const { game, recorder, session } = setup();
  session.start(1, 'training');
  session.requestJump(); session.requestJump();
  steps(session, 6);
  assert.equal(recorder.samples.at(-1)[5], 0);
  session.requestJump(); session.requestJump();
  steps(session, 6);
  assert.equal(recorder.samples.at(-1)[5], 1);
  assert.equal(game.bird.velocityY, CONFIG.jumpVelocity);
  steps(session, 12);
  assert.equal(recorder.samples.at(-1)[5], 0);
  assert.equal(recorder.jumpCount, 2);
});

test('paused, ready, gameover and human gameplay cannot collect training samples', () => {
  const { game, recorder, session } = setup();
  steps(session, 30);
  assert.equal(recorder.samples.length, 0);
  session.start(1, 'human');
  steps(session, 20); session.requestJump();
  assert.equal(game.bird.velocityY, CONFIG.jumpVelocity);
  assert.equal(recorder.samples.length, 0);
  session.start(1, 'training');
  steps(session, 12); game.pause();
  const count = recorder.samples.length;
  const time = game.elapsedSeconds;
  steps(session, 1000); session.requestJump();
  assert.equal(recorder.samples.length, count);
  assert.equal(game.elapsedSeconds, time);
  game.resume(); steps(session, 500);
  assert.equal(game.status, 'gameover');
  const deathCount = recorder.samples.length;
  steps(session, 100);
  assert.equal(recorder.samples.length, deathCount);
  assert.equal(recorder.sessions[0].endReason, 'collision');
  assert.equal(recorder.activeSession, null);
});

test('sessions append contiguous sample ranges with independent seeds and simulation clocks', () => {
  const { recorder, session } = setup();
  session.start(11, 'training'); steps(session, 12);
  session.start(22, 'training'); steps(session, 6);
  const data = recorder.exportDataset(session.game.getState());
  assert.deepEqual(data.sessions.map((s) => [s.seed, s.startIndex, s.sampleCount, s.endReason]),
    [[11, 0, 3, 'restart'], [22, 3, 2, 'in-progress']]);
  assert.equal(data.samples.length, 5);
  assert.equal(data.sessions[1].elapsedSeconds, session.game.elapsedSeconds);
  assert.deepEqual(data.normalization, { ...NORMALIZATION, version: 2, maxPipeSpeed: 270 });
  assert.equal(data.sampleTiming, 'before-action');
  assert.equal(data.counts.jump + data.counts.wait, data.counts.total);
  assert.deepEqual(JSON.parse(JSON.stringify(data)), data);
  assert.ok(data.samples.every((s) => s.length === 6 && s.every(Number.isFinite)));
});

test('export is an independent snapshot and empty/full data remains well formed', () => {
  const { recorder, session } = setup({ maxSamples: 2 });
  assert.equal(recorder.exportDataset().samples.length, 0);
  session.start(1, 'training'); steps(session, 12);
  assert.equal(recorder.samples.length, 2);
  assert.equal(recorder.full, true);
  const exported = recorder.exportDataset(session.game.getState());
  exported.samples[0][0] = -99;
  exported.sessions[0].sampleCount = -1;
  assert.ok(recorder.samples[0][0] >= 0);
  assert.equal(recorder.sessions[0].sampleCount, 2);
  steps(session, 500);
  session.start(2, 'training');
  assert.equal(recorder.sessions.length, 1);
  assert.equal(recorder.samples.length, 2);
});

test('clearing requires an ended session and resets all counters', () => {
  const { recorder, session } = setup();
  session.start(1, 'training');
  assert.throws(() => recorder.clear());
  steps(session, 500);
  recorder.clear();
  assert.equal(recorder.samples.length, 0);
  assert.equal(recorder.sessions.length, 0);
  assert.equal(recorder.jumpCount, 0);
  assert.equal(recorder.full, false);
});

test('finishing a paused session allows export and clearing without losing its score', () => {
  const { game, recorder, session } = setup();
  session.start(5, 'training'); steps(session, 12);
  game.score = 3; game.pause();
  session.finish();
  assert.equal(game.status, 'gameover');
  assert.equal(game.score, 3);
  assert.equal(recorder.sessions[0].endReason, 'stopped');
  assert.equal(recorder.exportDataset().sessions[0].score, 3);
  const count = recorder.samples.length;
  steps(session, 100);
  assert.equal(recorder.samples.length, count);
  recorder.clear();
  assert.equal(recorder.samples.length, 0);
});

test('three minutes of simulation maintain the fixed sampling cadence without drift', () => {
  const { game, recorder, session } = setup();
  session.start(12345, 'training');
  // A stable flight fixture tests collection duration independently of player skill.
  for (let i = 0; i < 180 / CONFIG.fixedStepSeconds; i++) {
    for (const pipe of game.pipes) pipe.gapCenterY = 320;
    game.bird.y = 320; game.bird.velocityY = 0;
    session.step();
  }
  session.finish();
  const data = recorder.exportDataset();
  assert.equal(data.samples.length, 3601);
  assert.equal(data.sessions[0].sampleCount, 3601);
  assert.ok(Math.abs(data.sessions[0].elapsedSeconds - 180) < 1e-8);
  assert.ok(data.samples.every((row) => row.length === 6 && row.every(Number.isFinite)));
});

test('unknown difficulty is rejected before changing an active recording; normal and hard flights retain distinct metadata', () => {
  const { game, recorder, session } = setup();
  session.start(42, 'training');
  steps(session, 12);
  const { exportedAt: beforeTime, ...before } = recorder.exportDataset(game.getState());
  const state = game.getState();
  assert.throws(() => session.start(99, 'human', 'unknown'), /difficulty/);
  assert.deepEqual(game.getState(), state);
  const { exportedAt: afterTime, ...after } = recorder.exportDataset(game.getState());
  assert.deepEqual(after, before);
  const count = recorder.samples.length;
  session.start(99, 'training', 'hard');
  steps(session, 12);
  assert.equal(game.difficulty, 'hard');
  assert.equal(recorder.samples.length, count + 3);
  session.start(100, 'training');
  assert.equal(game.difficulty, 'normal');
  assert.equal(game.pipeSpeed, 180);
  const data = recorder.exportDataset(game.getState());
  assert.equal(data.version, 3);
  assert.deepEqual(data.difficultyRules.hard, HARD_DIFFICULTY_RULES);
  assert.deepEqual(data.sessions.map((flight) => [flight.difficulty, flight.startIndex, flight.sampleCount]),
    [['normal', 0, 3], ['hard', 3, 3], ['normal', 6, 1]]);
  data.difficultyRules.hard.maxSpeedMultiplier = 99;
  assert.equal(recorder.exportDataset().difficultyRules.hard.maxSpeedMultiplier, 1.5);
});

test('hard recording follows accelerating physics and the same pre-action 20 Hz cadence', () => {
  const { game, recorder, session } = setup();
  session.start(12345, 'training', 'hard');
  for (let i = 0; i < 30 / CONFIG.fixedStepSeconds; i++) {
    for (const pipe of game.pipes) pipe.gapCenterY = 320;
    game.bird.y = 320; game.bird.velocityY = 0;
    if (i % 12 === 11) session.requestJump();
    session.step();
  }
  session.finish();
  const data = recorder.exportDataset();
  assert.ok(game.score > 0);
  assert.ok(game.pipeSpeed > CONFIG.pipeSpeed);
  assert.equal(data.sessions[0].difficulty, 'hard');
  assert.equal(data.samples.length, 601);
  assert.equal(data.counts.jump, 301);
  assert.equal(data.samples[2][1], CONFIG.gravity * CONFIG.fixedStepSeconds / NORMALIZATION.velocityScale);
});
