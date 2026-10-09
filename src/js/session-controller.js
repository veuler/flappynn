import { CONFIG, DIFFICULTIES } from './config.js';

export class SessionController {
  constructor(game, recorder, ai = null) {
    this.game = game;
    this.recorder = recorder;
    this.ai = ai;
    this.mode = 'human';
    this.ticksPerDecision = Math.round(CONFIG.decisionIntervalMs / 1000 / game.config.fixedStepSeconds);
    if (Math.abs(this.ticksPerDecision * game.config.fixedStepSeconds * 1000 - CONFIG.decisionIntervalMs) > 1e-8) {
      throw new Error('The decision interval must be an exact multiple of the physics step.');
    }
  }

  start(seed, mode, difficulty = 'normal') {
    if (!['human', 'training', 'ai'].includes(mode)) throw new Error('Unknown game mode.');
    if (!Object.hasOwn(DIFFICULTIES, difficulty)) throw new Error('Unknown difficulty.');
    if (mode === 'ai' && !this.ai?.ready) throw new Error('The verified CSS model is not ready.');
    this.recorder.endSession(this.game.getState(), 'restart');
    this.game.reset(seed, difficulty);
    this.mode = mode;
    this.pendingJump = false;
    this.tick = 0;
    this.lastJumpTick = 0;
    if (mode === 'ai') this.ai.reset();
    if (mode === 'training') {
      this.recorder.beginSession(this.game.getState());
      // The initial impulse is also a jump, recorded before Game.start changes velocity.
      this.recorder.record(this.game.getState(), 1);
    }
    this.game.start();
  }

  requestJump() {
    if (this.game.status !== 'running') return;
    if (this.mode === 'human') this.game.jump();
    else if (this.mode === 'training') this.pendingJump = true;
  }

  finish() {
    if (!['running', 'paused'].includes(this.game.status)) return;
    this.game.finish();
    this.pendingJump = false;
    this.recorder.endSession(this.game.getState(), 'stopped');
  }

  step() {
    if (this.game.status !== 'running') return;
    this.game.step();
    this.tick += 1;
    if (this.game.status === 'gameover') {
      this.recorder.endSession(this.game.getState(), 'collision');
      return;
    }
    if (this.mode === 'human' || this.tick % this.ticksPerDecision !== 0) return;
    const cooledDown = (this.tick - this.lastJumpTick) * this.game.config.fixedStepSeconds * 1000 >=
      CONFIG.minimumJumpIntervalMs - 1e-8;
    let requested = this.pendingJump;
    if (this.mode === 'ai') {
      try {
        requested = this.ai.evaluate(this.game.getState()).probability >= this.ai.threshold;
      } catch (error) {
        this.ai.fail(error);
        this.game.pause();
        return;
      }
    }
    const action = requested && cooledDown ? 1 : 0;
    if (this.mode === 'training') this.recorder.record(this.game.getState(), action);
    else this.ai.recordDecision(action, requested);
    this.pendingJump = false;
    if (action === 1) {
      this.game.jump();
      this.lastJumpTick = this.tick;
    }
  }
}
