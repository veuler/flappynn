import { CONFIG, DIFFICULTIES, HARD_DIFFICULTY_RULES } from './config.js';
import { normalizeState } from './normalization.js';
import { modelProfile } from './model-profiles.js';
import { validateRecordingDataset } from './recording-dataset.js';

export class TrainingRecorder {
  constructor({ maxSamples = CONFIG.maxTrainingSamples, profileId = 'speed-516' } = {}) {
    this.profile = modelProfile(profileId);
    this.maxSamples = maxSamples;
    this.clear();
  }

  get full() { return this.samples.length >= this.maxSamples; }

  importDataset(data) {
    if (this.activeSession) throw new Error('End your recording flight before importing.');
    validateRecordingDataset(data,this.profile.id);
    if (data.samples.length > this.maxSamples) throw new Error('This file exceeds the recording limit.');
    this.samples=data.samples.map(row=>[...row]);
    this.sessions=data.sessions.map(flight=>({...flight,difficulty:flight.difficulty || 'normal'}));
    this.jumpCount=data.counts.jump;
    this.dataKind=data.dataKind || 'human-gameplay-unverified';
  }

  clear() {
    if (this.activeSession) throw new Error('Recordings cannot be cleared during an active recording flight.');
    this.samples = [];
    this.sessions = [];
    this.activeSession = null;
    this.jumpCount = 0;
    this.dataKind='human-gameplay-unverified';
  }

  beginSession(state) {
    if (this.activeSession) throw new Error('End the active session first.');
    const difficulty = state.difficulty ?? 'normal';
    if (!Object.hasOwn(DIFFICULTIES, difficulty)) throw new Error('Unknown recording difficulty.');
    if (this.full) return;
    this.activeSession = {
      id: this.sessions.reduce((maximum,flight)=>Math.max(maximum,flight.id),0) + 1,
      seed: state.seed,
      difficulty,
      createdAt: new Date().toISOString(),
      startIndex: this.samples.length,
      sampleCount: 0,
      score: 0,
      elapsedSeconds: 0,
      endReason: null,
    };
    this.sessions.push(this.activeSession);
  }

  record(state, action) {
    if (!this.activeSession || this.full) return false;
    if (action !== 0 && action !== 1) throw new Error('An action must be either 0 or 1.');
    this.samples.push([...normalizeState(state, this.profile.normalization), action]);
    this.jumpCount += action;
    this.activeSession.sampleCount += 1;
    this.activeSession.elapsedSeconds = state.elapsedSeconds;
    this.activeSession.score = state.score;
    return true;
  }

  endSession(state, reason) {
    if (!this.activeSession) return;
    Object.assign(this.activeSession, {
      elapsedSeconds: state.elapsedSeconds,
      score: state.score,
      endReason: reason,
    });
    this.activeSession = null;
  }

  exportDataset(currentState) {
    const sessions = this.sessions.map((session) => ({ ...session }));
    if (this.activeSession && currentState) {
      Object.assign(sessions.at(-1), {
        elapsedSeconds: currentState.elapsedSeconds,
        score: currentState.score,
        endReason: 'in-progress',
      });
    }
    return {
      version: this.profile.datasetVersion,
      dataKind:this.dataKind,
      modelProfile: this.profile.id,
      difficultyRules: { hard: { ...HARD_DIFFICULTY_RULES } },
      exportedAt: new Date().toISOString(),
      sampleIntervalMs: CONFIG.decisionIntervalMs,
      minimumJumpIntervalMs: CONFIG.minimumJumpIntervalMs,
      clock: 'simulation',
      sampleTiming: 'before-action',
      inputNames: [...this.profile.inputNames],
      normalization: { ...this.profile.normalization },
      gameConfig: { ...CONFIG },
      counts: { total: this.samples.length, jump: this.jumpCount, wait: this.samples.length - this.jumpCount },
      sessions,
      samples: this.samples.map((sample) => [...sample]),
    };
  }
}
