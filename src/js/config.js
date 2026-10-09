export const CONFIG = Object.freeze({
  width: 480,
  height: 640,
  birdX: 124,
  birdRadius: 14,
  initialBirdY: 300,
  gravity: 1500,
  jumpVelocity: -440,
  pipeSpeed: 180,
  pipeWidth: 76,
  pipeGap: 190,
  pipeSpacing: 280,
  firstPipeX: 600,
  gapMargin: 64,
  fixedStepSeconds: 1 / 120,
  maxFrameSeconds: 0.1,
  decisionIntervalMs: 50,
  minimumJumpIntervalMs: 100,
  maxTrainingSamples: 100000,
});

// Difficulty is separate from the versioned normal-physics/model contract.
export const HARD_DIFFICULTY_RULES = Object.freeze({ speedGrowthPerPoint: 0.01, maxSpeedMultiplier: 1.5, speedResponseSeconds: 2 });
export const DIFFICULTIES = Object.freeze({
  normal: Object.freeze({ label: 'Normal', speedGrowthPerPoint: 0, maxSpeedMultiplier: 1, speedResponseSeconds: 2 }),
  hard: Object.freeze({ label: 'Hard', ...HARD_DIFFICULTY_RULES }),
});

export const INPUT_NAMES = Object.freeze(['birdY', 'velocityY', 'pipeDistance', 'gapY']);
export const NORMALIZATION = Object.freeze({
  version: 1,
  height: CONFIG.height,
  velocityScale: 1200,
  maxPipeDistance: 600,
});
