import { CONFIG, DIFFICULTIES } from './config.js';
import { createPipe, createRandom, nextPipe } from './pipes.js';
import { circleHitsRectangle, updateBird } from './physics.js';

export class Game {
  constructor({ seed = 12345, config = CONFIG, difficulty = 'normal' } = {}) {
    this.config = config;
    this.reset(seed, difficulty);
  }

  reset(seed = this.seed, difficulty = this.difficulty) {
    if (!Object.hasOwn(DIFFICULTIES, difficulty)) throw new Error('Unknown difficulty.');
    this.difficulty = difficulty;
    this.pipeSpeed = this.config.pipeSpeed;
    this.seed = seed >>> 0;
    this.random = createRandom(this.seed);
    this.status = 'ready';
    this.bird = { y: this.config.initialBirdY, velocityY: 0 };
    this.score = 0;
    this.elapsedSeconds = 0;
    this.pipes = Array.from({ length: 3 }, (_, index) =>
      createPipe(this.config.firstPipeX + index * this.config.pipeSpacing, this.random, this.config));
  }

  start() {
    if (this.status !== 'ready') return;
    this.status = 'running';
    this.jump();
  }

  jump() {
    if (this.status === 'running') this.bird.velocityY = this.config.jumpVelocity;
  }

  pause() {
    if (this.status === 'running') this.status = 'paused';
  }

  resume() {
    if (this.status === 'paused') this.status = 'running';
  }

  finish() {
    if (['running', 'paused'].includes(this.status)) this.status = 'gameover';
  }

  step(deltaSeconds = this.config.fixedStepSeconds) {
    if (this.status !== 'running') return;
    const config = this.config;
    this.elapsedSeconds += deltaSeconds;
    updateBird(this.bird, deltaSeconds, config);
    const difficulty = DIFFICULTIES[this.difficulty];
    const multiplier = Math.min(1 + this.score * difficulty.speedGrowthPerPoint, difficulty.maxSpeedMultiplier);
    const targetSpeed = config.pipeSpeed * multiplier;
    // A two-second smoothing time avoids a sudden acceleration at each point.
    this.pipeSpeed += (targetSpeed - this.pipeSpeed) * Math.min(deltaSeconds / difficulty.speedResponseSeconds, 1);
    for (const pipe of this.pipes) pipe.x -= this.pipeSpeed * deltaSeconds;

    while (this.pipes[0].x + config.pipeWidth < 0) this.pipes.shift();
    while (this.pipes.at(-1).x < config.width + config.pipeSpacing) {
      this.pipes.push(createPipe(this.pipes.at(-1).x + config.pipeSpacing, this.random, config));
    }

    const boundaryHit = this.bird.y - config.birdRadius <= 0 ||
      this.bird.y + config.birdRadius >= config.height;
    const pipeHit = this.pipes.some((pipe) => {
      const gapTop = pipe.gapCenterY - config.pipeGap / 2;
      const gapBottom = pipe.gapCenterY + config.pipeGap / 2;
      return circleHitsRectangle(config.birdX, this.bird.y, config.birdRadius,
        { x: pipe.x, y: 0, width: config.pipeWidth, height: gapTop }) ||
        circleHitsRectangle(config.birdX, this.bird.y, config.birdRadius,
          { x: pipe.x, y: gapBottom, width: config.pipeWidth, height: config.height - gapBottom });
    });
    if (boundaryHit || pipeHit) {
      this.status = 'gameover';
      return;
    }

    for (const pipe of this.pipes) {
      if (!pipe.passed && pipe.x + config.pipeWidth < config.birdX - config.birdRadius) {
        pipe.passed = true;
        this.score += 1;
      }
    }
  }

  getState() {
    const pipe = nextPipe(this.pipes, this.config.birdX, this.config);
    const followingPipe = pipe ? this.pipes[this.pipes.indexOf(pipe) + 1] : null;
    return {
      birdY: this.bird.y,
      birdVelocityY: this.bird.velocityY,
      nextPipeDistanceX: pipe ? Math.max(0, pipe.x - this.config.birdX) : null,
      nextPipeGapCenterY: pipe?.gapCenterY ?? null,
      followingPipeGapCenterY: followingPipe?.gapCenterY ?? null,
      score: this.score,
      elapsedSeconds: this.elapsedSeconds,
      status: this.status,
      seed: this.seed,
      difficulty: this.difficulty,
      pipeSpeed: this.pipeSpeed,
    };
  }
}
