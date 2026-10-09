import test from 'node:test';
import assert from 'node:assert/strict';
import { Game } from '../src/js/game.js';
import { CONFIG } from '../src/js/config.js';
import { createPipe, createRandom, nextPipe } from '../src/js/pipes.js';
import { circleHitsRectangle } from '../src/js/physics.js';

test('same seed and inputs reproduce a run including after reset', () => {
  const game = new Game({ seed: 12345 });
  const replay = () => {
    game.start();
    for (let i = 0; i < 400; i++) {
      if (i % 45 === 0) game.jump();
      game.step();
    }
    return { state: game.getState(), pipes: structuredClone(game.pipes) };
  };
  const first = replay();
  game.reset(12345);
  assert.deepEqual(replay(), first);
});

test('gap placement remains playable and seeds change the sequence', () => {
  const random = createRandom(42);
  for (let i = 0; i < 1000; i++) {
    const pipe = createPipe(600, random, CONFIG);
    assert.ok(pipe.gapCenterY - CONFIG.pipeGap / 2 >= CONFIG.gapMargin);
    assert.ok(pipe.gapCenterY + CONFIG.pipeGap / 2 <= CONFIG.height - CONFIG.gapMargin);
  }
  assert.notDeepEqual(new Game({ seed: 1 }).pipes, new Game({ seed: 2 }).pipes);
});

test('falling into the floor ends the game and freezes simulation', () => {
  const game = new Game();
  game.start();
  for (let i = 0; i < 400; i++) game.step();
  assert.equal(game.status, 'gameover');
  const before = game.getState();
  game.step(); game.jump();
  assert.deepEqual(game.getState(), before);
});

test('pause preserves state and resume advances without resetting the run', () => {
  const game = new Game();
  game.start(); game.step(); game.pause();
  const before = game.getState();
  game.step(); game.jump();
  assert.deepEqual(game.getState(), before);
  game.resume(); game.step();
  assert.ok(game.elapsedSeconds > before.elapsedSeconds);
});

test('next pipe is retained during overlap and advances after safe passage', () => {
  const game = new Game();
  game.pipes[0].x = CONFIG.birdX - CONFIG.pipeWidth;
  assert.equal(nextPipe(game.pipes, CONFIG.birdX, CONFIG), game.pipes[0]);
  assert.equal(game.getState().nextPipeDistanceX, 0);
  game.pipes[0].x = CONFIG.birdX - CONFIG.birdRadius - CONFIG.pipeWidth - 1;
  assert.equal(nextPipe(game.pipes, CONFIG.birdX, CONFIG), game.pipes[1]);
});

test('a safely cleared pipe awards exactly one point', () => {
  const game = new Game();
  game.start();
  game.pipes[0].x = CONFIG.birdX - CONFIG.birdRadius - CONFIG.pipeWidth - 1;
  game.pipes[0].gapCenterY = game.bird.y;
  game.step();
  assert.equal(game.score, 1);
  game.step();
  assert.equal(game.score, 1);
});

test('pipe collision is detected, including circle contact at a corner', () => {
  assert.equal(circleHitsRectangle(0, 0, 5, { x: 3, y: 4, width: 10, height: 10 }), true);
  assert.equal(circleHitsRectangle(0, 0, 4.9, { x: 3, y: 4, width: 10, height: 10 }), false);
  const game = new Game();
  game.start(); game.pipes[0].x = CONFIG.birdX; game.pipes[0].gapCenterY = 500;
  game.step();
  assert.equal(game.status, 'gameover');
  assert.equal(game.score, 0);
});

test('long flights keep generating pipes and expose finite state values', () => {
  const game = new Game();
  game.start();
  // Keep this fixture centered to isolate spawning from player skill.
  for (let i = 0; i < 7200; i++) {
    for (const pipe of game.pipes) pipe.gapCenterY = 320;
    game.bird.y = 320; game.bird.velocityY = 0;
    game.step();
    for (const name of ['birdY', 'birdVelocityY', 'nextPipeDistanceX', 'nextPipeGapCenterY']) {
      assert.ok(Number.isFinite(game.getState()[name]));
    }
  }
  assert.equal(game.status, 'running');
  assert.ok(game.score > 30);
  assert.ok(game.pipes.length <= 5);
});

function centeredStep(game) {
  for (const pipe of game.pipes) pipe.gapCenterY = 320;
  game.bird.y = 320;
  game.bird.velocityY = 0;
  game.step();
}

test('normal difficulty preserves the fixed pipe speed at every score', () => {
  const game = new Game();
  game.start();
  for (const score of [0, 1, 50, 1000]) {
    game.score = score;
    const pipe = game.pipes[0];
    const before = pipe.x;
    centeredStep(game);
    assert.equal(game.pipeSpeed, CONFIG.pipeSpeed);
    assert.equal(pipe.x, before - CONFIG.pipeSpeed * CONFIG.fixedStepSeconds);
  }
});

test('hard difficulty accelerates smoothly only after an earned point', () => {
  const game = new Game({ difficulty: 'hard' });
  game.start();
  while (game.score === 0) {
    centeredStep(game);
    assert.equal(game.pipeSpeed, 180);
  }
  const pipe = game.pipes[0];
  const before = pipe.x;
  centeredStep(game);
  assert.ok(game.pipeSpeed > 180 && game.pipeSpeed < 181.8);
  assert.equal(pipe.x, before - game.pipeSpeed * CONFIG.fixedStepSeconds);
  assert.equal(game.config.jumpVelocity, CONFIG.jumpVelocity);
  assert.equal(game.config.gravity, CONFIG.gravity);
});

test('long hard flights reach the speed cap without exceeding it or losing pipe continuity', () => {
  const game = new Game({ difficulty: 'hard' });
  game.start();
  let previousSpeed = game.pipeSpeed;
  for (let tick = 0; tick < 24000; tick++) {
    centeredStep(game);
    assert.ok(game.pipeSpeed >= previousSpeed && game.pipeSpeed <= 270);
    previousSpeed = game.pipeSpeed;
    for (let index = 1; index < game.pipes.length; index++) {
      assert.ok(Math.abs(game.pipes[index].x - game.pipes[index - 1].x - CONFIG.pipeSpacing) < 1e-7);
    }
  }
  assert.equal(game.status, 'running');
  assert.ok(game.score > 50);
  assert.ok(game.pipeSpeed > 269.99);
});

test('hard replay is deterministic, pause freezes acceleration and reset restores initial speed', () => {
  const game = new Game({ seed: 42, difficulty: 'hard' });
  const replay = () => {
    game.start();
    for (let tick = 0; tick < 1200; tick++) centeredStep(game);
    return { state: game.getState(), pipes: structuredClone(game.pipes) };
  };
  const first = replay();
  game.pause();
  game.step();
  assert.equal(game.pipeSpeed, first.state.pipeSpeed);
  game.resume();
  assert.deepEqual(game.getState(), first.state);
  assert.throws(() => game.reset(99, 'unknown'));
  assert.deepEqual(game.getState(), first.state);
  game.reset();
  assert.equal(game.pipeSpeed, 180);
  assert.equal(game.difficulty, 'hard');
  assert.deepEqual(replay(), first);
  game.reset(42, 'normal');
  assert.equal(game.pipeSpeed, 180);
  assert.equal(game.difficulty, 'normal');
});
