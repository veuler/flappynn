export function createRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function createPipe(x, random, config) {
  const minimum = config.gapMargin + config.pipeGap / 2;
  const maximum = config.height - minimum;
  return { x, gapCenterY: minimum + random() * (maximum - minimum), passed: false };
}

export function nextPipe(pipes, birdX, config) {
  return pipes.find((pipe) => pipe.x + config.pipeWidth >= birdX - config.birdRadius) ?? null;
}
