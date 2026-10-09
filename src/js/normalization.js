import { NORMALIZATION } from './config.js';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

export function normalizeState(state, normalization = NORMALIZATION) {
  const values = [state.birdY, state.birdVelocityY, state.nextPipeDistanceX, state.nextPipeGapCenterY];
  if (!values.every(Number.isFinite)) throw new Error('The game state contains missing or invalid values.');
  const inputs = [
    clamp(state.birdY / normalization.height, 0, 1),
    clamp(state.birdVelocityY / normalization.velocityScale, -1, 1),
    clamp(state.nextPipeDistanceX / normalization.maxPipeDistance, 0, 1),
    clamp(state.nextPipeGapCenterY / normalization.height, 0, 1),
  ];
  if (normalization.version >= 2) {
    if (!Number.isFinite(state.pipeSpeed)) throw new Error('The game state is missing pipe speed.');
    inputs.push(clamp(state.pipeSpeed / normalization.maxPipeSpeed, 0, 1));
  }
  if (normalization.version === 3) {
    if (!Number.isFinite(state.followingPipeGapCenterY)) throw new Error('The game state is missing the following pipe gap.');
    inputs.push(clamp(state.followingPipeGapCenterY / normalization.height, 0, 1));
  }
  return inputs;
}
