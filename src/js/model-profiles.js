import { INPUT_NAMES, NORMALIZATION } from './config.js';

export const MODEL_PROFILES = Object.freeze({
  'speed-516': Object.freeze({ id: 'speed-516', version: 2, datasetVersion: 3, architecture: [5, 16, 1],
    inputNames: [...INPUT_NAMES, 'pipeSpeed'], normalization: { ...NORMALIZATION, version: 2, maxPipeSpeed: 270 },
    base: './models/speed-516/', hardBase: './models/speed-516/hard/', learning: 'learning-516', api: '/api/learning-516' }),
  'lookahead-616': Object.freeze({ id: 'lookahead-616', version: 3, datasetVersion: 4, architecture: [6, 16, 1],
    inputNames: [...INPUT_NAMES, 'pipeSpeed', 'followingGapY'],
    normalization: { ...NORMALIZATION, version: 3, maxPipeSpeed: 270 },
    base: './models/lookahead-616/', hardBase: './models/lookahead-616/hard/', learning: 'learning-616', api: '/api/learning-616',
    rlBase: './models/rl-616/normal/', rlHardBase: './models/rl-616/hard/' }),
});
export const CSS_INPUT_PROPERTIES = ['--bird-y', '--velocity-y', '--pipe-distance', '--gap-y', '--pipe-speed', '--following-gap-y'];
export function modelProfile(id = 'speed-516') {
  if (!Object.hasOwn(MODEL_PROFILES, id)) throw new Error('Unknown model profile.');
  return MODEL_PROFILES[id];
}
