import { CONFIG, HARD_DIFFICULTY_RULES } from './config.js';
import { modelProfile } from './model-profiles.js';

export function validateRecordingDataset(data, profileId = 'speed-516') {
  const profile = modelProfile(profileId);
  const require = (condition, message) => { if (!condition) throw new Error(message); };
  const same = (actual, expected) => Object.entries(expected).every(([key,value]) => actual?.[key] === value);
  require(data && typeof data === 'object', 'Choose a recording JSON file.');
  require(data.version === profile.datasetVersion, 'This recording belongs to a different model lab. Record new flights for the six-input lab.');
  require(data.modelProfile === profileId && JSON.stringify(data.inputNames) === JSON.stringify(profile.inputNames), 'Recording inputs do not match this lab.');
  require(same(data.normalization,profile.normalization) && same(data.gameConfig,CONFIG), 'Recording physics do not match this game.');
  require(same(data.difficultyRules?.hard,HARD_DIFFICULTY_RULES), 'Recording difficulty rules do not match.');
  require(data.sampleIntervalMs === 50 && data.minimumJumpIntervalMs === 100 && data.clock === 'simulation' && data.sampleTiming === 'before-action', 'Recording timing does not match.');
  require(Array.isArray(data.samples) && data.samples.length > 0 && data.samples.length <= CONFIG.maxTrainingSamples, 'Recordings must contain 1 to 100,000 samples.');
  let jumps = 0;
  for (const row of data.samples) {
    require(Array.isArray(row) && row.length === profile.architecture[0]+1 && row.every(Number.isFinite), 'Invalid recording sample.');
    require(row.slice(0,-1).every((v,i)=>v >= (i===1 ? -1 : 0) && v<=1), 'A recording input is outside its range.');
    require(row.at(-1)===0 || row.at(-1)===1,'Recording actions must be 0 or 1.');jumps+=row.at(-1);
  }
  require(data.counts?.total===data.samples.length && data.counts.jump===jumps && data.counts.wait===data.samples.length-jumps,'Recording counts disagree.');
  require(Array.isArray(data.sessions) && data.sessions.length>0,'Recording flights are missing.');
  let offset=0; const ids=new Set();
  for(const flight of data.sessions) {
    require(Number.isInteger(flight.id) && flight.id>0 && !ids.has(flight.id),'Recording flight IDs are invalid.');ids.add(flight.id);
    require(Number.isInteger(flight.seed) && flight.seed>=0 && flight.seed<=4294967295,'Invalid recording seed.');
    require(['normal','hard'].includes(flight.difficulty),'Invalid recording difficulty.');
    require(flight.startIndex===offset && Number.isInteger(flight.sampleCount) && flight.sampleCount>0,'Recording flight ranges disagree.');
    require(Number.isInteger(flight.score) && flight.score>=0 && Number.isFinite(flight.elapsedSeconds) && flight.elapsedSeconds>=0 && flight.sampleCount<=Math.floor((flight.elapsedSeconds+1e-7)*20)+1,'Invalid flight score, duration or sample count.');
    require(['collision','restart','stopped','in-progress'].includes(flight.endReason),'Invalid flight ending.');offset+=flight.sampleCount;
  }
  require(offset===data.samples.length,'Recording flights do not cover the samples.');
  return data;
}
