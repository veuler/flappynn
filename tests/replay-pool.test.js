import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ReplayPool } from '../src/js/replay-pool.js';

const flight = (id, frameCount = 2, extra = {}) => ({ episodeId: id, seed: id * 10,
  source: 'training-replay', complete: true, frames: Array.from({ length: frameCount }, (_, t) => ({ t })), ...extra });

test('random choice can revisit an older flight and excludes the just-played flight', () => {
  const pool = new ReplayPool({ random: () => 0 });
  const flights = [flight(8), flight(9), flight(10)];
  flights.forEach(replay => pool.add(replay));
  assert.equal(pool.pick(flights[2]), flights[0]);
  assert.equal(pool.pick(flights[0]), flights[1]);
  pool.random = () => .99;
  assert.equal(pool.pick(flights[0]), flights[2]);
});

test('partial/live streams and duplicate polling do not fill the random pool', () => {
  const pool = new ReplayPool();
  const replay = flight(3);
  pool.add(replay); pool.add({ ...replay });
  pool.add(flight(4, 2, { complete: false }));
  pool.add(flight(5, 2, { source: 'live-training' }));
  assert.equal(pool.episodes.length, 1);
  assert.equal(pool.frameCount, 2);
  assert.equal(pool.pick(replay), null);
});

test('viewing history respects both memory limits and clears between jobs', () => {
  const pool = new ReplayPool({ maxEpisodes: 2, maxFrames: 5, random: () => 0 });
  pool.add(flight(1, 3)); pool.add(flight(2, 3));
  assert.deepEqual(pool.episodes.map(replay => replay.episodeId), [2]);
  pool.add(flight(3, 1)); pool.add(flight(4, 1));
  assert.deepEqual(pool.episodes.map(replay => replay.episodeId), [3, 4]);
  pool.add(flight(5, 6));
  assert.deepEqual(pool.episodes.map(replay => replay.episodeId), [3, 4]);
  pool.clear();
  assert.equal(pool.pick(), null);
  assert.equal(pool.frameCount, 0);
});

test('fast playback chooses an older random episode before a queued latest stream', () => {
  const pool = new ReplayPool({ random: () => 0 });
  const older = flight(8), current = flight(10);
  [older, flight(9), current].forEach(replay => pool.add(replay));
  const queued = flight(11, 2, { complete: false });
  assert.equal(pool.next(current, queued), older);
  assert.equal(pool.next({ ...current, complete: false }, queued), older);
  pool.clear();
  assert.equal(pool.next(current, queued), queued);
  assert.equal(pool.next(current, current), null);
});

test('live playback follows the queued environment without choosing random history', () => {
  const pool = new ReplayPool({ random: () => 0 });
  pool.add(flight(8));
  const current = flight(10, 2, { source: 'live-training' });
  const queued = flight(11, 2, { source: 'live-training', complete: false });
  assert.equal(pool.next(current, queued), queued);
  assert.equal(pool.next(current, null), null);
});

test('default pool retains several full 600-second flights for random choice', () => {
  const pool = new ReplayPool({ random: () => 0 });
  const flights = [flight(1, 12001), flight(2, 12001), flight(3, 12001), flight(4, 12001)];
  flights.forEach(replay => pool.add(replay));
  assert.equal(pool.episodes.length, 4);
  assert.equal(pool.pick(flights[3]), flights[0]);
  pool.add(flight(5, 12001));
  assert.equal(pool.episodes.length, 4);
  assert.ok(pool.frameCount <= 50000);
});
