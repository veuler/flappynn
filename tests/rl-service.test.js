import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { Readable } from 'node:stream';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { createRlService } from '../scripts/rl-service.js';

const host = '127.0.0.1:3030';
function fakeChild() {
  const child = new EventEmitter();
  child.stdout = { resume() {} };
  child.stderr = new EventEmitter();
  return child;
}
async function call(service, pathname, { method = 'GET', data, url = pathname, origin = `http://${host}`, contentType = 'application/json' } = {}) {
  const request = Readable.from(data === undefined ? [] : [JSON.stringify(data)]);
  request.method = method;
  request.url = url;
  request.headers = { host, ...(origin ? { origin } : {}), ...(contentType ? { 'content-type': contentType } : {}) };
  const response = { writeHead(status) { this.status = status; return this; }, end(text) { this.body = text ? JSON.parse(text) : null; } };
  await service(request, response, pathname);
  return response;
}
async function tempRoot() {
  return mkdtemp(path.resolve('artifacts', 'rl-service-test-'));
}

test('RL API enforces local JSON origin and exact settings before spawning', async () => {
  const root = await tempRoot();
  try {
    let spawned = false;
    const service = createRlService(root, { spawnWorker: () => { spawned = true; return fakeChild(); } });
    assert.equal((await call(service, '/api/rl/start', { method: 'POST', origin: 'https://example.com', data: {} })).status, 403);
    assert.equal((await call(service, '/api/rl/start', { method: 'POST', data: { difficulty: 'normal', mode: 'fast', numEnvs: 2, seconds: 300, initialization: 'random' } })).status, 400);
    assert.equal((await call(service, '/api/rl/start', { method: 'POST', data: { difficulty: 'normal', mode: 'live', numEnvs: 32, seconds: 300, initialization: 'random' } })).status, 400);
    assert.equal(spawned, false);
    const response = await call(service, '/api/rl/start', { method: 'POST', data: { difficulty: 'normal', mode: 'fast', numEnvs: 32, seconds: 300, initialization: 'random' } });
    assert.equal(response.status, 202);
    assert.equal(response.body.job.dataKind, 'reinforcement-learning-ppo');
    assert.equal(response.body.job.modelProfile, 'lookahead-616');
    assert.equal(spawned, true);
  } finally { await rm(root, { recursive: true, force: true }); }
});

for (const environmentCount of [256, 512]) test(`${environmentCount} environments launch in fast mode and remain invalid in live mode`, async () => {
  const root = await tempRoot();
  try {
    let workerArgs;
    const service = createRlService(root, { spawnWorker: (_python, args) => { workerArgs = args; return fakeChild(); } });
    const settings = { difficulty: 'hard', numEnvs: environmentCount, seconds: 300, initialization: 'random' };
    assert.equal((await call(service, '/api/rl/start', { method: 'POST', data: { ...settings, mode: 'live' } })).status, 400);
    assert.equal(workerArgs, undefined);
    const response = await call(service, '/api/rl/start', { method: 'POST', data: { ...settings, mode: 'fast' } });
    assert.equal(response.status, 202);
    assert.equal(response.body.job.numEnvs, environmentCount);
    assert.equal(workerArgs[workerArgs.indexOf('--num-envs') + 1], String(environmentCount));
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('RL pause, resume, stop and replay operations are isolated to the active UUID', async () => {
  const root = await tempRoot();
  try {
    const service = createRlService(root, { spawnWorker: () => fakeChild() });
    const started = await call(service, '/api/rl/start', { method: 'POST', data: { mode: 'live', numEnvs: 1, seconds: 300, initialization: 'random' } });
    const id = started.body.job.id;
    const folder = path.join(root, 'src', 'learning-rl-616', id);
    assert.equal((await call(service, '/api/rl/pause', { method: 'POST', data: { id: '00000000-0000-4000-8000-000000000000' } })).status, 400);
    assert.equal((await call(service, '/api/rl/pause', { method: 'POST', data: { id } })).status, 200);
    await writeFile(path.join(folder, 'status.json'), JSON.stringify({ state: 'paused', elapsedSeconds: 5,
      bestScore: 170, bestScoreComplete: true }));
    const resumed = await call(service, '/api/rl/resume', { method: 'POST', data: { id } });
    assert.equal(resumed.status, 200);
    assert.equal(resumed.body.job.bestScore, 170);
    assert.equal(resumed.body.job.bestScoreComplete, true);
    await writeFile(path.join(folder, 'replay.json'), JSON.stringify({ episodes: [{ score: 3 }] }));
    assert.deepEqual((await call(service, '/api/rl/replay', { url: `/api/rl/replay?id=${id}` })).body.replay.episodes, [{ score: 3 }]);
    assert.equal((await call(service, '/api/rl/replay', { url: '/api/rl/replay?id=..%2F..%2Fpackage.json' })).status, 400);
    assert.equal((await call(service, '/api/rl/stop', { method: 'POST', data: { id } })).status, 200);
    assert.equal(await readFile(path.join(folder, 'stop'), 'utf8'), 'stop');
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('saved initialization resolves only the selected RL artifact', async () => {
  const root = await tempRoot();
  try {
    await mkdir(path.join(root, 'artifacts'), { recursive: true });
    const selectedModel = path.join(root, 'artifacts', 'model-rl-616-hard.json');
    await writeFile(selectedModel, '{}');
    let launched;
    const service = createRlService(root, { spawnWorker: (_python, args) => { launched = args; return fakeChild(); } });
    const saved = await call(service, '/api/rl/start', { method: 'POST', data: { difficulty: 'hard', mode: 'fast', numEnvs: 64,
      seconds: 600, initialization: 'saved' } });
    assert.equal(saved.status, 202);
    assert.equal(launched[launched.indexOf('--model') + 1], selectedModel);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('checkpoint settings must match difficulty, mode and environment count but may choose a new duration', async () => {
  const root = await tempRoot();
  try {
    const previousId = '33333333-3333-4333-8333-333333333333';
    const previousFolder = path.join(root, 'src', 'learning-rl-616', previousId);
    await mkdir(previousFolder, { recursive: true });
    await writeFile(path.join(previousFolder, 'training-checkpoint.pt'), 'checkpoint');
    await writeFile(path.join(root, 'src', 'learning-rl-616', 'latest.json'), JSON.stringify({ id: previousId, state: 'awaiting-validation',
      difficulty: 'hard', mode: 'fast', numEnvs: 64, durationSeconds: 600, elapsedSeconds: 12.34 }));
    let launched;
    const service = createRlService(root, { spawnWorker: (_python, args) => { launched = args; return fakeChild(); } });
    const mismatch = await call(service, '/api/rl/start', { method: 'POST', data: { difficulty: 'hard', mode: 'live', numEnvs: 1,
      seconds: 900, initialization: 'checkpoint' } });
    assert.equal(mismatch.status, 400);
    const resumed = await call(service, '/api/rl/start', { method: 'POST', data: { difficulty: 'hard', mode: 'fast', numEnvs: 64,
      seconds: 900, initialization: 'checkpoint' } });
    assert.equal(resumed.status, 202);
    assert.equal(launched[launched.indexOf('--seconds') + 1], '900');
    assert.equal(launched[launched.indexOf('--checkpoint') + 1], path.join(previousFolder, 'training-checkpoint.pt'));
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('RL save stays blocked until parity is verified and startup recovers dead workers', async () => {
  const root = await tempRoot();
  const id = '11111111-1111-4111-8111-111111111111';
  const folder = path.join(root, 'src', 'learning-rl-616', id);
  try {
    await mkdir(folder, { recursive: true });
    await writeFile(path.join(root, 'src', 'learning-rl-616', 'latest.json'), JSON.stringify({ id, state: 'running', difficulty: 'normal', mode: 'fast', numEnvs: 32, durationSeconds: 300 }));
    await writeFile(path.join(folder, 'status.json'), JSON.stringify({ state: 'running', elapsedSeconds: 10 }));
    const service = createRlService(root);
    const recovered = await call(service, '/api/rl');
    assert.equal(recovered.body.job.state, 'failed');
    assert.equal(recovered.body.canResume, false);
    assert.equal((await call(service, '/api/rl/save', { method: 'POST', data: { id } })).status, 400);
    assert.match((await call(service, '/api/rl/save', { method: 'POST', data: { id } })).body.error, /parity/i);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('RL save installs only the candidate whose browser report passes exact CSS parity', async () => {
  const root = await tempRoot();
  const id = '22222222-2222-4222-8222-222222222222';
  const folder = path.join(root, 'src', 'learning-rl-616', id);
  try {
    const templateBytes = await readFile('artifacts/model-616.json');
    const elapsedSeconds = 12.34;
    const modelBytes = Buffer.from(JSON.stringify({ ...JSON.parse(templateBytes), training: {
      dataKind: 'reinforcement-learning-ppo', difficulty: 'normal', mode: 'fast', numEnvs: 32, trainingSeconds: elapsedSeconds,
    } }));
    const cssBytes = await readFile('src/models/lookahead-616/model.css');
    const meta = JSON.parse(await readFile('src/models/lookahead-616/model.meta.json'));
    const modelSha256 = createHash('sha256').update(modelBytes).digest('hex');
    const cssSha256 = createHash('sha256').update(cssBytes).digest('hex');
    const metadata = { ...meta, modelSha256, cssSha256, parityVerified: false, parity: undefined,
      dataKind: 'reinforcement-learning-ppo', trainingDifficulty: 'normal', trainingMode: 'fast', numEnvs: 32, trainingSeconds: elapsedSeconds };
    delete metadata.parity;
    const cases = Array.from({ length: 20 }, (_, index) => ({ id: `case-${index}`, inputs: [0, 0, 0, 0, 180],
      expected: { probability: 0.5, logit: 0, hidden: Array(16).fill(0) } }));
    const fixture = { version: 1, tolerance: 1e-4, threshold: 0.5, expectedProbe: 0.7310585786300049,
      cssSha256, modelSha256, cases };
    const report = { version: 1, passed: true, tolerance: 1e-4, probe: fixture.expectedProbe, cssSha256, modelSha256,
      browser: 'test browser', checkedAt: new Date().toISOString(), caseCount: cases.length,
      cases: cases.map(row => ({ id: row.id, inputs: row.inputs, actualProbability: 0.5, actualLogit: 0, actualHidden: Array(16).fill(0) })) };
    await mkdir(folder, { recursive: true });
    await mkdir(path.join(root, 'src', 'models', 'rl-616', 'normal'), { recursive: true });
    await mkdir(path.join(root, 'artifacts'), { recursive: true });
    await writeFile(path.join(root, 'artifacts', 'model-616.json'), 'existing lookahead-616 slot');
    await writeFile(path.join(folder, 'candidate.json'), modelBytes);
    await writeFile(path.join(folder, 'model.css'), cssBytes);
    await writeFile(path.join(folder, 'model.meta.json'), JSON.stringify(metadata));
    await writeFile(path.join(folder, 'parity-inputs.json'), JSON.stringify(fixture));
    await writeFile(path.join(root, 'src', 'learning-rl-616', 'latest.json'), JSON.stringify({ id, state: 'awaiting-validation', difficulty: 'normal', mode: 'fast', numEnvs: 32, durationSeconds: 300, elapsedSeconds }));
    const service = createRlService(root, { spawnVerifier: (_executable, args) => {
      const child = fakeChild();
      const metadataPath = args[args.indexOf('--metadata') + 1];
      writeFile(metadataPath, JSON.stringify({ ...metadata, parityVerified: true, parity: {
        browser: report.browser, checkedAt: report.checkedAt, caseCount: cases.length, tolerance: 1e-4,
        maximumProbabilityError: 0, maximumLogitError: 0, maximumHiddenError: 0, decisionMismatches: 0, nearThresholdCases: cases.length,
      } })).then(() => setImmediate(() => child.emit('exit', 0)));
      return child;
    } });
    assert.equal((await call(service, '/api/rl/save', { method: 'POST', data: { id } })).status, 400);
    const unverified = JSON.parse(await readFile(path.join(folder, 'model.meta.json'), 'utf8'));
    assert.equal(unverified.parityVerified, false);
    const parity = await call(service, '/api/rl/parity', { method: 'POST', data: { id, report } });
    assert.equal(parity.status, 200, parity.body?.error);
    assert.equal(parity.body.job.candidateCssVerified, true);
    const result = await call(service, '/api/rl/save', { method: 'POST', data: { id } });
    assert.equal(result.status, 200, result.body?.error);
    assert.equal(result.body.job.state, 'saved');
    assert.deepEqual(await readFile(path.join(root, 'artifacts', 'model-rl-616-normal.json')), modelBytes);
    assert.deepEqual(await readFile(path.join(root, 'src', 'models', 'rl-616', 'normal', 'model.css')), cssBytes);
    assert.equal(JSON.parse(await readFile(path.join(root, 'src', 'models', 'rl-616', 'normal', 'model.meta.json'))).parityVerified, true);
    assert.equal(await readFile(path.join(root, 'artifacts', 'model-616.json'), 'utf8'), 'existing lookahead-616 slot');
  } finally { await rm(root, { recursive: true, force: true }); }
});
