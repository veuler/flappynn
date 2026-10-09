import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { createLearningService, sameOrigin, sameModelBehavior, modelPaths } from '../scripts/learning-service.js';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { HARD_DIFFICULTY_RULES } from '../src/js/config.js';

test('training rejects other origins and host spoofing', () => {
  const headers = { host: '127.0.0.1:3030', origin: 'http://127.0.0.1:3030' };
  assert.equal(sameOrigin({ headers }), true);
  assert.equal(sameOrigin({ headers: { ...headers, origin: 'https://example.com' } }), false);
  assert.equal(sameOrigin({ headers: { ...headers, host: 'example.com:3030', origin: 'http://example.com:3030' } }), false);
  assert.equal(sameOrigin({ headers: { host: headers.host } }), false);
});

test('unchanged candidates skip gameplay only after CSS verification and matching current/baseline digests', async () => {
  const artifacts = path.resolve('artifacts');
  const root = await mkdtemp(path.join(artifacts, 'learning-service-test-'));
  const id = '33333333-3333-4333-8333-333333333333';
  const folder = path.join(root, 'src', 'learning-516', id);
  const original = await readFile('tests/fixtures/speed-model.json');
  const model = JSON.parse(original);
  const metadata = JSON.parse(await readFile('tests/fixtures/speed-model.meta.json'));
  const css = await readFile('tests/fixtures/speed-model.css');
  const digest = bytes => createHash('sha256').update(bytes).digest('hex');
  assert.equal(sameModelBehavior(model, { ...model, createdAt: 'a different export' }), true);
  assert.equal(sameModelBehavior({}, {}), false);
  try {
    await mkdir(path.join(folder, 'baseline'), { recursive: true });
    await mkdir(path.join(root, 'artifacts'));
    await mkdir(path.dirname(modelPaths(root).metadata), { recursive: true });
    for (const scenario of ['equal', 'weights', 'threshold', 'unverified', 'tampered-css', 'stale']) {
      const candidate = structuredClone(model);
      candidate.createdAt = 'different export';
      if (scenario === 'weights') candidate.w2[0] += 0.001;
      if (scenario === 'threshold') candidate.threshold = candidate.threshold === .62 ? .5 : .62;
      const candidateBytes = Buffer.from(JSON.stringify(candidate));
      await rm(path.join(folder, 'result.json'), { force: true });
      await writeFile(path.join(root, 'src', 'learning-516', 'latest.json'), JSON.stringify({ id, state: 'validating', threshold: model.threshold }));
      await writeFile(path.join(folder, 'candidate.json'), candidateBytes);
      await writeFile(path.join(folder, 'model.css'), scenario === 'tampered-css' ? 'changed CSS' : css);
      await writeFile(path.join(folder, 'model.meta.json'), JSON.stringify({ ...metadata, modelSha256: digest(candidateBytes), parityVerified: scenario !== 'unverified' }));
      await writeFile(path.join(folder, 'baseline', 'model.json'), original);
      await writeFile(path.join(folder, 'baseline', 'model.meta.json'), JSON.stringify(metadata));
      await writeFile(modelPaths(root).model, scenario === 'stale' ? candidateBytes : original);
      await writeFile(modelPaths(root).metadata, JSON.stringify(metadata));
      const service = createLearningService(root);
      const request = Readable.from([]);
      request.method = 'GET'; request.headers = { host: '127.0.0.1:3030' };
      const response = { writeHead() { return this; }, end(body) { this.body = JSON.parse(body); } };
      await service(request, response, '/api/learning-516');
      assert.equal(response.body.job.state, scenario === 'equal' ? 'kept' : 'validating', scenario);
      assert.equal(response.body.job.comparisonSkipped, scenario === 'equal' ? 'unchanged-weights' : undefined);
      assert.deepEqual(await readFile(modelPaths(root).model), scenario === 'stale' ? candidateBytes : original);
    }
  } finally {
    assert.ok(path.resolve(root).startsWith(`${artifacts}${path.sep}`));
    await rm(root, { recursive: true, force: true });
  }
});

test('training API validates requests before launching CPU work', async () => {
  const service = createLearningService(process.cwd());
  for (const [origin, data, expected] of [['https://example.com', {}, 403], ['http://127.0.0.1:3030', { seconds: -1, threshold: .62 }, 400], ['http://127.0.0.1:3030', { seconds: 300, threshold: .62, difficulty: 'unknown' }, 400]]) {
    const request = Readable.from([JSON.stringify(data)]);
    request.method = 'POST';
    request.headers = { host: '127.0.0.1:3030', origin, 'content-type': 'application/json' };
    const response = { writeHead(status) { this.status = status; return this; }, end(body) { this.body = JSON.parse(body); } };
    assert.equal(await service(request, response, '/api/learning-516/start'), true);
    assert.equal(response.status, expected);
    assert.ok(response.body.error);
  }
});

test('human reset passes random initialization to the worker, snapshots data and preserves saved weights', async () => {
  const root = await mkdtemp(path.resolve('artifacts', 'random-start-test-'));
  const source = modelPaths(root, 'normal', 'speed-516');
  const dataset = JSON.parse(await readFile('tests/fixtures/recording-flights-516.json'));
  const original = await readFile('artifacts/model-516.json');
  const metadata = await readFile('src/models/speed-516/model.meta.json');
  let launched;
  const worker = new EventEmitter(); worker.stdout = { resume() {} }; worker.stderr = new EventEmitter();
  try {
    await mkdir(path.dirname(source.metadata), { recursive: true });
    await mkdir(path.dirname(source.model), { recursive: true });
    await writeFile(source.model, original); await writeFile(source.metadata, metadata);
    await writeFile(source.css, await readFile('src/models/speed-516/model.css'));
    const service = createLearningService(root, 'speed-516', { spawnWorker: (...args) => { launched = args; return worker; } });
    const call = async initialization => {
      const request = Readable.from([JSON.stringify({ dataset, difficulty: 'hard', threshold: .5, initialization })]);
      request.method = 'POST'; request.headers = { host: '127.0.0.1:3030', origin: 'http://127.0.0.1:3030', 'content-type': 'application/json' };
      const response = { writeHead(status) { this.status = status; return this; }, end(text) { this.body = JSON.parse(text); } };
      await service(request, response, '/api/learning-516/human'); return response;
    };
    assert.equal((await call('zero-everything')).status, 400);
    assert.equal(launched, undefined);
    const response = await call('random');
    assert.equal(response.status, 202); assert.equal(response.body.job.initialization, 'random');
    assert.equal(launched[1][launched[1].indexOf('--initialization') + 1], 'random');
    const folder = path.join(root, 'src', 'learning-516', response.body.job.id);
    assert.deepEqual(JSON.parse(await readFile(path.join(folder, 'recordings.json'))), dataset);
    assert.deepEqual(await readFile(path.join(folder, 'baseline', 'model.json')), original);
    assert.deepEqual(await readFile(source.model), original);
    assert.deepEqual(await readFile(source.metadata), metadata);
  } finally {
    assert.ok(root.startsWith(path.resolve('artifacts') + path.sep));
    await rm(root, { recursive: true, force: true });
  }
});

test('manual acceptance permits score regressions but still rejects incomplete, unverified, tampered, mismatched and stale bundles', async () => {
  const root = await mkdtemp(path.resolve('artifacts', 'manual-promotion-test-'));
  const id = '66666666-6666-4666-8666-666666666666';
  const folder = path.join(root, 'src', 'learning-516', id);
  const source = modelPaths(root, 'normal', 'speed-516');
  const target = modelPaths(root, 'hard', 'speed-516');
  const original = await readFile('artifacts/model-516.json');
  const originalMeta = JSON.parse(await readFile('src/models/speed-516/model.meta.json'));
  const css = await readFile('src/models/speed-516/model.css');
  const candidate = Buffer.from(JSON.stringify({ ...JSON.parse(original), threshold: .62 }));
  const digest = bytes => createHash('sha256').update(bytes).digest('hex');
  const candidateMeta = { ...originalMeta, threshold: .62, modelSha256: digest(candidate), trainingDifficulty: 'hard', difficultyConfig: { ...HARD_DIFFICULTY_RULES } };
  const call = async (service, data) => {
    const request = Readable.from([JSON.stringify(data)]);
    request.method = 'POST'; request.headers = { host: '127.0.0.1:3030', origin: 'http://127.0.0.1:3030', 'content-type': 'application/json' };
    const response = { writeHead(status) { this.status = status; return this; }, end(text) { this.body = JSON.parse(text); } };
    await service(request, response, '/api/learning-516/promote'); return response;
  };
  try {
    await mkdir(path.join(folder, 'baseline'), { recursive: true });
    await mkdir(path.dirname(source.metadata), { recursive: true });
    await mkdir(path.dirname(source.model), { recursive: true });
    // These other slots must remain untouched throughout all scenarios.
    const other = modelPaths(root, 'normal', 'lookahead-616'); const otherHard = modelPaths(root, 'hard', 'lookahead-616');
    await mkdir(path.dirname(other.metadata), { recursive: true });
    await mkdir(path.dirname(otherHard.metadata), { recursive: true });
    await writeFile(other.model, 'other normal'); await writeFile(otherHard.model, 'other hard');
    for (const scenario of ['incomplete', 'unverified', 'tampered-css', 'wrong-profile', 'wrong-difficulty', 'mismatch', 'stale', 'backup-changed', 'accepted']) {
      await rm(path.join(folder, 'result.json'), { force: true });
      const baseline = { modelSha256: originalMeta.modelSha256, threshold: .62, difficulty: 'hard', maximumSimulationSeconds: 180,
        runs: Array.from({ length: 20 }, (_, i) => ({ seed: i + 1, score: 10, survivalSeconds: 20 })) };
      const result = { ...baseline, modelSha256: candidateMeta.modelSha256, runs: baseline.runs.map((run, i) => ({ ...run, score: i === 0 ? 5 : 11 })) };
      const offline = structuredClone(result);
      if (scenario === 'mismatch') offline.runs[0].score++;
      await writeFile(path.join(root, 'src', 'learning-516', 'latest.json'), JSON.stringify({ id, state: scenario === 'incomplete' ? 'validating' : 'kept', threshold: .62, difficulty: 'hard' }));
      await writeFile(path.join(folder, 'report.json'), JSON.stringify({ difficulty: 'hard', baselineTest: baseline, candidateTest: offline }));
      await writeFile(path.join(folder, 'browser-gameplay.json'), JSON.stringify({ baseline, candidate: result, promoted: false }));
      await writeFile(path.join(folder, 'candidate.json'), candidate);
      await writeFile(path.join(folder, 'model.css'), scenario === 'tampered-css' ? 'modified CSS' : css);
      await writeFile(path.join(folder, 'model.meta.json'), JSON.stringify({ ...candidateMeta, ...(scenario === 'unverified' ? { parityVerified: false } : {}), ...(scenario === 'wrong-profile' ? { modelProfile: 'lookahead-616' } : {}), ...(scenario === 'wrong-difficulty' ? { trainingDifficulty: 'normal' } : {}) }));
      await writeFile(path.join(folder, 'baseline', 'model.json'), scenario === 'backup-changed' ? 'changed backup' : original);
      await writeFile(path.join(folder, 'baseline', 'model.css'), css);
      await writeFile(path.join(folder, 'baseline', 'model.meta.json'), JSON.stringify(originalMeta));
      await writeFile(source.model, scenario === 'stale' ? candidate : original);
      await writeFile(source.metadata, JSON.stringify(originalMeta)); await writeFile(source.css, css);
      const service = createLearningService(root, 'speed-516');
      assert.equal((await call(service, { id: 'wrong-id' })).status, 400);
      const response = await call(service, { id });
      if (scenario === 'accepted') {
        assert.equal(response.status, 200); assert.equal(response.body.job.promotionMode, 'manual');
        assert.equal(response.body.job.state, 'promoted');
        assert.deepEqual(await readFile(target.model), candidate);
        assert.deepEqual(await readFile(target.css), css);
        assert.equal(JSON.parse(await readFile(target.metadata)).modelSha256, candidateMeta.modelSha256);
        assert.deepEqual(await readFile(path.join(folder, 'baseline', 'model.json')), original);
        assert.equal((await call(service, { id })).status, 400);
        const reloaded = await call(createLearningService(root, 'speed-516'), { id });
        assert.equal(reloaded.status, 400); // A restarted service also recognizes the saved promotion.
      } else {
        assert.equal(response.status, 400, scenario);
        await assert.rejects(readFile(target.model), { code: 'ENOENT' });
      }
      assert.deepEqual(await readFile(source.model), scenario === 'stale' ? candidate : original);
      assert.equal(await readFile(other.model, 'utf8'), 'other normal');
      assert.equal(await readFile(otherHard.model, 'utf8'), 'other hard');
    }
  } finally {
    assert.ok(root.startsWith(path.resolve('artifacts') + path.sep));
    await rm(root, { recursive: true, force: true });
  }
});

test('normal and hard slots are separate and unknown difficulty cannot construct a path', () => {
  const root = process.cwd();
  assert.notEqual(modelPaths(root).model, modelPaths(root, 'hard').model);
  assert.notEqual(modelPaths(root).css, modelPaths(root, 'hard').css);
  assert.notEqual(modelPaths(root).metadata, modelPaths(root, 'hard').metadata);
  assert.throws(() => modelPaths(root, '../escape'));
});

test('hard promotion rejects normal measurements and writes only the hard slot, including its first fallback baseline', async () => {
  const artifacts = path.resolve('artifacts');
  const root = await mkdtemp(path.join(artifacts, 'learning-service-test-'));
  const id = '44444444-4444-4444-8444-444444444444';
  const folder = path.join(root, 'src', 'learning-516', id);
  const normal = modelPaths(root);
  const hard = modelPaths(root, 'hard');
  const original = await readFile('tests/fixtures/speed-model.json');
  const originalMeta = JSON.parse(await readFile('tests/fixtures/speed-model.meta.json'));
  const css = await readFile('tests/fixtures/speed-model.css');
  const candidate = Buffer.from(JSON.stringify({ ...JSON.parse(original), threshold: .62 }));
  const candidateMeta = { ...originalMeta, threshold: .62, trainingDifficulty: 'hard', difficultyConfig: { ...HARD_DIFFICULTY_RULES }, modelSha256: createHash('sha256').update(candidate).digest('hex') };
  const call = async (service, operation, data) => {
    const request = Readable.from([JSON.stringify(data)]);
    request.method = 'POST'; request.headers = { host: '127.0.0.1:3030', origin: 'http://127.0.0.1:3030', 'content-type': 'application/json' };
    const response = { writeHead(status) { this.status = status; return this; }, end(body) { this.body = JSON.parse(body); } };
    await service(request, response, `/api/learning-516/${operation}`);
    return response;
  };
  try {
    await mkdir(path.join(folder, 'baseline'), { recursive: true });
    await mkdir(path.join(root, 'artifacts'));
    await mkdir(path.dirname(modelPaths(root).metadata), { recursive: true });
    await writeFile(normal.model, original); await writeFile(normal.metadata, JSON.stringify(originalMeta)); await writeFile(normal.css, css);
    for (const existingHard of [false, true]) {
      if (existingHard) {
        await mkdir(path.dirname(hard.metadata), { recursive: true });
        await writeFile(hard.model, original); await writeFile(hard.metadata, JSON.stringify(originalMeta)); await writeFile(hard.css, css);
      }
      await rm(path.join(folder, 'result.json'), { force: true });
      await writeFile(path.join(root, 'src', 'learning-516', 'latest.json'), JSON.stringify({ id, state: 'validating', threshold: .62, difficulty: 'hard' }));
      await writeFile(path.join(folder, 'baseline', 'model.json'), original);
      await writeFile(path.join(folder, 'baseline', 'model.meta.json'), JSON.stringify(originalMeta));
      await writeFile(path.join(folder, 'candidate.json'), candidate);
      await writeFile(path.join(folder, 'model.css'), css);
      await writeFile(path.join(folder, 'model.meta.json'), JSON.stringify(candidateMeta));
      const baseline = { modelSha256: originalMeta.modelSha256, difficulty: 'hard', threshold: .62, maximumSimulationSeconds: 180,
        runs: Array.from({ length: 20 }, (_, i) => ({ seed: i + 1, score: 10, survivalSeconds: 20 })) };
      const result = { ...baseline, modelSha256: candidateMeta.modelSha256, runs: baseline.runs.map(run => ({ ...run, score: 11 })) };
      await writeFile(path.join(folder, 'report.json'), JSON.stringify({ difficulty: 'hard', baselineTest: baseline, candidateTest: result }));
      const service = createLearningService(root);
      const claim = await call(service, 'claim', { id });
      assert.equal(claim.body.claimed, true);
      const wrong = await call(service, 'result', { id, token: claim.body.token, baseline: { ...baseline, difficulty: 'normal' }, candidate: result });
      assert.equal(wrong.status, 400);
      const accepted = await call(service, 'result', { id, token: claim.body.token, baseline, candidate: result });
      assert.equal(accepted.body.job.state, 'promoted');
      assert.deepEqual(await readFile(hard.model), candidate);
      assert.deepEqual(await readFile(normal.model), original);
      assert.deepEqual(await readFile(normal.css), css);
      assert.deepEqual(JSON.parse(await readFile(normal.metadata)), originalMeta);
    }
  } finally {
    assert.ok(path.resolve(root).startsWith(`${artifacts}${path.sep}`));
    await rm(root, { recursive: true, force: true });
  }
});

test('failed training exposes validation retry only when a complete candidate was saved', async () => {
  const artifacts = path.resolve('artifacts');
  const root = await mkdtemp(path.join(artifacts, 'learning-service-test-'));
  const id = '22222222-2222-4222-8222-222222222222';
  const folder = path.join(root, 'src', 'learning-516', id);
  try {
    await mkdir(folder, { recursive: true });
    await writeFile(path.join(root, 'src', 'learning-516', 'latest.json'), JSON.stringify({ id, state: 'failed' }));
    const service = createLearningService(root);
    const retryAvailable = async () => {
      const request = Readable.from([]);
      request.method = 'GET'; request.headers = { host: '127.0.0.1:3030' };
      const response = { writeHead() { return this; }, end(body) { this.body = JSON.parse(body); } };
      await service(request, response, '/api/learning-516');
      return response.body.job.canRetryValidation;
    };
    assert.equal(await retryAvailable(), false);
    await writeFile(path.join(folder, 'candidate.json'), '{}');
    assert.equal(await retryAvailable(), false);
    await writeFile(path.join(folder, 'model.meta.json'), '{}');
    await writeFile(path.join(folder, 'report.json'), '{}');
    assert.equal(await retryAvailable(), true);
  } finally {
    assert.ok(path.resolve(root).startsWith(`${artifacts}${path.sep}`));
    await rm(root, { recursive: true, force: true });
  }
});

test('deployment gate preserves equal, regressed, mismatched or stale models and installs a better verified candidate', async () => {
  const artifacts = path.resolve('artifacts');
  const root = await mkdtemp(path.join(artifacts, 'learning-service-test-'));
  assert.ok(root.startsWith(`${artifacts}${path.sep}`));
  const id = '11111111-1111-4111-8111-111111111111';
  const folder = path.join(root, 'src', 'learning-516', id);
  const original = await readFile('tests/fixtures/speed-model.json');
  const originalMeta = JSON.parse(await readFile('tests/fixtures/speed-model.meta.json', 'utf8'));
  const originalCss = await readFile('tests/fixtures/speed-model.css');
  const candidate = Buffer.from(JSON.stringify({ ...JSON.parse(original), threshold: .62 }));
  const candidateMeta = { ...originalMeta, threshold: .62, modelSha256: createHash('sha256').update(candidate).digest('hex') };
  const call = async (service, operation, data) => {
    const request = Readable.from([JSON.stringify(data)]);
    request.method = 'POST';
    request.headers = { host: '127.0.0.1:3030', origin: 'http://127.0.0.1:3030', 'content-type': 'application/json' };
    const response = { writeHead(status) { this.status = status; return this; }, end(body) { this.body = JSON.parse(body); } };
    await service(request, response, `/api/learning-516/${operation}`);
    return response;
  };
  try {
    await mkdir(path.join(folder, 'baseline'), { recursive: true });
    await mkdir(path.join(root, 'artifacts'));
    await mkdir(path.dirname(modelPaths(root).metadata), { recursive: true });
    for (const scenario of ['equal', 'small-regression', 'too-little-gain', 'regression', 'mismatch', 'stale', 'better']) {
      const baseline = { modelSha256: originalMeta.modelSha256, threshold: .62, maximumSimulationSeconds: 180,
        runs: Array.from({ length: 20 }, (_, i) => ({ seed: i + 1, score: 10, survivalSeconds: 20 })) };
      const result = { ...baseline, modelSha256: candidateMeta.modelSha256,
        runs: baseline.runs.map((run, i) => ({ ...run, score: scenario === 'equal' ? 10 : scenario === 'regression' && i < 3 ? 8 : scenario === 'small-regression' ? i === 0 ? 9 : 12 : scenario === 'too-little-gain' ? i < 11 ? 10 : 11 : 11 })) };
      const report = { baselineTest: baseline, candidateTest: JSON.parse(JSON.stringify(result)) };
      if (scenario === 'mismatch') report.candidateTest.runs[0].score++;
      await writeFile(path.join(root, 'src', 'learning-516', 'latest.json'), JSON.stringify({ id, state: 'validating', threshold: .62 }));
      await writeFile(path.join(folder, 'report.json'), JSON.stringify(report));
      await writeFile(path.join(folder, 'model.meta.json'), JSON.stringify(candidateMeta));
      await writeFile(path.join(folder, 'baseline', 'model.meta.json'), JSON.stringify(originalMeta));
      await writeFile(path.join(folder, 'candidate.json'), candidate);
      await writeFile(path.join(folder, 'model.css'), originalCss);
      await writeFile(modelPaths(root).model, original);
      await writeFile(path.join(root, 'src', 'model.css'), originalCss);
      await writeFile(modelPaths(root).metadata, JSON.stringify(scenario === 'stale' ? { ...originalMeta, modelSha256: '0'.repeat(64) } : originalMeta));
      const service = createLearningService(root);
      const claim = await call(service, 'claim', { id });
      assert.equal(claim.body.claimed, true);
      assert.equal(claim.body.job.state, 'validating');
      const otherTab = await call(service, 'claim', { id });
      assert.equal(otherTab.body.claimed, false);
      const progress = await call(service, 'progress', { id, token: claim.body.token, phase: 'baseline', completed: 7, currentSeed: 8 });
      assert.equal(progress.body.job.validationProgress.completed, 7);
      assert.equal(progress.body.job.validationProgress.total, 40);
      const foreignProgress = await call(service, 'progress', { id, token: 'other-tab', phase: 'candidate', completed: 20 });
      assert.equal(foreignProgress.status, 400);
      const request = Readable.from([]);
      request.method = 'GET'; request.headers = { host: '127.0.0.1:3030' };
      const observed = { writeHead() { return this; }, end(body) { this.body = JSON.parse(body); } };
      await service(request, observed, '/api/learning-516');
      assert.equal(observed.body.job.validationProgress.currentSeed, 8);
      const response = await call(service, 'result', { id, token: claim.body.token, baseline, candidate: result });
      if (scenario === 'mismatch' || scenario === 'stale') assert.equal(response.status, 400);
      else {
        const accepted = ['better', 'small-regression'].includes(scenario);
        assert.equal(response.body.job.state, accepted ? 'promoted' : 'kept');
        assert.equal(response.body.job.automaticAcceptance.accepted, accepted);
        assert.equal(response.body.job.automaticAcceptance.policy.maximumCatastrophicRegressions, 1);
      }
      assert.deepEqual(await readFile(modelPaths(root).model), ['better', 'small-regression'].includes(scenario) ? candidate : original);
      // Each scenario starts from a fresh independent job fixture.
      await rm(path.join(folder, 'result.json'), { force: true });
    }
  } finally {
    assert.ok(path.resolve(root).startsWith(`${artifacts}${path.sep}`));
    await rm(root, { recursive: true, force: true });
  }
});
