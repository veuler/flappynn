import { spawn } from 'node:child_process';
import { randomUUID, createHash } from 'node:crypto';
import { mkdir, readFile, copyFile, writeFile, rename, access } from 'node:fs/promises';
import path from 'node:path';
import { HARD_DIFFICULTY_RULES } from '../src/js/config.js';
import { modelProfile } from '../src/js/model-profiles.js';
import { validateRecordingDataset } from '../src/js/recording-dataset.js';
import { validateMetadata } from '../src/js/ai-controller.js';
import { evaluatePromotion, PROMOTION_POLICY } from '../src/js/promotion-policy.js';

const json = (response, status, value) => response.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }).end(JSON.stringify(value));
export function sameModelBehavior(a, b) {
  const fields = ['version', 'architecture', 'activation', 'outputActivation', 'w1', 'b1', 'w2', 'b2', 'threshold',
    'inputNames', 'normalization', 'gameConfig', 'sampleIntervalMs', 'minimumJumpIntervalMs'];
  return fields.every(key => Object.hasOwn(a, key) && Object.hasOwn(b, key) && JSON.stringify(a[key]) === JSON.stringify(b[key]));
}
export function modelPaths(root, difficulty = 'normal', profileId = 'speed-516') {
  const profile = modelProfile(profileId);
  if (!['normal', 'hard'].includes(difficulty)) throw new Error('Unknown training difficulty.');
  const suffix = profile.id === 'lookahead-616' ? '616' : '516';
  const base = path.join(root, 'src', 'models', profile.id, ...(difficulty === 'hard' ? ['hard'] : []));
  return { model: path.join(root, 'artifacts', `model-${suffix}${difficulty === 'hard' ? '-hard' : ''}.json`),
    css: path.join(base, 'model.css'), metadata: path.join(base, 'model.meta.json') };
}
export function sameOrigin(request) {
  return ['127.0.0.1', 'localhost'].includes((request.headers.host || '').split(':')[0]) &&
    request.headers.origin === `http://${request.headers.host}`;
}
async function body(request) {
  let text = '';
  for await (const chunk of request) {
    text += chunk;
    if (Buffer.byteLength(text) > 16*1024*1024) throw new Error('The request exceeds 16 MiB.');
  }
  return JSON.parse(text || '{}');
}
function python(executable, args, cwd) {
  return new Promise((resolve, reject) => {
    const process = spawn(executable, args, { cwd, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    let error = '';
    process.stdout.resume();
    process.stderr.on('data', (chunk) => { error = (error + chunk).slice(-4000); });
    process.on('error', reject);
    process.on('exit', (code) => code === 0 ? resolve() : reject(new Error(error || `Python exit code: ${code}`)));
  });
}
export function createLearningService(root, profileId = 'speed-516', { spawnWorker = spawn } = {}) {
  const profile = modelProfile(profileId);
  const slots = (difficulty = 'normal') => modelPaths(root, difficulty, profile.id);
  const matchesProfile = meta => meta.version === profile.version &&
    JSON.stringify(meta.architecture) === JSON.stringify(profile.architecture) && meta.modelProfile === profile.id;
  const executable = path.join(root, '.venv', 'Scripts', 'python.exe');
  const learningRoot = path.join(root, 'src', profile.learning);
  let job = null;
  let child = null;
  let validating = false;
  let launching = false;
  let validationOwner = null;
  let validationExpires = 0;
  let checkingUnchanged = false;
  const active = () => job && ['starting', 'running', 'evaluating', 'awaiting-validation', 'validating'].includes(job.state);
  async function baselineSlot(difficulty) {
    const target = slots(difficulty);
    if (difficulty === 'hard') {
      try { await access(target.metadata); }
      catch (error) { if (error.code === 'ENOENT') return slots(); throw error; }
    }
    return target;
  }
  async function checkComparison(folder, baseline, candidate) {
    const report = JSON.parse(await readFile(path.join(folder, 'report.json'), 'utf8'));
    const meta = JSON.parse(await readFile(path.join(folder, 'model.meta.json'), 'utf8'));
    const baselineMeta = JSON.parse(await readFile(path.join(folder, 'baseline', 'model.meta.json'), 'utf8'));
    if (!matchesProfile(meta) || !matchesProfile(baselineMeta)) throw new Error('Candidate belongs to a different model architecture.');
    const difficulty = job.difficulty || 'normal';
    if ((report.difficulty || 'normal') !== difficulty || (meta.trainingDifficulty || 'normal') !== difficulty) throw new Error('The candidate training difficulty does not match.');
    if (difficulty === 'hard' && !Object.entries(HARD_DIFFICULTY_RULES).every(([key, value]) => meta.difficultyConfig?.[key] === value)) throw new Error('The hard model speed rules do not match.');
    const digest = bytes => createHash('sha256').update(bytes).digest('hex');
    if (meta.parityVerified !== true || meta.threshold !== job.threshold ||
        digest(await readFile(path.join(folder, 'candidate.json'))) !== meta.modelSha256 ||
        digest(await readFile(path.join(folder, 'model.css'))) !== meta.cssSha256) throw new Error('The candidate files changed after validation.');
    const mean = (part, expectedHash) => {
      if (part?.modelSha256 !== expectedHash || part.threshold !== job.threshold || (part.difficulty || 'normal') !== difficulty || part.maximumSimulationSeconds !== 180 || part.runs?.length !== 20) throw new Error('The benchmark model, threshold, difficulty, or duration does not match.');
      for (let i = 0; i < 20; i++) {
        const row = part.runs[i];
        if (row.seed !== i + 1 || !Number.isInteger(row.score) || row.score < 0 || !Number.isFinite(row.survivalSeconds) || row.survivalSeconds < 0 || row.survivalSeconds > 180.01) throw new Error('Invalid benchmark run.');
      }
      return part.runs.reduce((sum, row) => sum + row.score, 0) / 20;
    };
    const before = mean(baseline, baselineMeta.modelSha256);
    const after = mean(candidate, meta.modelSha256);
    // Manual acceptance may override performance policy, never numerical validation.
    for (const [browser, offline] of [[baseline, report.baselineTest], [candidate, report.candidateTest]]) {
      if (browser.runs.some((run, i) => run.score !== offline?.runs?.[i]?.score || !Number.isFinite(offline.runs[i].survivalSeconds) || Math.abs(run.survivalSeconds - offline.runs[i].survivalSeconds) > 0.05)) throw new Error('Python simulation and CSS gameplay disagree; the saved model is preserved.');
    }
    return { meta, baselineMeta, before, after };
  }
  async function installCandidate(folder, meta, baselineMeta) {
    const difficulty = job.difficulty || 'normal';
    const source = await baselineSlot(difficulty);
    const target = slots(difficulty);
    const current = JSON.parse(await readFile(source.metadata, 'utf8'));
    const currentModelDigest = createHash('sha256').update(await readFile(source.model)).digest('hex');
    if (current.modelSha256 !== baselineMeta.modelSha256 || currentModelDigest !== baselineMeta.modelSha256) throw new Error('The active model changed during training; the stale candidate was not loaded.');
    // The job's baseline remains the rollback snapshot. Existing flights keep their model.
    await mkdir(path.dirname(target.metadata), { recursive: true });
    for (const [source, destination] of [['candidate.json', target.model], ['model.css', target.css], ['model.meta.json', target.metadata]]) {
      await copyFile(path.join(folder, source), `${destination}.next`);
      await rename(`${destination}.next`, destination);
    }
  }
  async function snapshot() {
    if (!job) {
      try {
        const saved = JSON.parse(await readFile(path.join(learningRoot, 'latest.json'), 'utf8'));
        if (/^[a-f0-9-]{36}$/.test(saved.id || '')) job = saved;
      } catch { /* first run */ }
    }
    if (job && ['starting', 'running', 'evaluating'].includes(job.state)) {
      try { job = { ...job, ...JSON.parse(await readFile(path.join(learningRoot, job.id, 'status.json'), 'utf8')) }; } catch { /* worker is starting */ }
    }
    if (job) {
      if (['awaiting-validation', 'validating'].includes(job.state)) {
        try { job = { ...job, ...JSON.parse(await readFile(path.join(learningRoot, job.id, 'validation-progress.json'), 'utf8')) }; } catch { /* not started */ }
      }
      try { job = { ...job, ...JSON.parse(await readFile(path.join(learningRoot, job.id, 'result.json'), 'utf8')) }; } catch { /* no final result yet */ }
      const folder = path.join(learningRoot, job.id);
      if (['awaiting-validation', 'validating', 'kept', 'promoted', 'failed'].includes(job.state)) {
        try {
          const metadata = JSON.parse(await readFile(path.join(folder, 'model.meta.json'), 'utf8'));
          job.candidateCssVerified = metadata.parityVerified === true;
        } catch { job.candidateCssVerified = false; }
      }
      if (job.state === 'validating' && job.candidateCssVerified && !validating && !checkingUnchanged) {
        checkingUnchanged = true;
        try {
          const currentSlot = await baselineSlot(job.difficulty || 'normal');
          const [candidateBytes, baselineBytes, metadataBytes, baselineMetaBytes, currentBytes, currentMetaBytes, cssBytes] = await Promise.all([
            readFile(path.join(folder, 'candidate.json')), readFile(path.join(folder, 'baseline', 'model.json')),
            readFile(path.join(folder, 'model.meta.json')), readFile(path.join(folder, 'baseline', 'model.meta.json')),
            readFile(currentSlot.model), readFile(currentSlot.metadata), readFile(path.join(folder, 'model.css')),
          ]);
          const metadata = JSON.parse(metadataBytes);
          const baselineMeta = JSON.parse(baselineMetaBytes);
          const currentMeta = JSON.parse(currentMetaBytes);
          const digest = bytes => createHash('sha256').update(bytes).digest('hex');
          if (metadata.parityVerified === true && digest(candidateBytes) === metadata.modelSha256 && digest(cssBytes) === metadata.cssSha256 &&
              (job.difficulty !== 'hard' || Object.entries(HARD_DIFFICULTY_RULES).every(([key, value]) => metadata.difficultyConfig?.[key] === value)) &&
              digest(baselineBytes) === baselineMeta.modelSha256 && digest(currentBytes) === baselineMeta.modelSha256 &&
              currentMeta.modelSha256 === baselineMeta.modelSha256 && sameModelBehavior(JSON.parse(candidateBytes), JSON.parse(baselineBytes))) {
            const unchangedJob = { ...job, state: 'kept', comparisonSkipped: 'unchanged-weights', candidateModelSha256: metadata.modelSha256 };
            await writeFile(path.join(folder, 'result.json'), JSON.stringify(unchangedJob, null, 2));
            job = unchangedJob;
            validationOwner = null;
          }
        } catch { /* A missing or changed file cannot qualify for the shortcut. */ }
        finally { checkingUnchanged = false; }
      }
      if (job.state === 'failed') {
        try {
          await Promise.all(['candidate.json', 'model.meta.json', 'report.json'].map(file => access(path.join(learningRoot, job.id, file))));
          job.canRetryValidation = true;
        } catch { job.canRetryValidation = false; }
      }
    }
    return { job, promotionPolicy: PROMOTION_POLICY };
  }
  return async (request, response, pathname) => {
    if (pathname !== profile.api && !pathname.startsWith(`${profile.api}/`)) return false;
    pathname = `/api/learning${pathname.slice(profile.api.length)}`;
    try {
      if (!['127.0.0.1', 'localhost'].includes((request.headers.host || '').split(':')[0])) {
        json(response, 403, { error: 'Localhost only.' }); return true;
      }
      if (request.method === 'GET' && pathname === '/api/learning') {
        json(response, 200, await snapshot()); return true;
      }
      if (request.method !== 'POST' || !sameOrigin(request) || !request.headers['content-type']?.startsWith('application/json')) {
        json(response, 403, { error: 'A JSON request from the same local origin is required.' }); return true;
      }
      const data = await body(request);
      if (profile.id === 'lookahead-616' && pathname === '/api/learning/start') {
        json(response, 410, { error: 'Parameter-search training has been removed from this lab. Start PPO training in the RL panel.' });
        return true;
      }
      if (pathname === '/api/learning/start' || pathname === '/api/learning/human') {
        const human = pathname === '/api/learning/human';
        const difficulty = data.difficulty ?? 'normal';
        const initialization = human ? data.initialization ?? 'saved' : 'saved';
        if (!['saved', 'random'].includes(initialization)) throw new Error('Unknown human training initialization.');
        slots(difficulty);
        if ((!human && ![300, 600, 900].includes(data.seconds)) || !Number.isFinite(data.threshold) || data.threshold <= 0 || data.threshold >= 1) throw new Error('Invalid duration or threshold.');
        if (human) {
          validateRecordingDataset(data.dataset,profile.id);
          const flights=data.dataset.sessions.filter(flight=>(flight.difficulty || 'normal')===difficulty);
          if (flights.reduce((sum,flight)=>sum+flight.sampleCount,0)<200 || new Set(flights.map(flight=>flight.seed)).size<2) throw new Error('Record at least 200 samples on two different seeds for this difficulty.');
        }
        await snapshot();
        if (active() || launching || validating || checkingUnchanged) { json(response, 409, { error: 'A training session is already running.' }); return true; }
        launching = true;
        try {
        const id = randomUUID();
        const folder = path.join(learningRoot, id);
        const source = await baselineSlot(difficulty);
        const sourceMeta = JSON.parse(await readFile(source.metadata, 'utf8'));
        if (!matchesProfile(sourceMeta)) throw new Error('Baseline belongs to a different model architecture.');
        await mkdir(path.join(folder, 'baseline'), { recursive: true });
        await Promise.all([
          copyFile(source.model, path.join(folder, 'baseline', 'model.json')),
          copyFile(source.css, path.join(folder, 'baseline', 'model.css')),
          copyFile(source.metadata, path.join(folder, 'baseline', 'model.meta.json')),
        ]);
        if(human) await writeFile(path.join(folder,'recordings.json'),JSON.stringify(data.dataset));
        job = { id, state: 'starting', kind: human ? 'human' : 'self-play', modelProfile: profile.id, difficulty, durationSeconds: human ? 0 : data.seconds, threshold: data.threshold, elapsedSeconds: 0, generation: 0, episodes: 0,
          ...(human ? {initialization,sampleCount:data.dataset.sessions.filter(flight=>(flight.difficulty || 'normal')===difficulty).reduce((sum,flight)=>sum+flight.sampleCount,0),maximumEpochs:500} : {}) };
        await writeFile(path.join(learningRoot, 'latest.json'), JSON.stringify(job));
        const workerArgs=human ? ['scripts/human_train.py','--model',path.join(folder,'baseline','model.json'),'--data',path.join(folder,'recordings.json'),
          '--output',folder,'--difficulty',difficulty,'--threshold',String(data.threshold),'--initialization',initialization] : ['scripts/self_train.py', '--model', path.join(folder, 'baseline', 'model.json'),
          '--output', folder, '--seconds', String(data.seconds), '--threshold', String(data.threshold), '--seed', String(Date.now() >>> 0), '--difficulty', difficulty];
        child = spawnWorker(executable, workerArgs,
        { cwd: root, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
        let stderr = '';
        child.stdout.resume();
        child.stderr.on('data', (chunk) => { stderr = (stderr + chunk).slice(-4000); });
        child.on('error', async (error) => {
          if (job?.id === id) {
            job = { ...job, state: 'failed', error: error.message };
            await writeFile(path.join(folder, 'result.json'), JSON.stringify(job)).catch(() => {});
          }
        });
        child.on('exit', async (code) => {
          if (job?.id !== id) return;
          child = null;
          if (code !== 0) {
            job = { ...job, state: 'failed', error: stderr || 'Training stopped.' };
            await writeFile(path.join(folder, 'result.json'), JSON.stringify(job)).catch(() => {});
            return;
          }
          await snapshot();
          job = { ...job, state: 'awaiting-validation' };
        });
        json(response, 202, { job }); return true;
        } finally { launching = false; }
      }
      await snapshot();
      if (!job || data.id !== job.id) throw new Error('The training job ID does not match.');
      const folder = path.join(learningRoot, job.id);
      if (pathname === '/api/learning/promote') {
        if (job.state !== 'kept' || job.comparisonSkipped || validating || launching) throw new Error('Complete the CSS and gameplay comparison before saving the candidate.');
        validating = true;
        try {
          const comparison = JSON.parse(await readFile(path.join(folder, 'browser-gameplay.json'), 'utf8'));
          const { meta, baselineMeta, before, after } = await checkComparison(folder, comparison.baseline, comparison.candidate);
          validateMetadata(meta, profile.id);
          validateMetadata(baselineMeta, profile.id);
          const digest = bytes => createHash('sha256').update(bytes).digest('hex');
          if (digest(await readFile(path.join(folder, 'baseline', 'model.json'))) !== baselineMeta.modelSha256 ||
              digest(await readFile(path.join(folder, 'baseline', 'model.css'))) !== baselineMeta.cssSha256) throw new Error('The previous model backup changed; the saved model is preserved.');
          await installCandidate(folder, meta, baselineMeta);
          job = { ...job, state: 'promoted', promotionMode: 'manual', baselineMeanScore: before, candidateMeanScore: after, candidateModelSha256: meta.modelSha256 };
          await writeFile(path.join(folder, 'browser-gameplay.json'), JSON.stringify({ ...comparison, promoted: true, automaticPromotion: comparison.promoted, promotionMode: 'manual' }, null, 2));
          await writeFile(path.join(folder, 'result.json'), JSON.stringify(job, null, 2));
          json(response, 200, { job }); return true;
        } finally { validating = false; }
      }
      if (pathname === '/api/learning/retry') {
        if (job.state !== 'failed') throw new Error('Only failed validation can be retried.');
        const metadata = JSON.parse(await readFile(path.join(folder, 'model.meta.json'), 'utf8'));
        await readFile(path.join(folder, 'candidate.json'));
        await readFile(path.join(folder, 'report.json'));
        await rename(path.join(folder, 'result.json'), path.join(folder, `failure-${Date.now()}.json`));
        job = { ...job, state: metadata.parityVerified ? 'validating' : 'awaiting-validation', error: null,
          validationProgress: { phase: metadata.parityVerified ? 'baseline' : 'css', completed: 0, phaseCompleted: 0, total: 40, currentSeed: null } };
        validationOwner = null;
        await writeFile(path.join(folder, 'validation-progress.json'), JSON.stringify({ state: job.state, validationProgress: job.validationProgress }));
        await writeFile(path.join(learningRoot, 'latest.json'), JSON.stringify(job));
        json(response, 200, { job }); return true;
      }
      if (pathname === '/api/learning/claim') {
        if (!['awaiting-validation', 'validating'].includes(job.state)) throw new Error('The model is not ready for validation.');
        if (validationOwner && Date.now() < validationExpires) {
          json(response, 200, { claimed: false }); return true;
        }
        validationOwner = randomUUID();
        validationExpires = Date.now() + 120000;
        json(response, 200, { claimed: true, token: validationOwner, job }); return true;
      }
      if (['/api/learning/parity', '/api/learning/progress', '/api/learning/result', '/api/learning/failure'].includes(pathname) && (!validationOwner || data.token !== validationOwner)) {
        throw new Error('This validation belongs to another tab.');
      }
      if (pathname === '/api/learning/progress') {
        if (!['awaiting-validation', 'validating'].includes(job.state)) throw new Error('Validation is already finished.');
        const { phase, completed, currentSeed = null } = data;
        if (!['css', 'baseline', 'candidate', 'finalizing'].includes(phase) || !Number.isInteger(completed) || completed < 0 || completed > 20 ||
            (currentSeed !== null && (!Number.isInteger(currentSeed) || currentSeed < 1 || currentSeed > 20))) throw new Error('Invalid validation progress.');
        validationExpires = Date.now() + 120000;
        job.validationProgress = { phase, phaseCompleted: completed, currentSeed, total: 40,
          completed: phase === 'finalizing' ? 40 : phase === 'candidate' ? 20 + completed : phase === 'css' ? 0 : completed };
        await writeFile(path.join(folder, 'validation-progress.json'), JSON.stringify({ state: job.state, validationProgress: job.validationProgress }));
        json(response, 200, { job }); return true;
      }
      if (pathname === '/api/learning/stop') {
        if (['starting', 'running'].includes(job.state)) await writeFile(path.join(folder, 'stop'), 'stop');
        json(response, 200, { job }); return true;
      }
      if (pathname === '/api/learning/parity') {
        if (job.state !== 'awaiting-validation' || validating) throw new Error('The model is not ready for validation.');
        validating = true;
        try {
          await writeFile(path.join(folder, 'browser-parity.json'), JSON.stringify(data.report));
          await python(executable, ['scripts/verify_parity.py', '--report', path.join(folder, 'browser-parity.json'),
            '--fixture', path.join(folder, 'parity-inputs.json'), '--css', path.join(folder, 'model.css'),
            '--metadata', path.join(folder, 'model.meta.json')], root);
          job = { ...job, state: 'validating' };
          await writeFile(path.join(folder, 'validation-progress.json'), JSON.stringify({ state: 'validating', validationProgress: job.validationProgress }));
          json(response, 200, { job }); return true;
        } finally { validating = false; }
      }
      if (pathname === '/api/learning/result') {
        if (job.state !== 'validating' || validating) throw new Error('CSS validation is required.');
        validating = true;
        try {
          const { meta, baselineMeta, before, after } = await checkComparison(folder, data.baseline, data.candidate);
          const automaticAcceptance = evaluatePromotion(data.baseline.runs, data.candidate.runs);
          const promoted = automaticAcceptance.accepted;
          await writeFile(path.join(folder, 'browser-gameplay.json'), JSON.stringify({ ...data, promoted, automaticAcceptance }, null, 2));
          if (promoted) await installCandidate(folder, meta, baselineMeta);
          job = { ...job, state: promoted ? 'promoted' : 'kept', automaticAcceptance, baselineMeanScore: before, candidateMeanScore: after,
            candidateModelSha256: meta.modelSha256 };
          validationOwner = null;
          await writeFile(path.join(folder, 'result.json'), JSON.stringify(job, null, 2));
          json(response, 200, { job }); return true;
        } finally { validating = false; }
      }
      if (pathname === '/api/learning/failure') {
        if (!['awaiting-validation', 'validating'].includes(job.state)) throw new Error('Validation is already finished.');
        job = { ...job, state: 'failed', error: String(data.error || 'Validation could not be completed.').slice(0, 500) };
        validationOwner = null;
        await writeFile(path.join(folder, 'result.json'), JSON.stringify(job));
        json(response, 200, { job }); return true;
      }
      json(response, 404, { error: 'Unknown training operation.' });
    } catch (error) { json(response, 400, { error: error.message }); }
    return true;
  };
}
