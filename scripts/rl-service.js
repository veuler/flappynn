import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { access, copyFile, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { validateMetadata } from '../src/js/ai-controller.js';
import { CONFIG, HARD_DIFFICULTY_RULES } from '../src/js/config.js';
import { modelProfile } from '../src/js/model-profiles.js';

const json = (response, status, value) => response.writeHead(status, {
  'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
}).end(JSON.stringify(value));
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const ACTIVE_STATES = new Set(['starting', 'running', 'paused', 'evaluating']);
const STATUS_FIELDS = ['state', 'durationSeconds', 'elapsedSeconds', 'totalSteps', 'episodes', 'updates', 'meanScore',
  'bestScore', 'bestScoreComplete', 'meanReward', 'entropy', 'actorWeightDelta', 'policyVersion', 'history', 'error'];

export function sameRlOrigin(request) {
  return ['127.0.0.1', 'localhost'].includes((request.headers.host || '').split(':')[0]) &&
    request.headers.origin === `http://${request.headers.host}`;
}

async function readBody(request) {
  let text = '';
  for await (const chunk of request) {
    text += chunk;
    if (Buffer.byteLength(text) > 4 * 1024 * 1024) throw new Error('The request exceeds 4 MiB.');
  }
  return JSON.parse(text || '{}');
}

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

function runPython(executable, args, cwd, spawnProcess) {
  return new Promise((resolve, reject) => {
    const process = spawnProcess(executable, args, { cwd, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
    let stderr = '';
    process.stdout.resume();
    process.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
    process.once('error', reject);
    process.once('exit', code => code === 0 ? resolve() : reject(new Error(stderr || `Python exit code: ${code}`)));
  });
}

export function createRlService(root, { spawnWorker = spawn, spawnVerifier = spawn } = {}) {
  root = path.resolve(root);
  const learningRoot = path.join(root, 'src', 'learning-rl-616');
  const python = path.join(root, '.venv', 'Scripts', 'python.exe');
  const profile = modelProfile('lookahead-616');
  let job = null;
  let child = null;
  let launching = false;
  let validating = false;
  let saving = false;

  const folderFor = id => path.join(learningRoot, id);
  const savedPaths = difficulty => ({
    model: path.join(root, 'artifacts', `model-rl-616-${difficulty}.json`),
    css: path.join(root, 'src', 'models', 'rl-616', difficulty, 'model.css'),
    metadata: path.join(root, 'src', 'models', 'rl-616', difficulty, 'model.meta.json'),
  });
  const jobActive = () => job && ACTIVE_STATES.has(job.state);

  async function snapshot() {
    if (!job) {
      try {
        const latest = JSON.parse(await readFile(path.join(learningRoot, 'latest.json'), 'utf8'));
        if (UUID.test(latest.id || '')) job = latest;
      } catch { /* No RL job has been started yet. */ }
    }
    if (job) {
      const folder = folderFor(job.id);
      try {
        const status = JSON.parse(await readFile(path.join(folder, 'status.json'), 'utf8'));
        job = { ...job, ...Object.fromEntries(STATUS_FIELDS.filter(key => Object.hasOwn(status, key)).map(key => [key, status[key]])) };
      } catch { /* The worker may not have published its first status yet. */ }
      try {
        const result = JSON.parse(await readFile(path.join(folder, 'result.json'), 'utf8'));
        job = { ...job, ...result };
      } catch { /* No final result yet. */ }
      if (ACTIVE_STATES.has(job.state) && !child) {
        // A PID is saved before returning from start so a service restart can distinguish
        // a live worker from a stale status file. Signal 0 works for Windows and POSIX.
        let alive = false;
        if (Number.isInteger(job.workerPid) && job.workerPid > 0) {
          try { process.kill(job.workerPid, 0); alive = true; } catch { alive = false; }
        }
        if (!alive) {
          job = { ...job, state: 'failed', error: 'The RL worker ended while the service was offline.' };
          await writeFile(path.join(folder, 'result.json'), JSON.stringify(job, null, 2)).catch(() => {});
          await writeFile(path.join(learningRoot, 'latest.json'), JSON.stringify(job)).catch(() => {});
        }
      }
      try {
        const metadata = JSON.parse(await readFile(path.join(folder, 'model.meta.json'), 'utf8'));
        job = { ...job, candidateCssVerified: metadata.parityVerified === true };
      } catch { job = { ...job, candidateCssVerified: false }; }
    }
    const saved = {};
    for (const difficulty of ['normal', 'hard']) {
      try {
        const paths = savedPaths(difficulty);
        const metadata = JSON.parse(await readFile(paths.metadata, 'utf8'));
        await Promise.all([access(paths.model), access(paths.css)]);
        saved[difficulty] = metadata.parityVerified === true && metadata.modelProfile === 'lookahead-616' && metadata.dataKind === 'reinforcement-learning-ppo';
      }
      catch { saved[difficulty] = false; }
    }
    let canResume = false;
    if (job) {
      try { await access(path.join(folderFor(job.id), 'training-checkpoint.pt')); canResume = true; } catch { /* no checkpoint */ }
    }
    return { job, saved, canResume };
  }

  async function writeLatest(next) {
    job = next;
    await mkdir(learningRoot, { recursive: true });
    await writeFile(path.join(learningRoot, 'latest.json'), JSON.stringify(job, null, 2));
  }

  async function installCandidate(folder, difficulty, meta) {
    const target = savedPaths(difficulty);
    const backup = path.join(folder, 'previous-slot');
    await mkdir(backup, { recursive: true });
    const backupState = {};
    for (const key of ['model', 'css', 'metadata']) {
      try { await copyFile(target[key], path.join(backup, path.basename(target[key]))); backupState[key] = true; }
      catch (error) { if (error.code !== 'ENOENT') throw error; backupState[key] = false; }
    }
    await writeFile(path.join(backup, 'slot.json'), JSON.stringify({ difficulty, files: backupState }, null, 2));
    await mkdir(path.dirname(target.model), { recursive: true });
    await mkdir(path.dirname(target.css), { recursive: true });
    const replacements = [['candidate.json', target.model], ['model.css', target.css], ['model.meta.json', target.metadata]];
    const committed = [];
    try {
      for (const [sourceName, destination] of replacements) {
        const temporary = `${destination}.next-${job.id}`;
        await copyFile(path.join(folder, sourceName), temporary);
        await rename(temporary, destination);
        committed.push(destination);
      }
    } catch (error) {
      for (const destination of committed.reverse()) {
        const key = replacements.find(([, pathName]) => pathName === destination)?.[0] === 'candidate.json' ? 'model' :
          destination === target.css ? 'css' : 'metadata';
        if (backupState[key]) {
          const restore = `${destination}.restore-${job.id}`;
          await copyFile(path.join(backup, path.basename(destination)), restore).then(() => rename(restore, destination)).catch(() => {});
        } else await rm(destination, { force: true }).catch(() => {});
      }
      throw error;
    }
    return { backup: path.relative(root, backup), modelSha256: meta.modelSha256, cssSha256: meta.cssSha256 };
  }

  return async (request, response, pathname) => {
    if (pathname !== '/api/rl' && !pathname.startsWith('/api/rl/')) return false;
    try {
      if (!['127.0.0.1', 'localhost'].includes((request.headers.host || '').split(':')[0])) {
        json(response, 403, { error: 'Localhost only.' }); return true;
      }
      if (request.method === 'GET' && pathname === '/api/rl') {
        json(response, 200, await snapshot()); return true;
      }
      if (request.method === 'GET' && pathname === '/api/rl/replay') {
        const id = new URL(request.url, `http://${request.headers.host}`).searchParams.get('id') || '';
        if (!UUID.test(id)) throw new Error('A valid job ID is required.');
        let replay;
        try {
          const replayPath = path.join(folderFor(id), 'replay.json');
          const info = await stat(replayPath);
          if (info.size > 8 * 1024 * 1024) throw new Error('The replay exceeds 8 MiB.');
          replay = JSON.parse(await readFile(replayPath, 'utf8'));
        }
        catch (error) { if (error.code === 'ENOENT') replay = null; else throw error; }
        json(response, 200, { replay }); return true;
      }
      if (request.method !== 'POST' || !sameRlOrigin(request) || !request.headers['content-type']?.startsWith('application/json')) {
        json(response, 403, { error: 'A JSON request from the same local origin is required.' }); return true;
      }
      const data = await readBody(request);
      if (pathname === '/api/rl/start') {
        const { difficulty = 'normal', mode, numEnvs, seconds, initialization = 'saved' } = data;
        if (!['normal', 'hard'].includes(difficulty) || !['live', 'fast'].includes(mode) || ![1, 32, 64, 128, 256, 512].includes(numEnvs) ||
            ![300, 600, 900, 1800].includes(seconds) || !['random', 'saved', 'checkpoint'].includes(initialization)) {
          throw new Error('Invalid RL training settings.');
        }
        if (mode === 'live' && numEnvs !== 1) throw new Error('Live training uses exactly one environment.');
        await snapshot();
        if (jobActive() || launching || validating || saving) { json(response, 409, { error: 'An RL training job is already active.' }); return true; }
        launching = true;
        try {
          const id = randomUUID();
          const folder = folderFor(id);
          await mkdir(folder, { recursive: true });
          const args = ['scripts/rl_train.py', '--output', folder, '--difficulty', difficulty, '--mode', mode,
            '--num-envs', String(numEnvs), '--seconds', String(seconds), '--initialization', initialization];
          if (initialization === 'saved') {
            const model = savedPaths(difficulty).model;
            await access(model);
            args.push('--model', model);
          } else if (initialization === 'checkpoint') {
            if (!job || job.difficulty !== difficulty || job.mode !== mode || job.numEnvs !== numEnvs) {
              throw new Error('The latest checkpoint settings do not match this run.');
            }
            const checkpoint = path.join(folderFor(job.id), 'training-checkpoint.pt');
            await access(checkpoint);
            args.push('--checkpoint', checkpoint);
          }
          const next = { id, state: 'starting', modelProfile: 'lookahead-616', dataKind: 'reinforcement-learning-ppo',
            difficulty, mode, numEnvs, durationSeconds: seconds, initialization, elapsedSeconds: 0, totalSteps: 0,
            episodes: 0, updates: 0, policyVersion: 1, startedAt: new Date().toISOString() };
          await writeLatest(next);
          child = spawnWorker(python, args, { cwd: root, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
          const launchedChild = child;
          if (Number.isInteger(child.pid)) {
            job = { ...job, workerPid: child.pid };
            await writeLatest(job);
          }
          let stderr = '';
          child.stdout?.resume?.();
          child.stderr?.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
          child.once('error', async error => {
            if (job?.id !== id) return;
            child = null;
            job = { ...job, state: 'failed', error: error.message };
            await writeFile(path.join(folder, 'result.json'), JSON.stringify(job, null, 2)).catch(() => {});
            await writeLatest(job).catch(() => {});
          });
          child.once('exit', async code => {
            if (child === launchedChild) child = null;
            if (job?.id !== id) return;
            if (code !== 0) job = { ...job, state: 'failed', error: stderr || `RL worker exited with code ${code}.` };
            else {
              try {
                const status = JSON.parse(await readFile(path.join(folder, 'status.json'), 'utf8'));
                job = { ...job, ...Object.fromEntries(STATUS_FIELDS.filter(key => Object.hasOwn(status, key)).map(key => [key, status[key]])) };
              } catch { /* Keep the last published progress. */ }
              if (ACTIVE_STATES.has(job.state)) job = { ...job, state: 'awaiting-validation' };
            }
            await writeFile(path.join(folder, 'result.json'), JSON.stringify(job, null, 2)).catch(() => {});
            await writeLatest(job).catch(() => {});
          });
          json(response, 202, { job }); return true;
        } finally { launching = false; }
      }
      await snapshot();
      if (!job || data.id !== job.id) throw new Error('The RL job ID does not match.');
      const folder = folderFor(job.id);
      if (pathname === '/api/rl/pause') {
        if (!jobActive()) throw new Error('Only an active RL job can be paused.');
        await writeFile(path.join(folder, 'pause'), 'pause');
        json(response, 200, { job }); return true;
      }
      if (pathname === '/api/rl/resume') {
        if (job.state !== 'paused') throw new Error('Only a paused RL job can be resumed.');
        await rm(path.join(folder, 'pause'), { force: true });
        json(response, 200, { job }); return true;
      }
      if (pathname === '/api/rl/stop') {
        if (!jobActive()) throw new Error('Only an active RL job can be stopped.');
        await writeFile(path.join(folder, 'stop'), 'stop');
        json(response, 200, { job }); return true;
      }
      if (pathname === '/api/rl/parity') {
        if (job.state !== 'awaiting-validation' || validating) throw new Error('The RL candidate is not ready for parity validation.');
        validating = true;
        try {
          const reportPath = path.join(folder, 'browser-parity.json');
          await rm(path.join(folder, 'parity-receipt.json'), { force: true });
          await writeFile(reportPath, JSON.stringify(data.report));
          await runPython(python, ['scripts/verify_parity.py', '--report', reportPath,
            '--fixture', path.join(folder, 'parity-inputs.json'), '--css', path.join(folder, 'model.css'),
            '--metadata', path.join(folder, 'model.meta.json')], root, spawnVerifier);
          const metadata = JSON.parse(await readFile(path.join(folder, 'model.meta.json'), 'utf8'));
          const receipt = {};
          for (const [key, file] of Object.entries({ report: reportPath, fixture: path.join(folder, 'parity-inputs.json'),
            css: path.join(folder, 'model.css'), model: path.join(folder, 'candidate.json'), metadata: path.join(folder, 'model.meta.json') })) {
            receipt[key] = sha256(await readFile(file));
          }
          await writeFile(path.join(folder, 'parity-receipt.json'), JSON.stringify(receipt, null, 2));
          job = { ...job, candidateCssVerified: metadata.parityVerified === true };
          await writeLatest(job);
          json(response, 200, { job }); return true;
        } finally { validating = false; }
      }
      if (pathname === '/api/rl/save') {
        if (job.state !== 'awaiting-validation' || saving || validating) throw new Error('The RL candidate must pass CSS parity before saving.');
        saving = true;
        try {
          const candidatePath = path.join(folder, 'candidate.json');
          const cssPath = path.join(folder, 'model.css');
          const metadataPath = path.join(folder, 'model.meta.json');
          const [candidateBytes, cssBytes, metadataBytes] = await Promise.all([readFile(candidatePath), readFile(cssPath), readFile(metadataPath)]);
          const metadata = JSON.parse(metadataBytes);
          const candidate = JSON.parse(candidateBytes);
          const training = candidate.training;
          const elapsedSeconds = job.elapsedSeconds;
          const receipt = JSON.parse(await readFile(path.join(folder, 'parity-receipt.json'), 'utf8'));
          const currentDigests = {
            report: sha256(await readFile(path.join(folder, 'browser-parity.json'))),
            fixture: sha256(await readFile(path.join(folder, 'parity-inputs.json'))),
            css: sha256(cssBytes), model: sha256(candidateBytes), metadata: sha256(metadataBytes),
          };
          validateMetadata(metadata, profile.id);
          if (Object.keys(currentDigests).some(key => receipt[key] !== currentDigests[key]) ||
              metadata.dataKind !== 'reinforcement-learning-ppo' || metadata.version !== profile.version ||
              JSON.stringify(metadata.architecture) !== JSON.stringify(profile.architecture) || metadata.modelProfile !== 'lookahead-616' ||
              metadata.trainingDifficulty !== job.difficulty || metadata.trainingMode !== job.mode ||
              metadata.numEnvs !== job.numEnvs || !Number.isFinite(metadata.trainingSeconds) || metadata.trainingSeconds < 0 ||
              !Number.isFinite(elapsedSeconds) || metadata.trainingSeconds !== elapsedSeconds ||
              training?.dataKind !== 'reinforcement-learning-ppo' || training.difficulty !== job.difficulty ||
              training.mode !== job.mode || training.numEnvs !== job.numEnvs || training.trainingSeconds !== elapsedSeconds ||
              (job.difficulty === 'hard' && !Object.entries(HARD_DIFFICULTY_RULES).every(([key, value]) => metadata.difficultyConfig?.[key] === value)) ||
              metadata.sampleIntervalMs !== CONFIG.decisionIntervalMs || metadata.minimumJumpIntervalMs !== CONFIG.minimumJumpIntervalMs ||
              metadata.parityVerified !== true || metadata.modelSha256 !== sha256(candidateBytes) || metadata.cssSha256 !== sha256(cssBytes)) {
            throw new Error('The RL candidate metadata, settings, or file hashes do not match this job.');
          }
          const result = await installCandidate(folder, job.difficulty, metadata);
          job = { ...job, state: 'saved', candidateModelSha256: metadata.modelSha256, savedAt: new Date().toISOString(), ...result };
          await writeFile(path.join(folder, 'result.json'), JSON.stringify(job, null, 2));
          await writeLatest(job);
          json(response, 200, { job }); return true;
        } finally { saving = false; }
      }
      json(response, 404, { error: 'Unknown RL operation.' });
    } catch (error) { json(response, 400, { error: error.message }); }
    return true;
  };
}
