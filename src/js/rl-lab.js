import { CONFIG } from './config.js';
import { Game } from './game.js';
import { Renderer } from './renderer.js';
import { AiController } from './ai-controller.js';
import { BrainVisualizer } from './brain-visualizer.js';
import { TrainingRecorder } from './training-recorder.js';
import { SessionController } from './session-controller.js';
import { modelProfile } from './model-profiles.js';
import { cssNumberReader } from './css-numbers.js';
import { ReplayPool } from './replay-pool.js';


export function bindRlLab({ root = document, prefix = '', onSaved = () => {} } = {}) {
  const $ = (id) => root.querySelector(`#${prefix}${id}`);
  const profile = modelProfile('lookahead-616');
  const canvas = $('game');
  const renderer = new Renderer(canvas, CONFIG);
  const idleGame = new Game();
  const brain = new BrainVisualizer($('brain-network'), $('network-status'), profile.architecture);
  const neuronGrid = $('neuron-grid');
  const neuronBars = [];
  for (let i = 0; i < 16; i += 1) {
    const row = document.createElement('div');
    row.className = 'rl-neuron';
    const label = document.createElement('span'); label.textContent = `H${i + 1}`;
    const track = document.createElement('i'); track.className = 'rl-neuron-track';
    const bar = document.createElement('i'); track.append(bar);
    const value = document.createElement('code'); value.textContent = '—';
    row.append(label, track, value); neuronGrid.append(row); neuronBars.push({ bar, value });
  }

  let job = null;
  let saved = { normal: false, hard: false };
  let currentReplay = null;
  let queuedReplay = null;
  const replayPool = new ReplayPool();
  let playbackCursor = 0;
  let playbackClock = null;
  let playbackPaused = false;
  let pollTimer = null;
  let polling = false;
  let demoGame = null;
  let demoSession = null;
  let demoAI = null;
  let demoState = null;
  let demoLastTime = null;
  let demoAccumulator = 0;
  let jumpCount = 0;
  let waitCount = 0;
  let countedReplayFrames = new Set();
  let validation = null;
  let candidateCss = '';
  let candidateFixture = null;
  let candidateDigest = '';
  let parityPassed = false;
  let parityFrame = null;
  let lastBrainFrame = '';
  let lastJobId = null;
  let snapshotEpoch = 0;

  const activeStates = new Set(['starting', 'running', 'paused', 'evaluating']);
  const fmt = (value, digits = 1) => Number.isFinite(Number(value)) ? Number(value).toLocaleString('en-US', { maximumFractionDigits: digits }) : '—';
  const clock = (value) => { const seconds = Math.max(0, Math.floor(Number(value) || 0)); return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`; };
  const escapeText = (value) => String(value ?? '');

  async function api(url, options = {}) {
    const response = await fetch(url, { cache: 'no-store', ...options, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers } });
    const text = await response.text();
    let data = null;
    if (text) { try { data = JSON.parse(text); } catch { data = { error: text.slice(0, 800) }; } }
    if (!response.ok) throw new Error(data?.error || data?.message || `RL service returned HTTP ${response.status}.`);
    return data;
  }
  function setMessage(id, message, tone = '') {
    const target = $(id); target.textContent = escapeText(message); target.dataset.tone = tone;
  }
  function setJobState(state) {
    const labels = { offline: 'SERVICE OFFLINE', starting: 'STARTING', running: 'TRAINING', paused: 'PAUSED', evaluating: 'EVALUATING', 'awaiting-validation': 'VALIDATION REQUIRED', failed: 'FAILED', saved: 'SAVED', completed: 'COMPLETED', stopped: 'STOPPED' };
    $('job-state').textContent = labels[state] || 'NO TRAINING JOB';
    $('job-pill').dataset.state = state || 'idle';
  }
  function jobId() { return job?.id; }
  function updateFlightScore(value) {
    const score = Number(value);
    const text = Number.isFinite(score) ? String(Math.max(0, Math.floor(score))) : '0';
    if ($('flight-score').textContent !== text) $('flight-score').textContent = text;
  }
  function setOverlay(title, text, hidden = false) {
    $('game-overlay-title').textContent = title;
    $('game-overlay-text').textContent = text;
    $('game-overlay').hidden = hidden;
  }
  function currentSourceCaption(replay) {
    if (!replay) return 'WAITING';
    return replay.source === 'live-training' ? 'LIVE TRAINING' : 'RANDOM REPLAY';
  }
  function toRenderGame(frame, replay) {
    const pipes = (Array.isArray(frame.pipes) ? frame.pipes : []).map((pipe) => ({ x: Number(pipe.x) || 0, gapCenterY: Number(pipe.gapCenterY) || 320, passed: Boolean(pipe.passed) }));
    return { bird: { y: Number(frame.y) || CONFIG.initialBirdY, velocityY: Number(frame.vy) || 0 }, pipes,
      elapsedSeconds: Number(frame.t) || 0, score: Number(frame.score) || 0, pipeSpeed: Number(frame.pipeSpeed) || CONFIG.pipeSpeed,
      status: frame.status === 'running' ? 'running' : (replay?.complete ? 'gameover' : 'running') };
  }
  function updateBrain(frame, replay, frameIndex = 0) {
    if (!frame) {
      brain.update(null, true, 'ready');
      lastBrainFrame = '';
      $('jump-score').textContent = '—'; $('decision').textContent = 'WAITING';
      $('policy-version').textContent = 'NO FRAME'; $('network-status').textContent = 'WAITING FOR FRAME';
      for (let i = 0; i < 6; i += 1) $(`input-${i}`).textContent = '—';
      neuronBars.forEach(({ bar, value }) => { bar.style.width = '0%'; value.textContent = '—'; });
      return;
    }
    const inputs = Array.isArray(frame.inputs) ? frame.inputs : [];
    const hidden = Array.isArray(frame.hidden) ? frame.hidden : [];
    const probability = Number(frame.probability);
    if (frame.decision === false) {
      brain.update(brain.lastPrediction, true, playbackPaused ? 'paused' : frame.status);
      $('decision').textContent = 'NO DECISION';
      $('network-status').textContent = 'NO DECISION · LAST POLICY VALUES HELD';
      $('jump-count').textContent = fmt(jumpCount, 0); $('wait-count').textContent = fmt(waitCount, 0);
      return;
    }
    const prediction = { inputs, hidden, probability: Number.isFinite(probability) ? probability : 0 };
    const streamFinished = replay && frameIndex === replay.frames.length - 1 &&
      ['awaiting-validation', 'failed', 'saved', 'completed', 'stopped'].includes(job?.state);
    const gameStatus = playbackPaused || streamFinished ? 'paused' : frame.status === 'running' ? 'running' : replay?.complete ? 'gameover' : 'paused';
    const visualFrameKey = replay ? `${replay.episodeId}:${frameIndex}:${gameStatus}` : '';
    if (visualFrameKey !== lastBrainFrame) { brain.update(prediction, true, gameStatus); lastBrainFrame = visualFrameKey; }
    $('jump-score').textContent = Number.isFinite(probability) ? `${(probability * 100).toFixed(1)}%` : '—';
    $('decision').textContent = frame.action === 1 || frame.action === 'jump' ? 'JUMP' : frame.requestedAction && frame.action !== 1 ? 'COOLDOWN' : 'WAIT';
    $('policy-version').textContent = `POLICY V${frame.policyVersion ?? replay?.policyVersion ?? '—'}`;
    $('network-status').textContent = replay?.source === 'live-training' ? 'LIVE · RECORDED VALUES' : 'REPLAY · RECORDED VALUES';
    inputs.slice(0, 6).forEach((value, i) => { $(`input-${i}`).textContent = fmt(value, 3); });
    for (let i = inputs.length; i < 6; i += 1) $(`input-${i}`).textContent = '—';
    neuronBars.forEach(({ bar, value }, i) => {
      const activation = Number(hidden[i]);
      if (!Number.isFinite(activation)) { bar.style.width = '0%'; value.textContent = '—'; return; }
      // This mirrors the existing visualizer's activity scale; it is not a weight display.
      bar.style.width = `${Math.max(0, Math.min(100, activation / 4 * 100))}%`;
      value.textContent = activation.toFixed(2);
    });
    $('jump-count').textContent = fmt(jumpCount, 0); $('wait-count').textContent = fmt(waitCount, 0);
  }
  function countReplayFrames(replay, from, to) {
    for (let index = Math.max(0, from); index <= Math.min(to, replay.frames.length - 1); index += 1) {
      if (countedReplayFrames.has(index)) continue;
      countedReplayFrames.add(index);
      const frame = replay.frames[index];
      if (frame.decision === false) continue;
      if (frame.action === 1 || frame.action === 'jump') jumpCount += 1; else waitCount += 1;
    }
    $('jump-count').textContent = fmt(jumpCount, 0); $('wait-count').textContent = fmt(waitCount, 0);
  }
  function showReplay(replay) {
    canvas.setAttribute('aria-label', 'Recorded training flight');
    currentReplay = replay;
    playbackCursor = 0;
    playbackClock = null;
    updatePlayback.accumulator = 0;
    jumpCount = 0; waitCount = 0; countedReplayFrames = new Set(); lastBrainFrame = '';
    $('flight-title').textContent = `Episode ${replay.episodeId ?? '—'}`;
    $('flight-source').textContent = currentSourceCaption(replay);
    $('source-note').textContent = replay.source === 'live-training'
      ? 'Live training frames are sampled from the one active environment. Each neural value is the exact recorded policy output for that frame.'
      : 'Flights are randomly picked from recent recordings; neural values belong to the policy version recorded in that flight.';
    $('playback-toggle').disabled = !replay.frames?.length;
    playbackPaused = false;
    $('playback-toggle').textContent = 'Pause replay';
    setOverlay('Loading episode…', 'Waiting for the first recorded policy frame.', !replay.complete);
    if (!replay.frames?.length) setOverlay('Episode starting', `Episode ${replay.episodeId ?? ''} has not produced a frame yet.`, false);
  }
  function acceptReplay(next) {
    if (!next || !Array.isArray(next.frames)) return;
    replayPool.add(next);
    if (!currentReplay || !currentReplay.frames?.length) { showReplay(next); return; }
    if (next.episodeId === currentReplay.episodeId) {
      currentReplay = { ...currentReplay, ...next };
      return;
    }
    if (currentReplay.complete && playbackCursor >= currentReplay.frames.length - 1 && !playbackPaused) {
      const selected = replayPool.next(currentReplay, next);
      if (selected) { queuedReplay = null; showReplay(selected); }
    } else if (next.source === 'live-training' || !next.complete) {
      queuedReplay = next;
    }
  }
  function updatePlayback(now) {
    requestAnimationFrame(updatePlayback);
    if (activeDemo()) return;
    const replay = currentReplay;
    if (!replay?.frames?.length) {
      renderer.draw(idleGame);
      updateFlightScore(0);
      brain.draw(now);
      return;
    }
    const frames = replay.frames;
    const priorCursor = playbackCursor;
    const speed = Number($('playback-speed').value) || 1;
    if (playbackClock === null) playbackClock = now;
    const delta = Math.min(0.1, Math.max(0, (now - playbackClock) / 1000));
    playbackClock = now;
    if (!playbackPaused && frames.length > 1) {
      // Retain real samples; the cursor only advances across available recorded frames.
      updatePlayback.accumulator = Math.min(0.5, (updatePlayback.accumulator || 0) + delta * speed);
      while (playbackCursor < frames.length - 1) {
        const frameTime = Number(frames[playbackCursor].t) || 0;
        const nextTime = Number(frames[playbackCursor + 1].t) || frameTime;
        const nextInterval = Math.max(1 / 120, Math.min(0.2, nextTime - frameTime || 1 / 20));
        if (updatePlayback.accumulator < nextInterval) break;
        updatePlayback.accumulator -= nextInterval;
        playbackCursor = Math.min(playbackCursor + 1, frames.length - 1);
      }
    }
    const index = Math.min(playbackCursor, frames.length - 1);
    const frame = frames[index];
    countReplayFrames(replay, countedReplayFrames.size ? priorCursor + 1 : 0, index);
    renderer.draw(toRenderGame(frame, replay));
    updateFlightScore(frame.score);
    updateBrain(frame, replay, index);
    brain.draw(now);
    const duration = Number(frames.at(-1)?.t) || 0;
    const time = Number(frame.t) || 0;
    $('playback-progress').style.width = `${duration > 0 ? Math.min(100, time / duration * 100) : (index / Math.max(1, frames.length - 1) * 100)}%`;
    $('playback-time').textContent = `Seed ${replay.seed} · t ${time.toFixed(2)} s · score ${fmt(frame.score, 0)} · frame ${index + 1}/${frames.length}`;
    const streamEnded = Boolean(job && ['awaiting-validation', 'failed', 'saved', 'completed', 'stopped'].includes(job.state));
    const frameEnded = frame.status === 'gameover' || frame.status === 'truncated';
    const atEnd = index === frames.length - 1 && (replay.complete || streamEnded || frameEnded);
    if (atEnd) {
      if (replay.complete) setOverlay('Episode complete', `Score ${fmt(frame.score, 0)} · seed ${replay.seed ?? '—'} · policy v${frame.policyVersion ?? replay.policyVersion ?? '—'}`, false);
      else if (streamEnded) setOverlay('Training stopped', `Last available frame · episode incomplete · score ${fmt(frame.score, 0)} · seed ${replay.seed ?? '—'}.`, false);
      else if (frameEnded) setOverlay('Episode ended', `Score ${fmt(frame.score, 0)} · seed ${replay.seed ?? '—'} · waiting for the complete replay record.`, false);
      if (!playbackPaused) {
        const next = replayPool.next(replay, queuedReplay);
        if (next) { queuedReplay = null; showReplay(next); }
      }
    } else setOverlay('', '', true);
  }
  function renderJob(data) {
    job = data?.job || null;
    saved = data?.saved || saved;
    const state = job?.state || '';
    if (job?.id !== lastJobId) {
      lastJobId = job?.id ?? null;
      if (job) {
        $('difficulty').value = job.difficulty;
        $('training-mode').value = job.mode;
        if (job.mode === 'fast') $('num-envs').value = String(job.numEnvs);
        if (Array.from($('duration').options).some(option => option.value === String(job.durationSeconds))) {
          $('duration').value = String(job.durationSeconds);
        }
        $('initialization').value = job.initialization;
        $('demo-difficulty').value = job.difficulty;
        $('env-field').hidden = job.mode === 'live';
        $('num-envs').disabled = job.mode === 'live';
        $('mode-help').textContent = job.mode === 'live'
          ? 'Live training runs one environment at real game speed; the displayed flight follows its recorded decisions.'
          : 'Fast training runs parallel environments; the displayed episode is a recording of real training decisions.';
      }
      validation = null; candidateCss = ''; candidateFixture = null; candidateDigest = '';
      currentReplay = null; queuedReplay = null; replayPool.clear(); countedReplayFrames = new Set();
      playbackCursor = 0; playbackClock = null; playbackPaused = false; jumpCount = 0; waitCount = 0; lastBrainFrame = '';
      $('playback-toggle').disabled = true; $('jump-count').textContent = '0'; $('wait-count').textContent = '0';
      $('playback-toggle').textContent = 'Pause replay';
      $('flight-title').textContent = 'Training episode'; $('flight-source').textContent = 'WAITING';
      updateBrain(null);
      setOverlay(job ? 'Waiting for an episode' : 'Ready to train', job ? 'Waiting for the next recorded policy frame.' : 'Start a job to watch real policy decisions.', false);
    }
    parityPassed = job?.candidateCssVerified === true;
    $('saved-status').textContent = `Normal: ${saved.normal ? 'available' : 'not saved'} · Hard: ${saved.hard ? 'available' : 'not saved'}`;
    const checkpointOption = $('initialization').querySelector('option[value="checkpoint"]');
    const expectedMode = $('training-mode').value;
    const expectedEnvs = expectedMode === 'live' ? 1 : Number($('num-envs').value);
    const checkpointFits = data?.canResume === true && job && job.difficulty === $('difficulty').value && job.mode === expectedMode && job.numEnvs === expectedEnvs;
    checkpointOption.disabled = !checkpointFits;
    checkpointOption.title = checkpointFits ? 'Initialize from the matching latest RL checkpoint.' : 'The latest checkpoint is unavailable or its run settings do not match.';
    const savedOption = $('initialization').querySelector('option[value="saved"]');
    savedOption.disabled = !saved[$('difficulty').value];
    savedOption.title = savedOption.disabled ? `No saved ${$('difficulty').value} RL policy is available.` : 'Initialize from this difficulty’s saved RL policy.';
    if (savedOption.disabled && $('initialization').value === 'saved') $('initialization').value = 'random';
    if (checkpointOption.disabled && $('initialization').value === 'checkpoint') $('initialization').value = 'random';
    $('demo-play').disabled = !saved[$('demo-difficulty').value] || Boolean(activeDemo()) || activeStates.has(state);
    setJobState(state);
    const running = ['starting', 'running', 'evaluating'].includes(state);
    const hasJob = Boolean(job);
    const trainingLocked = Boolean(job && activeStates.has(state));
    $('train-start').disabled = trainingLocked || Boolean(activeDemo());
    $('setup-fields').disabled = Boolean(job && activeStates.has(state));
    $('train-pause').disabled = !['running', 'paused'].includes(state);
    $('train-pause').textContent = state === 'paused' ? 'Resume training' : 'Pause training';
    $('train-stop').disabled = !['starting', 'running', 'paused', 'evaluating'].includes(state);
    const stateMessage = {
      '': 'The RL service is ready. Choose a run configuration and start training.',
      starting: 'The training worker is starting. The setup is locked until it reports a state.',
      running: 'Training is running. Pause controls training only; replay playback has its own pause control.',
      paused: 'Training is paused. You can resume it or stop and validate its candidate.',
      evaluating: 'Training ended and the service is evaluating the candidate. Saving stays locked until validation is ready.',
      'awaiting-validation': parityPassed ? 'Candidate passed CSS parity and is ready to save.' : 'Candidate is ready. Run browser CSS parity before saving.',
      failed: job?.error || 'Training failed. The service reported no additional details.',
      saved: 'This candidate passed validation and was saved.',
      completed: 'Training completed. Validate the candidate before saving.',
    };
    if (stateMessage[state]) setMessage('api-status', stateMessage[state], state === 'failed' ? 'error' : '');
    if (activeDemo() && !trainingLocked) setMessage('api-status', 'Close the saved CSS demo to enable training.');
    $('metric-episodes').textContent = fmt(job?.episodes, 0);
    $('metric-steps').textContent = fmt(job?.totalSteps, 0);
    $('metric-score').textContent = fmt(job?.meanScore, 2);
    const bestScore = job?.bestScore;
    const hasBestScore = Number.isInteger(bestScore) && bestScore >= 0;
    const bestScoreComplete = job?.bestScoreComplete !== false;
    $('metric-best-score').textContent = hasBestScore ? `${bestScoreComplete ? '' : '≥ '}${fmt(bestScore, 0)}` : '—';
    $('metric-best-score').parentElement.title = hasBestScore
      ? bestScoreComplete ? 'Highest score across all completed training flights, including resumed checkpoints.'
        : 'At least this score. The older checkpoint did not record its full score history.'
      : job?.id ? 'This run started before best-score tracking was added. New runs record it automatically.'
        : 'Highest score across all completed training flights.';
    $('metric-reward').textContent = fmt(job?.meanReward, 3);
    $('metric-entropy').textContent = fmt(job?.entropy, 3);
    $('metric-delta').textContent = fmt(job?.actorWeightDelta, 5);
    $('metric-version').textContent = `POLICY V${job?.policyVersion ?? '—'}`;
    const duration = Number(job?.durationSeconds) || Number($('duration').value);
    const elapsed = Number(job?.elapsedSeconds) || 0;
    $('job-elapsed').textContent = `${clock(elapsed)} / ${clock(duration)}`;
    $('job-progress').style.width = `${Math.min(100, duration ? elapsed / duration * 100 : 0)}%`;
    const ready = ['awaiting-validation', 'completed'].includes(state);
    $('verify-css').disabled = !ready || state === 'failed' || !job?.id;
    $('save-policy').disabled = !parityPassed || state !== 'awaiting-validation';
    $('validation-badge').textContent = parityPassed ? 'PARITY PASSED' : ready ? 'READY TO VERIFY' : 'LOCKED';
    $('validation-copy').textContent = parityPassed ? (state === 'saved' ? 'The saved RL policy matches the candidate that passed CSS parity.' : 'The service verified the candidate CSS and its exact file hashes. The policy is ready to save.') : ready ? 'The candidate is ready for an isolated browser parity run. A verified byte hash is required before save.' : 'A completed training run must pass browser parity before its CSS policy can be saved.';
    if (!ready && !parityPassed) { validation = null; candidateCss = ''; candidateFixture = null; candidateDigest = ''; setMessage('validation-status', 'Waiting for a completed candidate.'); }
    else if (parityPassed && !validation) setMessage('validation-status', 'Browser parity passed and was verified by the RL service.');
    if (hasJob && state === 'failed' && !job.error) setMessage('api-status', 'Training failed; the service did not provide an error description.', 'error');
  }
  function activeDemo() { return Boolean(demoGame && demoAI); }
  async function refresh() {
    if (polling) return;
    polling = true;
    const epoch = snapshotEpoch;
    try {
      const data = await api('/api/rl');
      if (epoch !== snapshotEpoch) return;
      renderJob(data);
      const snapshotId = job?.id;
      if (snapshotId && ['running', 'paused', 'starting', 'evaluating', 'awaiting-validation', 'saved'].includes(job.state)) {
        const payload = await api(`/api/rl/replay?id=${encodeURIComponent(snapshotId)}`);
        if (epoch === snapshotEpoch && job?.id === snapshotId) acceptReplay(payload?.replay || null);
      }
    } catch (error) {
      setMessage('api-status', `RL service unavailable: ${error.message}`, 'error');
      if (!job) setJobState('offline');
    } finally {
      polling = false;
      const fastLive = job?.mode === 'live' || $('training-mode').value === 'live';
      const delay = job?.state === 'running' && fastLive ? 200 : 1000;
      clearTimeout(pollTimer); pollTimer = setTimeout(refresh, delay);
    }
  }
  async function postAction(path, payload = {}) {
    try { await api(path, { method: 'POST', body: JSON.stringify(payload) }); await refreshNow(); }
    catch (error) { setMessage('api-status', error.message, 'error'); }
  }
  async function refreshNow() {
    if (polling) return;
    polling = true;
    const epoch = snapshotEpoch;
    try {
      const data = await api('/api/rl');
      if (epoch !== snapshotEpoch) return;
      renderJob(data);
      const snapshotId = job?.id;
      if (snapshotId && ['running', 'paused', 'starting', 'evaluating', 'awaiting-validation', 'saved'].includes(job.state)) {
        const payload = await api(`/api/rl/replay?id=${encodeURIComponent(snapshotId)}`);
        if (epoch === snapshotEpoch && job?.id === snapshotId) acceptReplay(payload?.replay || null);
      }
    } catch (error) { setMessage('api-status', error.message, 'error'); }
    finally { polling = false; }
  }
  $('train-start').addEventListener('click', async () => {
    if (activeDemo()) { setMessage('api-status', 'Close the saved CSS demo to enable training.'); return; }
    const mode = $('training-mode').value;
    const numEnvs = mode === 'live' ? 1 : Number($('num-envs').value);
    setMessage('api-status', 'Starting RL training…');
    snapshotEpoch += 1;
    $('train-start').disabled = true;
    try {
      const data = await api('/api/rl/start', { method: 'POST', body: JSON.stringify({ difficulty: $('difficulty').value, mode, numEnvs, seconds: Number($('duration').value), initialization: $('initialization').value }) });
      snapshotEpoch += 1;
      if (data?.job) renderJob(data); else await refreshNow();
      if (data?.job?.id) { currentReplay = null; queuedReplay = null; jumpCount = 0; waitCount = 0; playbackCursor = 0; playbackClock = null; $('playback-toggle').disabled = true; $('flight-title').textContent = 'Training episode'; $('flight-source').textContent = 'WAITING'; $('source-note').textContent = 'Training samples actions from policy probabilities to explore; saved CSS demos use a fixed 0.50 threshold.'; updateBrain(null); $('jump-count').textContent = '0'; $('wait-count').textContent = '0'; setOverlay('Training is starting', 'Waiting for the first real training episode.', false); }
    } catch (error) { $('train-start').disabled = false; setMessage('api-status', error.message, 'error'); }
  });
  $('training-mode').addEventListener('change', () => {
    const live = $('training-mode').value === 'live';
    $('env-field').hidden = live; $('num-envs').disabled = live;
    $('mode-help').textContent = live
      ? 'Live mode runs one environment. The displayed game is the current training episode; actual new frames are polled up to five times per second.'
      : 'Fast training runs parallel environments. Episodes appear as training replays from the policy version that produced them.';
  });
  $('difficulty').addEventListener('change', () => {
    const savedOption = $('initialization').querySelector('option[value="saved"]');
    savedOption.disabled = !saved[$('difficulty').value];
    if (savedOption.disabled && $('initialization').value === 'saved') {
      $('initialization').value = 'random';
      setMessage('api-status', `No saved ${$('difficulty').value} RL policy is available. Using fresh random weights.`, 'error');
    }
    refreshNow();
  });
  $('training-mode').addEventListener('change', refreshNow);
  $('num-envs').addEventListener('change', refreshNow);
  $('duration').addEventListener('change', refreshNow);
  $('initialization').addEventListener('change', refreshNow);
  $('train-pause').addEventListener('click', () => postAction(`/api/rl/${job?.state === 'paused' ? 'resume' : 'pause'}`, { id: jobId() }));
  $('train-stop').addEventListener('click', () => postAction('/api/rl/stop', { id: jobId() }));
  $('playback-toggle').addEventListener('click', () => { playbackPaused = !playbackPaused; $('playback-toggle').textContent = playbackPaused ? 'Resume replay' : 'Pause replay'; });
  $('playback-speed').addEventListener('change', () => { updatePlayback.accumulator = 0; });
  $('demo-difficulty').addEventListener('change', () => { $('demo-play').disabled = !saved[$('demo-difficulty').value] || Boolean(activeDemo()) || Boolean(job && activeStates.has(job.state)); });
  $('demo-play').addEventListener('click', async () => {
    if (job && activeStates.has(job.state)) { $('demo-status').textContent = 'Pause or stop training before starting a saved policy demo.'; return; }
    $('demo-play').disabled = true;
    $('train-start').disabled = true;
    setMessage('api-status', 'Close the saved CSS demo to enable training.');
    const difficulty = $('demo-difficulty').value;
    const base = `./models/rl-616/${difficulty}/`;
    $('demo-status').textContent = `Loading saved ${difficulty} CSS policy…`;
    const sink = document.createElement('div'); sink.className = 'css-neural-model'; sink.setAttribute('aria-hidden', 'true'); document.body.append(sink);
    demoAI = new AiController(sink, base, 'lookahead-616');
    const ok = await demoAI.initialize(base);
    if (!ok) { $('demo-status').textContent = `Policy could not be loaded: ${demoAI.error}`; sink.remove(); demoAI = null; $('demo-play').disabled = !saved[difficulty]; $('train-start').disabled = Boolean(job && activeStates.has(job.state)); return; }
    demoAI.setThreshold(0.5);
    demoGame = new Game({ seed: crypto.getRandomValues(new Uint32Array(1))[0], difficulty });
    demoSession = new SessionController(demoGame, new TrainingRecorder({ profileId: 'lookahead-616' }), demoAI);
    demoState = 'running'; demoSession.start(demoGame.seed, 'ai', difficulty);
    canvas.setAttribute('aria-label', 'Saved CSS policy evaluation');
    currentReplay = null; queuedReplay = null; $('flight-title').textContent = 'Saved CSS policy demo'; $('flight-source').textContent = 'CSS EVALUATION';
    $('source-note').textContent = `This flight runs the saved CSS model at a fixed ${demoAI.threshold.toFixed(2)} threshold. It evaluates the policy and does not train or change weights.`;
    $('demo-status').textContent = `Playing saved ${difficulty} policy · deterministic threshold ${demoAI.threshold.toFixed(2)}.`;
    $('demo-play').disabled = true; $('demo-pause').disabled = false; $('demo-reset').disabled = false;
    $('playback-toggle').disabled = true; setOverlay('', '', true); demoLastTime = null; demoAccumulator = 0;
  });
  $('demo-pause').addEventListener('click', () => {
    if (!demoGame) return;
    if (demoGame.status === 'running') { demoGame.pause(); $('demo-pause').textContent = 'Resume'; $('demo-status').textContent = 'Saved CSS policy demo paused.'; }
    else if (demoGame.status === 'paused') { demoGame.resume(); $('demo-pause').textContent = 'Pause'; $('demo-status').textContent = 'Playing saved CSS policy · evaluation only.'; }
  });
  $('demo-reset').addEventListener('click', () => {
    demoGame?.finish(); demoAI?.style?.remove(); demoAI?.sink?.remove(); demoAI = null; demoSession = null; demoGame = null;
    $('demo-status').textContent = 'Demo closed. Choose your saved Normal or Hard model to watch another flight.';
    $('demo-pause').disabled = true; $('demo-reset').disabled = true; $('demo-play').disabled = !saved[$('demo-difficulty').value] || Boolean(job && activeStates.has(job.state));
    $('train-start').disabled = Boolean(job && activeStates.has(job.state));
    $('demo-pause').textContent = 'Pause'; $('playback-toggle').disabled = !currentReplay?.frames?.length;
    jumpCount = 0; waitCount = 0; $('jump-count').textContent = '0'; $('wait-count').textContent = '0';
    $('flight-title').textContent = 'Training episode'; $('flight-source').textContent = 'WAITING';
    $('source-note').textContent = 'Training replays use probability-sampled actions for exploration. Saved CSS demos use a fixed 0.50 threshold.';
    $('playback-time').textContent = 'Waiting for an episode'; $('playback-progress').style.width = '0%'; updateBrain(null); setOverlay('Ready to train', 'Start a job to watch real policy decisions.', false);
    if (!(job && activeStates.has(job.state))) setMessage('api-status', 'The RL service is ready. Choose a run configuration and start training.');
    if (currentReplay) { playbackClock = null; }
  });
  function demoFrame(timestamp) {
    requestAnimationFrame(demoFrame);
    if (!demoGame || !demoSession) return;
    if (demoLastTime !== null && demoGame.status === 'running') {
      demoAccumulator += Math.min((timestamp - demoLastTime) / 1000, CONFIG.maxFrameSeconds);
      while (demoAccumulator >= CONFIG.fixedStepSeconds && demoGame.status === 'running') { demoSession.step(); demoAccumulator -= CONFIG.fixedStepSeconds; }
    }
    demoLastTime = timestamp;
    renderer.draw(demoGame);
    updateFlightScore(demoGame.score);
    $('playback-time').textContent = `Seed ${demoGame.seed} · t ${demoGame.elapsedSeconds.toFixed(1)} s · score ${demoGame.score}`;
    brain.update(demoAI.lastPrediction, true, demoGame.status);
    const prediction = demoAI.lastPrediction;
    if (prediction) {
      $('jump-score').textContent = `${(prediction.probability * 100).toFixed(1)}%`;
      $('decision').textContent = demoAI.decision;
      $('policy-version').textContent = 'SAVED CSS POLICY';
      $('network-status').textContent = 'LIVE · SAVED CSS POLICY';
      prediction.inputs.forEach((v, i) => { if (i < 6) $(`input-${i}`).textContent = fmt(v, 3); });
      prediction.hidden.forEach((activation, i) => { neuronBars[i].bar.style.width = `${Math.max(0, Math.min(100, activation / 4 * 100))}%`; neuronBars[i].value.textContent = activation.toFixed(2); });
      $('jump-count').textContent = fmt(demoAI.jumpCount, 0); $('wait-count').textContent = fmt(demoAI.waitCount, 0);
    }
    brain.draw(timestamp);
    if (demoGame.status === 'gameover') { $('demo-status').textContent = `Evaluation ended · score ${demoGame.score}. Close this demo to run another episode.`; $('demo-pause').disabled = true; setOverlay('Evaluation complete', `Score ${demoGame.score} · saved ${demoGame.difficulty} CSS model`, false); }
  }
  $('verify-css').addEventListener('click', async () => {
    if (!job?.id) return;
    const candidateJobId = job.id;
    $('verify-css').disabled = true; parityPassed = false; setMessage('validation-status', 'Loading candidate CSS and parity cases…');
    try {
      const folder = `./learning-rl-616/${encodeURIComponent(candidateJobId)}/`;
      const [cssResponse, fixtureResponse] = await Promise.all([
        fetch(`${folder}model.css`, { cache: 'no-store' }),
        fetch(`${folder}parity-inputs.json`, { cache: 'no-store' }),
      ]);
      if (!cssResponse.ok || !fixtureResponse.ok) throw new Error(`Candidate artifact fetch failed (CSS ${cssResponse.status}, parity cases ${fixtureResponse.status}).`);
      candidateCss = await cssResponse.text();
      candidateFixture = await fixtureResponse.json();
      if (typeof candidateCss !== 'string' || !Array.isArray(candidateFixture?.cases) || candidateFixture.cases.length !== 4680) throw new Error('Candidate artifacts are incomplete or do not contain the required 4,680 parity cases.');
      candidateDigest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(candidateCss)))].map(v => v.toString(16).padStart(2, '0')).join('');
      if (candidateFixture.cssSha256 && candidateFixture.cssSha256 !== candidateDigest) throw new Error('CSS hash does not match the parity fixture.');
      if (JSON.stringify(candidateFixture.architecture) !== JSON.stringify([6, 16, 1])) throw new Error('Parity fixture architecture is not 6 → 16 → 1.');
      if (job?.id !== candidateJobId) throw new Error('The active training job changed while its candidate was loading.');
      await runIsolatedParity(candidateJobId);
    } catch (error) { setMessage('validation-status', error.message, 'error'); $('validation-badge').textContent = 'FAILED'; }
    finally { $('verify-css').disabled = !['awaiting-validation', 'completed'].includes(job?.state); }
  });
  function runIsolatedParity(validationJobId) {
    const numberReaderSource = cssNumberReader.toString();
    return new Promise((resolve, reject) => {
      parityFrame?.remove();
      const frame = document.createElement('iframe'); frame.hidden = true; frame.title = 'Isolated CSS parity check'; frame.setAttribute('sandbox', 'allow-scripts');
      frame.srcdoc = `<!doctype html><html><head><base href="${location.origin}${location.pathname.replace(/[^/]*$/, '')}"></head><body><div id="sink" class="css-neural-model"></div><script>
        let style; const sink=document.getElementById('sink');
        const numericToken=/^[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:e[+-]?\\d+)?$/i;
        const cssNumberReader=(${numberReaderSource});
        addEventListener('message',event=>{if(event.source!==parent||event.data?.type!=='run-rl-css-parity')return;try{
          style?.remove();style=document.createElement('style');style.textContent=event.data.css;document.head.append(style);
          const inputNames=['--bird-y','--velocity-y','--pipe-distance','--gap-y','--pipe-speed','--following-gap-y'];
          const probe=cssNumberReader(sink)('--nn-probe');let maxProbability=0,maxLogit=0,maxHidden=0,mismatches=0;
          const cases=event.data.fixture.cases.map(sample=>{sample.inputs.forEach((v,i)=>sink.style.setProperty(inputNames[i],String(v)));const read=cssNumberReader(sink);const probability=read('--jump-probability'),logit=read('--output-z'),hidden=Array.from({length:16},(_,i)=>read('--h'+(i+1)));maxProbability=Math.max(maxProbability,Math.abs(probability-sample.expected.probability));maxLogit=Math.max(maxLogit,Math.abs(logit-sample.expected.logit));maxHidden=Math.max(maxHidden,...hidden.map((v,i)=>Math.abs(v-sample.expected.hidden[i])));if(Math.abs(sample.expected.probability-event.data.fixture.threshold)>=1e-4&&(probability>=event.data.fixture.threshold)!==(sample.expected.probability>=event.data.fixture.threshold))mismatches++;return{id:sample.id,inputs:sample.inputs,actualProbability:probability,actualLogit:logit,actualHidden:hidden}});
          const passed=Math.abs(probe-event.data.fixture.expectedProbe)<1e-4&&Math.max(maxProbability,maxLogit,maxHidden)<1e-4&&mismatches===0;
          parent.postMessage({type:'rl-css-parity-result',report:{version:1,passed,checkedAt:new Date().toISOString(),browser:navigator.userAgent,modelSha256:event.data.fixture.modelSha256,cssSha256:event.data.cssSha256,caseCount:cases.length,tolerance:1e-4,probe,maximumProbabilityError:maxProbability,maximumLogitError:maxLogit,maximumHiddenError:maxHidden,decisionMismatches:mismatches,cases}},'*');
        }catch(error){parent.postMessage({type:'rl-css-parity-error',error:String(error.message||error)},'*')}});
      </script></body></html>`;
      parityFrame = frame;
      const timeout = setTimeout(() => { cleanup(); reject(new Error('Isolated parity check timed out.')); }, 15000);
      const cleanup = () => { clearTimeout(timeout); window.removeEventListener('message', onMessage); frame.remove(); parityFrame = null; };
      const onMessage = async (event) => {
        if (event.source !== frame.contentWindow || !['rl-css-parity-result', 'rl-css-parity-error'].includes(event.data?.type)) return;
        cleanup();
        if (event.data.type === 'rl-css-parity-error') { reject(new Error(event.data.error)); return; }
        const report = event.data.report;
        const browserPassed = report.passed === true && report.cssSha256 === candidateDigest && report.caseCount === 4680 && report.tolerance === 1e-4;
        if (!browserPassed) {
          parityPassed = false;
          const probeError = Math.abs(Number(report.probe) - candidateFixture.expectedProbe);
          setMessage('validation-status', `Parity failed · probability error ${Number(report.maximumProbabilityError).toExponential(2)}, logit error ${Number(report.maximumLogitError).toExponential(2)}, activation error ${Number(report.maximumHiddenError).toExponential(2)}, probe error ${probeError.toExponential(2)} · limit < 1e-4 · decision mismatches ${report.decisionMismatches}.`, 'error');
          $('validation-badge').textContent = 'FAILED'; $('save-policy').disabled = true; resolve(); return;
        }
        try {
          if (job?.id !== validationJobId) throw new Error('The active training job changed during verification.');
          const accepted = await api('/api/rl/parity', { method: 'POST', body: JSON.stringify({ id: validationJobId, report }) });
          if (accepted?.job?.candidateCssVerified !== true) throw new Error(accepted?.error || 'The service did not verify the browser parity report.');
          if (job?.id !== validationJobId) throw new Error('The active training job changed during verification.');
          parityPassed = true; job = { ...job, candidateCssVerified: true }; validation = { jobId: job.id, report };
          setMessage('validation-status', `Passed 4,680 cases · max probability error ${Number(report.maximumProbabilityError).toExponential(2)} · max activation error ${Number(report.maximumHiddenError).toExponential(2)}.`, 'success');
          $('validation-badge').textContent = 'PARITY PASSED'; $('save-policy').disabled = false;
        } catch (error) {
          parityPassed = false; setMessage('validation-status', error.message, 'error'); $('validation-badge').textContent = 'FAILED'; $('save-policy').disabled = true;
        }
        resolve();
      };
      window.addEventListener('message', onMessage);
      frame.addEventListener('load', () => frame.contentWindow.postMessage({ type: 'run-rl-css-parity', css: candidateCss, fixture: candidateFixture, cssSha256: candidateDigest }, '*'), { once: true });
      document.body.append(frame);
    });
  }
  $('save-policy').addEventListener('click', async () => {
    if (!parityPassed || !job?.id) return;
    const savedDifficulty = job.difficulty;
    $('save-policy').disabled = true; setMessage('validation-status', 'Saving the parity-verified CSS policy…');
    try {
      const result = await api('/api/rl/save', { method: 'POST', body: JSON.stringify({ id: job.id }) });
      saved = result?.saved || { ...saved, [job.difficulty]: true };
      setMessage('validation-status', 'Policy saved. It can now be run as a CSS evaluation demo.', 'success');
      await refreshNow();
      await onSaved(savedDifficulty);
    } catch (error) { $('save-policy').disabled = false; setMessage('validation-status', error.message, 'error'); }
  });
  window.addEventListener('resize', () => renderer.resize());
  $('num-envs').disabled = $('training-mode').value === 'live';
  $('env-field').hidden = $('training-mode').value === 'live';
  renderer.draw(idleGame);
  updateBrain(null);
  requestAnimationFrame(updatePlayback);
  requestAnimationFrame(demoFrame);
  refresh();

}

export async function mountRlLab(root, { prefix = '', onSaved } = {}) {
  const response = await fetch('./templates/rl-workspace.html', { cache: 'no-store' });
  if (!response.ok) throw new Error('The RL workspace could not be loaded.');
  root.innerHTML = await response.text();
  root.insertBefore(root.querySelector('.rl-lower-grid'), root.querySelector('.rl-workspace'));
  if (prefix) {
    const title = root.querySelector('.rl-intro h1');
    const heading = document.createElement('h2');
    heading.innerHTML = title.innerHTML;
    title.replaceWith(heading);
    root.querySelectorAll('[id]').forEach(element => { element.id = prefix + element.id; });
    root.querySelectorAll('[for]').forEach(element => { element.htmlFor = prefix + element.htmlFor; });
  }
  bindRlLab({ root, prefix, onSaved });
}
