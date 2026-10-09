// Its own document prevents candidate CSS from replacing the live game's weights.
import { AiController } from './ai-controller.js';
import { CONFIG } from './config.js';
import { Game } from './game.js';
import { SessionController } from './session-controller.js';
import { TrainingRecorder } from './training-recorder.js';
import { CSS_INPUT_PROPERTIES, modelProfile } from './model-profiles.js';
import { cssNumberReader } from './css-numbers.js';

const id = new URL(location.href).searchParams.get('job');
const profile = modelProfile(new URL(location.href).searchParams.get('profile') || 'speed-516');
const sink = document.getElementById('ai-output');
const status = document.getElementById('check-status');
const base = `./${profile.learning}/${id}/`;
let token = null;
const signal = (state, waiting = false) => {
  status.textContent = state;
  parent.postMessage({ type: 'learning-check', id, state, waiting }, location.origin);
};
async function post(operation, data) {
  const response = await fetch(`${profile.api}/${operation}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id, token, ...data }) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Validation could not be saved.');
  return result;
}
async function progress(phase, completed, currentSeed = null) {
  await post('progress', { phase, completed, currentSeed });
}
async function parity() {
  await progress('css', 0);
  signal('Validating CSS/Python states…');
  const [fixture, css] = await Promise.all([
    fetch(`${base}parity-inputs.json`, { cache: 'no-store' }).then((r) => r.json()),
    fetch(`${base}model.css`, { cache: 'no-store' }).then((r) => r.text()),
  ]);
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(css)))].map((v) => v.toString(16).padStart(2, '0')).join('');
  if (digest !== fixture.cssSha256) throw new Error('The candidate CSS does not match its reference.');
  const style = document.createElement('style');
  style.textContent = css;
  document.head.append(style);
  try {
    const probe = cssNumberReader(sink)('--nn-probe');
    const cases = [];
    for (const sample of fixture.cases) {
      CSS_INPUT_PROPERTIES.slice(0, profile.architecture[0]).forEach((name, i) => sink.style.setProperty(name, String(sample.inputs[i])));
      const numeric = cssNumberReader(sink);
      cases.push({ id: sample.id, inputs: sample.inputs, actualProbability: numeric('--jump-probability'),
        actualLogit: numeric('--output-z'), actualHidden: Array.from({ length: profile.architecture[1] }, (_, i) => numeric(`--h${i + 1}`)) });
    }
    await post('parity', { report: { version: 1, passed: true, tolerance: 1e-4, probe, cases, caseCount: cases.length,
      modelSha256: fixture.modelSha256, cssSha256: fixture.cssSha256, browser: navigator.userAgent, checkedAt: new Date().toISOString() } });
  } finally { style.remove(); }
}
async function benchmark(modelBase, threshold, label, phase, difficulty = 'normal') {
  await progress(phase, 0, 1);
  const ai = new AiController(sink, modelBase, profile.id);
  if (!await ai.initialize()) throw new Error(ai.error);
  ai.setThreshold(threshold);
  const game = new Game();
  const session = new SessionController(game, new TrainingRecorder(), ai);
  const runs = [];
  try {
    for (let seed = 1; seed <= 20; seed++) {
      session.start(seed, 'ai', difficulty);
      let yieldedAt = performance.now();
      let reportedAt = yieldedAt;
      for (let tick = 0; tick < 180 / CONFIG.fixedStepSeconds && game.status === 'running'; tick++) {
        session.step();
        if (tick % 120 === 0 && performance.now() - yieldedAt > 16) {
          if (performance.now() - reportedAt > 1000) {
            await progress(phase, seed - 1, seed);
            reportedAt = performance.now();
          }
          await new Promise((resolve) => setTimeout(resolve, 0));
          yieldedAt = performance.now();
        }
      }
      if (game.status === 'paused') throw new Error(ai.error || 'CSS inference stopped.');
      runs.push({ seed, score: game.score, survivalSeconds: game.elapsedSeconds });
      await progress(phase, seed, seed < 20 ? seed + 1 : null);
      signal(`${label}: ${seed}/20 seeds · 3 minutes/run`);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return { modelSha256: ai.metadata.modelSha256, threshold, difficulty, maximumSimulationSeconds: 180, runs };
  } finally { ai.style?.remove(); }
}
try {
  if (!/^[a-f0-9-]{36}$/.test(id || '')) throw new Error('Invalid training job ID.');
  const { job } = await fetch(profile.api, { cache: 'no-store' }).then((r) => r.json());
  if (job?.id !== id) throw new Error('The training job changed.');
  let claim = await post('claim', {});
  while (!claim.claimed) {
    signal('Comparison is running in another tab; progress is shown here too.', true);
    await new Promise((resolve) => setTimeout(resolve, 5000));
    claim = await post('claim', {});
  }
  if (claim.claimed) {
    token = claim.token;
    signal('Validation started in this tab.');
    // The job may have advanced while this tab waited for its claim.
    if (claim.job.state === 'awaiting-validation') await parity();
    const baseline = await benchmark(`${base}baseline/`, claim.job.threshold, 'Saved model', 'baseline', claim.job.difficulty || 'normal');
    const candidate = await benchmark(base, claim.job.threshold, 'New model', 'candidate', claim.job.difficulty || 'normal');
    await progress('finalizing', 20);
    await post('result', { baseline, candidate });
    signal('Validation completed.');
  }
} catch (error) {
  signal(error.message);
  if (token) await post('failure', { error: error.message }).catch(() => {});
}
