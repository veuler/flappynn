import { CONFIG, DIFFICULTIES } from './config.js';
import { Game } from './game.js';
import { Renderer } from './renderer.js';
import { bindInput } from './input.js';
import { TrainingRecorder } from './training-recorder.js';
import { SessionController } from './session-controller.js';
import { renderState, renderTraining, renderBrain } from './debug-ui.js';
import { downloadDataset } from './download.js';
import { AiController } from './ai-controller.js';
import { BrainVisualizer } from './brain-visualizer.js';
import { bindSelfLearning } from './self-learning-ui.js';
import { modelProfile } from './model-profiles.js';
import { mountRlLab } from './rl-lab.js';

const get = (id) => document.getElementById(id);
const canvas = get('game');
const game = new Game();
const renderer = new Renderer(canvas, CONFIG);
const profile = modelProfile(document.body.dataset.modelProfile || 'lookahead-616');
const recorder = new TrainingRecorder({ profileId: profile.id });
const ai = new AiController(get('ai-output'), profile.base, profile.id);
const brainVisualizer = new BrainVisualizer(get('brain-network'), get('network-status'), profile.architecture);
const session = new SessionController(game, recorder, ai);
const bestScores = { normal: 0, hard: 0 };
let lastTime = null;
let accumulator = 0;
let lastStatus = null;
let startingRun = false;
let switchingModel = false;
const humanOnly = profile.id === 'lookahead-616';
const statuses = { ready: 'READY', running: 'FLYING', paused: 'PAUSED', gameover: 'FLIGHT ENDED' };

function readSeed() {
  if (get('sequence').value === 'random') {
    return crypto.getRandomValues(new Uint32Array(1))[0];
  }
  const value = Number(get('seed').value);
  if (!Number.isInteger(value) || value < 0 || value > 4294967295 || get('seed').value.trim() === '') {
    get('seed').setCustomValidity('Enter an integer between 0 and 4294967295.');
    get('seed').reportValidity();
    return null;
  }
  get('seed').setCustomValidity('');
  return value;
}

async function startRun() {
  if (startingRun || switchingModel) return;
  const seed = readSeed();
  if (seed === null) return;
  startingRun = true;
  try {
    if (get('mode').value === 'ai' && !ai.isTrial) {
      // A restart ends the old flight before another model can be installed.
      session.finish();
      if (!await ai.loadSaved(get('difficulty').value)) throw new Error(ai.error);
      get('threshold').value = ai.threshold;
    }
    session.start(seed, get('mode').value, get('difficulty').value);
  } catch (error) {
    get('model-status').textContent = error.message;
    return;
  } finally { startingRun = false; }
  get('clear-confirmation').hidden = true;
  accumulator = 0;
  lastTime = null;
  canvas.focus({ preventScroll: true });
}

function jump() {
  if (game.status === 'ready' || game.status === 'gameover') startRun();
  else if (game.status === 'paused') togglePause();
  else session.requestJump();
}

function togglePause() {
  if (game.status === 'running') game.pause();
  else if (game.status === 'paused') game.resume();
  accumulator = 0;
  lastTime = null;
  canvas.focus({ preventScroll: true });
}

function renderUI() {
  const state = game.getState();
  bestScores[state.difficulty] = Math.max(bestScores[state.difficulty], state.score);
  get('score').textContent = state.score;
  const activeRun = ['running', 'paused'].includes(state.status);
  const shownDifficulty = activeRun ? state.difficulty : get('difficulty').value;
  const runDifficulty = state.status === 'ready' ? shownDifficulty : state.difficulty;
  get('best-score').textContent = bestScores[runDifficulty];
  get('difficulty-label').textContent = DIFFICULTIES[runDifficulty].label.toLocaleUpperCase('en-US');
  get('flight-speed').textContent = `${state.pipeSpeed.toFixed(1)} px/s`;
  get('time').textContent = `${state.elapsedSeconds.toFixed(1)} s`;
  renderState(get, state, session.mode === 'ai' ? ai.lastPrediction?.inputs : null, profile.normalization);
  renderTraining(get, recorder, session.mode, state.status);
  get('active-seed').textContent = state.seed;
  get('mode').disabled = activeRun || startingRun || switchingModel;
  if (get('model-source')) get('model-source').disabled = activeRun || startingRun || switchingModel || ai.isTrial;
  get('finish').disabled = !['running', 'paused'].includes(state.status);
  const selectedMode = get('mode').value;
  get('difficulty').disabled = activeRun || startingRun || switchingModel;
  get('start').disabled = startingRun || switchingModel || selectedMode === 'ai' && !ai.ready;
  get('difficulty-note').textContent = selectedMode === 'training' ? `Your inputs are recorded with the ${DIFFICULTIES[shownDifficulty].label.toLowerCase()} difficulty saved in each flight.` :
    shownDifficulty === 'hard' ? 'Speed rises smoothly with each point, up to 270 px/s.' : 'Normal mode keeps a steady speed of 180 px/s.';
  get('hard-model-note').hidden = selectedMode !== 'ai' || shownDifficulty !== 'hard';
  get('hard-model-note').textContent = ai.metadata?.trainingDifficulty === 'hard' ? 'Using the model trained on hard courses.' : 'A saved hard model loads at takeoff; otherwise, the normal model is used.';
  renderBrain(get, ai, selectedMode);
  brainVisualizer.update(selectedMode === 'ai' ? ai.lastPrediction : null, true, selectedMode === 'ai' ? state.status : 'ready');
  get('jump-hint').textContent = selectedMode === 'ai' ? 'start a flight' : 'jump';
  canvas.setAttribute('aria-label', selectedMode === 'ai' ?
    'The CSS network controls the bird. Start with Space or a tap; pause with P.' :
    'Game area. Jump with Space, Up, W, or a tap; pause with P.');
  get('mode-label').textContent = { human: 'HUMAN MODE', training: 'INPUT RECORDING', ai: 'CSS AI MODE' }[selectedMode];
  get('start').textContent = state.status === 'ready' ?
    { human: 'Start flying', training: 'Start a recording flight', ai: 'Start an AI flight' }[selectedMode] : 'Restart flight';
  if (state.status === lastStatus) return;
  lastStatus = state.status;
  get('status').textContent = statuses[state.status];
  get('status').dataset.state = state.status;
  get('announcement').textContent = state.status === 'gameover' ? `Flight ended. Score: ${state.score}.` : statuses[state.status];
  get('pause').disabled = !['running', 'paused'].includes(state.status);
  get('pause').textContent = state.status === 'paused' ? 'Resume' : 'Pause';
  get('overlay').hidden = state.status === 'running';
  const messages = {
    ready: ['Ready for takeoff?', 'Fly through the gaps and earn a point for each pipe.', 'SPACE or tap → start'],
    paused: ['Take a breath.', 'Resume your flight when you are ready.', 'P or tap → resume'],
    gameover: ['Another flight?', `${state.score} pipes cleared. Every flight is a fresh attempt.`, 'SPACE or tap → restart'],
  };
  if (messages[state.status]) {
    const [title, text, hint] = messages[state.status];
    get('overlay-title').textContent = title;
    get('overlay-text').textContent = text;
    get('overlay-hint').textContent = hint;
  }
}

function frame(timestamp) {
  if (lastTime !== null && game.status === 'running') {
    accumulator += Math.min((timestamp - lastTime) / 1000, CONFIG.maxFrameSeconds);
    while (accumulator >= CONFIG.fixedStepSeconds && game.status === 'running') {
      session.step();
      accumulator -= CONFIG.fixedStepSeconds;
    }
    if (game.status !== 'running') accumulator = 0;
  }
  lastTime = timestamp;
  renderer.draw(game);
  renderUI();
  brainVisualizer.draw(timestamp);
  requestAnimationFrame(frame);
}

bindInput({ canvas, onJump: jump, onPause: togglePause });
get('start').addEventListener('click', startRun);
get('pause').addEventListener('click', togglePause);
get('finish').addEventListener('click', () => {
  session.finish();
  accumulator = 0;
  renderUI();
});
get('mode').addEventListener('change', () => {
  if (get('mode').value === 'ai') ai.reset();
  renderUI();
});
get('difficulty').addEventListener('change', () => {
  renderUI();
  if (ai.savedFamily === 'rl') changeModelFamily();
});
async function changeModelFamily() {
  if (!get('model-source') || startingRun || switchingModel || ['running', 'paused'].includes(game.status)) return;
  switchingModel = true;
  ai.setSavedFamily(get('model-source').value);
  ai.ready = false;
  ai.style?.remove(); ai.style = null;
  ai.isTrial = false; ai.reset(); renderUI();
  try {
    if (!await ai.loadSaved(get('difficulty').value, { force: true })) {
      get('model-status').textContent = ai.savedFamily === 'rl' ? 'Save a CSS-verified RL policy for this difficulty first.' : ai.error;
      ai.error = get('model-status').textContent;
    }
    get('threshold').value = ai.threshold;
  } finally { switchingModel = false; renderUI(); }
}
get('model-source')?.addEventListener('change', changeModelFamily);
get('threshold').addEventListener('input', () => ai.setThreshold(Number(get('threshold').value)));
get('export').addEventListener('click', () => {
  if (recorder.samples.length === 0) return;
  pauseWhenAway();
  downloadDataset(recorder.exportDataset(game.getState()));
});
get('import-recordings').addEventListener('click',()=>get('recording-file').click());
get('recording-file').addEventListener('change',async event=>{
  const file=event.target.files[0]; if (!file) return;
  try {
    if (['running','paused'].includes(game.status)) throw new Error('End your flight before importing recordings.');
    if (recorder.samples.length) throw new Error('Download and clear the current recordings before importing another file.');
    if (file.size > 16*1024*1024) throw new Error('The recording file exceeds 16 MiB.');
    recorder.importDataset(JSON.parse(await file.text()));
    const difficulties=new Set(recorder.sessions.map(flight=>flight.difficulty));
    if(difficulties.size===1) get('difficulty').value=[...difficulties][0];
    get('recording-import-status').textContent=`Imported ${recorder.samples.length.toLocaleString('en-US')} samples from ${recorder.sessions.length} flights.`;
  } catch(error) {get('recording-import-status').textContent=error.message;}
  finally {event.target.value='';renderUI();}
});
get('clear').addEventListener('click', () => { get('clear-confirmation').hidden = false; });
get('cancel-clear').addEventListener('click', () => { get('clear-confirmation').hidden = true; });
get('confirm-clear').addEventListener('click', () => {
  if (['running', 'paused'].includes(game.status)) return;
  recorder.clear();
  get('recording-import-status').textContent='';
  get('clear-confirmation').hidden = true;
  get('announcement').textContent = 'Recording data cleared.';
});
get('sequence').addEventListener('change', () => {
  get('seed').disabled = get('sequence').value === 'random';
});
get('seed').addEventListener('input', () => get('seed').setCustomValidity(''));
function pauseWhenAway() {
  game.pause();
  accumulator = 0;
  lastTime = null;
}
document.addEventListener('visibilitychange', () => { if (document.hidden) pauseWhenAway(); });
window.addEventListener('blur', pauseWhenAway);
window.addEventListener('resize', () => renderer.resize());
window.addEventListener('beforeunload', (event) => {
  if (recorder.samples.length === 0) return;
  event.preventDefault();
  event.returnValue = '';
});
renderUI();
bindSelfLearning({ get, ai, recorder, humanOnly, canApply: () => !startingRun && !switchingModel && !['running', 'paused'].includes(game.status),
  onBusy: (busy) => { switchingModel = busy; renderUI(); },
  onApplied: (difficulty) => {
    if (get('model-source')) get('model-source').value = ai.savedFamily;
    if (['normal', 'hard'].includes(difficulty)) get('difficulty').value = difficulty;
    get('threshold').value = ai.threshold; renderUI();
  } });
if (humanOnly) {
  const rlRoot = get('rl-workspace-root');
  mountRlLab(rlRoot, { prefix: 'ppo-', onSaved: async (difficulty) => {
    if (startingRun || switchingModel || ['running', 'paused'].includes(game.status)) return;
    get('difficulty').value = difficulty;
    get('model-source').value = 'rl';
    await changeModelFamily();
  } }).catch(error => { rlRoot.textContent = error.message; });
}
ai.initialize().then(() => {
  get('threshold').value = ai.threshold;
  if (ai.ready && new URL(location.href).searchParams.get('mode') === 'ai') get('mode').value = 'ai';
  renderUI();
});
requestAnimationFrame(frame);
