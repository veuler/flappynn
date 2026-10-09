import { normalizeState } from './normalization.js';

export function renderState(get, state, decisionInputs = null, normalization) {
  get('bird-y').textContent = `${state.birdY.toFixed(1)} px`;
  get('velocity').textContent = `${state.birdVelocityY.toFixed(1)} px/s`;
  get('distance').textContent = state.nextPipeDistanceX === null ? '—' : `${state.nextPipeDistanceX.toFixed(1)} px`;
  get('gap-y').textContent = state.nextPipeGapCenterY === null ? '—' : `${state.nextPipeGapCenterY.toFixed(1)} px`;
  try {
    const values = decisionInputs ?? normalizeState(state, normalization);
    ['norm-bird-y', 'norm-velocity', 'norm-distance', 'norm-gap-y'].forEach((id, index) => {
      get(id).textContent = values[index].toFixed(3);
    });
    if (values.length >= 5) get('norm-pipe-speed').textContent = values[4].toFixed(3);
    if (values.length >= 6) {
      get('following-gap-y').textContent = `${state.followingPipeGapCenterY.toFixed(1)} px`;
      get('norm-following-gap-y').textContent = values[5].toFixed(3);
    }
    get('state-error').hidden = true;
  } catch (error) {
    get('state-error').hidden = false;
    get('state-error').textContent = error.message;
  }
}

export function renderTraining(get, recorder, mode, status) {
  get('sample-count').textContent = recorder.samples.length;
  get('jump-count').textContent = recorder.jumpCount;
  get('wait-count').textContent = recorder.samples.length - recorder.jumpCount;
  get('session-count').textContent = recorder.sessions.length;
  const hardFlights = recorder.sessions.filter((flight) => flight.difficulty === 'hard').length;
  get('recording-difficulties').textContent = `Flights · Normal ${recorder.sessions.length - hardFlights} · Hard ${hardFlights}`;
  get('export').disabled = recorder.samples.length === 0;
  get('clear').disabled = recorder.samples.length === 0 || ['running', 'paused'].includes(status);
  get('recording-status').textContent = recorder.full ? 'LIMIT REACHED' :
    mode === 'training' && status === 'running' ? 'RECORDING ON' :
      mode === 'training' && status === 'paused' ? 'RECORDING PAUSED' : 'RECORDING OFF';
  get('capacity-note').hidden = !recorder.full;
}

export function renderBrain(get, ai, selectedMode) {
  get('brain-panel').hidden = false;
  get('ai-option').disabled = !ai.ready;
  get('threshold').disabled = !ai.ready;
  get('threshold-value').textContent = ai.threshold.toFixed(2);
  get('model-status').textContent = ai.ready && ai.isTrial ? 'Trial model active; the saved model is unchanged.' : ai.ready && ['warm-start-legacy', 'warm-start-lookahead'].includes(ai.metadata.dataKind) ? 'Transferred starting weights; train to learn from the added input.' : ai.ready && ai.metadata.trainingDifficulty === 'hard' ? 'Hard-course model ready.' : ai.ready ?
    ai.metadata.dataKind === 'synthetic-test' ? 'Synthetic test model ready.' :
      ai.metadata.dataKind === 'reinforcement-learning-ppo' ? 'CSS policy trained with PPO is ready.' :
      ai.metadata.dataKind === 'self-play-neuroevolution' ? 'Self-trained CSS model ready.' : 'CSS model trained on human recordings is ready.' :
    ai.error || 'Checking the CSS model…';
  const active = selectedMode === 'ai';
  const prediction = active ? ai.lastPrediction : null;
  get('brain-probability').textContent = prediction ? `${(prediction.probability * 100).toFixed(1)}%` : '—';
  get('brain-decision').textContent = !active ? 'AI OFF' : ai.decision === 'COOLDOWN' ? 'COOLDOWN' : ai.decision;
  get('brain-decision').dataset.action = active ? ai.decision : 'OFF';
  get('brain-jump-count').textContent = (active ? ai.jumpCount : 0).toLocaleString('en-US');
  get('brain-wait-count').textContent = (active ? ai.waitCount : 0).toLocaleString('en-US');
  get('brain-logit').textContent = prediction ? prediction.logit.toFixed(3) : '—';
  get('brain-cycles').textContent = (active ? ai.decisionCount : 0).toLocaleString('en-US');
  for (let index = 0; index < (ai.profile?.architecture[1] ?? 16); index++) {
    const activation = prediction?.hidden[index];
    get(`neuron-value-${index + 1}`).textContent = activation === undefined ? '—' : activation.toFixed(3);
    get(`neuron-bar-${index + 1}`).style.setProperty('--activation', String(activation ?? 0));
  }
}
