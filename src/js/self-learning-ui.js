export function bindSelfLearning({ get, ai, recorder, canApply, onApplied, onBusy = () => {}, humanOnly = false }) {
  const api = ai.profile.api;
  const learning = ai.profile.learning;
  let lastJob = null;
  let frame = null;
  let checking = null;
  let pending = false;
  let waitingId = null;
  let autoAppliedId = null;
  const activeStates = ['starting', 'running', 'evaluating', 'awaiting-validation', 'validating'];
  async function post(operation, data = {}) {
    const response = await fetch(`${api}/${operation}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The training service is unavailable.');
    return result;
  }
  function render(job) {
    lastJob = job;
    const active = job && activeStates.includes(job.state);
    const trial = ai.isTrial;
    const unchanged = job?.comparisonSkipped === 'unchanged-weights';
    const human = job?.kind === 'human';
    if (!humanOnly) get('self-train-start').disabled = pending || active || !ai.ready || trial;
    const flights=recorder.sessions.filter(flight=>flight.difficulty===get('difficulty').value);
    get('human-train').disabled=pending || active || (!humanOnly && !ai.ready) || trial || !canApply() || flights.reduce((sum,flight)=>sum+flight.sampleCount,0)<200 || new Set(flights.map(flight=>flight.seed)).size<2;
    get('human-train-initialization').disabled=pending || active || !canApply();
    const randomStart=get('human-train-initialization').value==='random';
    get('human-initialization-note').textContent=randomStart ? 'Starts a new model with random weights; your recordings and saved model stay intact until a candidate is accepted.' : 'Continues the saved model using your recordings.';
    get('human-train').textContent=randomStart ? 'Train from scratch with my recordings' : 'Train from my recordings';
    get('import-recordings').disabled=pending || active || !canApply() || recorder.samples.length>0;
    get('self-train-try').hidden = !job?.candidateCssVerified || unchanged || trial || job.state === 'promoted';
    get('self-train-try').textContent = job?.state === 'kept' ? 'Try candidate' : 'Try without waiting for comparison';
    get('self-train-try').disabled = pending || !canApply();
    get('self-train-restore').hidden = !trial;
    get('self-train-restore').disabled = pending || !canApply();
    const canPromote=job?.state==='kept' && job.candidateCssVerified && !unchanged;
    get('self-train-promote').hidden=!canPromote;
    get('self-train-promote').disabled=pending || !canApply();
    get('self-train-promote-note').hidden=!canPromote;
    get('self-train-trial-note').hidden = !trial && get('self-train-try').hidden;
    get('self-train-trial-note').textContent = trial ? 'Trial model active on this page; return to the saved model before training again.' : 'Try the candidate on this page now; refreshing returns to the saved model.';
    if (!humanOnly) {
      get('self-train-duration').disabled = active;
      get('self-train-difficulty').disabled = active || pending;
      if (active && job.difficulty) get('self-train-difficulty').value = job.difficulty;
    }
    get('self-train-stop').hidden = !['starting', 'running', 'evaluating'].includes(job?.state);
    get('self-train-retry').hidden = job?.state !== 'failed' || job.canRetryValidation !== true;
    get('self-train-error').hidden = job?.state !== 'failed' || !job.error;
    get('self-train-error-detail').textContent = job?.error || '';
    const applied = job?.candidateModelSha256 && ai.metadata?.modelSha256 === job.candidateModelSha256;
    get('self-train-load').hidden = job?.state !== 'promoted' || applied;
    get('self-train-load').disabled = !canApply();
    get('self-train-progress').value = job ? human ? !['starting','running','failed'].includes(job.state) ? 1 : Math.min(1,(job.epoch || 0)/(job.maximumEpochs || 500)) : Math.min(1, (job.elapsedSeconds || 0) / job.durationSeconds) : 0;
    const finished = ['promoted', 'kept'].includes(job?.state);
    const comparison = job?.validationProgress;
    get('self-compare').hidden = unchanged || !comparison && !finished && !['awaiting-validation', 'validating'].includes(job?.state);
    const completed = finished ? 40 : comparison?.completed || 0;
    get('self-compare-count').textContent = `${completed} / 40 · ${Math.round(completed / 40 * 100)}%`;
    if (!finished && (comparison?.phase === 'css' || job?.state === 'awaiting-validation' && !comparison)) get('self-compare-progress').removeAttribute('value');
    else get('self-compare-progress').value = completed;
    const phases = { baseline: 'Saved model', candidate: 'New model' };
    get('self-compare-stage').textContent = finished ? 'Both models have completed their 20-seed benchmark.' :
      job?.state === 'failed' ? 'Validation stopped. You can retry with the saved candidate.' :
      phases[comparison?.phase] ? `${phases[comparison.phase]}: ${comparison.phaseCompleted}/20 completed${comparison.currentSeed ? ` · running seed ${comparison.currentSeed}` : ''}` :
      comparison?.phase === 'finalizing' ? '40/40 runs completed. Checking the results…' : 'Checking CSS calculations first…';
    get('self-compare-location').hidden = waitingId !== job?.id || !active;
    get('self-compare-location').textContent = 'Comparison is running in another tab; progress updates here too.';
    get('self-train-details').textContent = job ? human ? `${job.difficulty === 'hard' ? 'Hard' : 'Normal'} · ${job.initialization==='random' ? 'Random starting weights' : 'Continue saved weights'} · ${(job.sampleCount || 0).toLocaleString('en-US')} recorded samples · epoch ${job.epoch || 0} · threshold ${job.threshold.toFixed(2)}` : `${job.difficulty === 'hard' ? 'Hard' : 'Normal'} · ${job.generation || 0} generations · ${(job.episodes || 0).toLocaleString('en-US')} trials · threshold ${job.threshold.toFixed(2)}` : humanOnly ? 'Learns your recorded jump and wait decisions; PPO training is configured above.' : 'Starts with the chosen course model and decision threshold.';
    const acceptance=job?.automaticAcceptance;
    get('self-train-score-summary').hidden=!acceptance;
    if (acceptance) get('self-train-score-summary').textContent=`Median ${acceptance.baseline.median.toFixed(1)} → ${acceptance.candidate.median.toFixed(1)} · P10 (lower-score courses) ${acceptance.baseline.p10.toFixed(1)} → ${acceptance.candidate.p10.toFixed(1)} · Severe regressions ${acceptance.catastrophicRegressions.length}/${acceptance.policy.maximumCatastrophicRegressions}`;
    const remaining = Math.ceil(Math.max(0, (job?.durationSeconds || 0) - (job?.elapsedSeconds || 0)));
    const labels = {
      starting: 'Preparing local training…',
      running: human ? `Learning from your recorded inputs · epoch ${job?.epoch || 0} / ${job?.maximumEpochs || 500}` : `Learning from its own flights · remaining ${Math.floor(remaining / 60)}:${String(remaining % 60).padStart(2, '0')}`,
      evaluating: 'Preparing the candidate model…',
      'awaiting-validation': 'Starting CSS and gameplay validation…',
      validating: comparison?.phase === 'finalizing' ? 'Comparison complete; evaluating the result…' : 'Comparing models.',
      promoted: `${job?.promotionMode==='manual' ? applied ? 'Your chosen model is in use' : 'Your chosen model is saved' : applied ? 'New model in use' : 'New model ready'} · average ${job?.baselineMeanScore?.toFixed(1)} → ${job?.candidateMeanScore?.toFixed(1)} pipes`,
      kept: unchanged ? 'Weights and threshold are unchanged; the duplicate comparison was skipped.' : `Saved model kept · average ${job?.baselineMeanScore?.toFixed(1)} → ${job?.candidateMeanScore?.toFixed(1)} pipes${acceptance ? ` · ${acceptance.reasons.join(' ')}` : job?.candidateMeanScore>job?.baselineMeanScore ? ' · some courses regressed under the earlier acceptance rule.' : ''}`,
      failed: job?.canRetryValidation ? 'Validation stopped; retry with the saved candidate.' : 'Training stopped; the saved model is safe and you can start a new session.',
    };
    get('self-train-status').textContent = labels[job?.state] || (humanOnly ? 'Record at least 200 samples on two seeds, then train from your recordings.' : 'Start with a 5-minute session.');
    if (job?.state === 'awaiting-validation' && !canApply()) get('self-train-status').textContent = 'Training finished. End your flight to start CSS and gameplay checks.';
    get('self-train-report').hidden = !job || active || job.state === 'failed' && !job.canRetryValidation;
    if (job) get('self-train-report').href = `./${learning}/${job.id}/report.json`;
    if (job && ['awaiting-validation', 'validating'].includes(job.state) && checking !== job.id && canApply()) {
      checking = job.id;
      frame?.remove();
      frame = document.createElement('iframe');
      frame.title = 'Independent CSS validation for the training candidate';
      frame.className = 'learning-check-frame';
      frame.src = `./learning-check.html?job=${job.id}&profile=${ai.profile.id}`;
      document.body.append(frame);
    }
    if (job && !active) { frame?.remove(); frame = null; waitingId = null; checking = null; }
    if (human && job.state==='promoted' && ai.ready && !applied && !pending && canApply() && autoAppliedId!==job.id) {
      autoAppliedId=job.id; switchModel(null,false,job.difficulty);
    }
  }
  async function poll() {
    try {
      const response = await fetch(api, { cache: 'no-store' });
      if (!response.ok) throw new Error('Run npm start to open the current local training server.');
      const result = await response.json();
      get('self-train-policy-note').textContent=result.promotionPolicy ? `Auto-save needs a ${Math.round((result.promotionPolicy.minimumMeanMultiplier-1)*100)}% higher average, no lower median, P10 within 10%, and at most one severe course regression; manual saving remains available after comparison.` : 'This server is using the earlier acceptance rule; restart it after the current comparison finishes to enable the updated settings.';
      render(humanOnly && result.job?.kind !== 'human' ? null : result.job);
    } catch (error) { get('self-train-status').textContent = error.message; }
    setTimeout(poll, 1000);
  }
  if (!humanOnly) get('self-train-start').addEventListener('click', async () => {
    pending = true; render(lastJob);
    try { render((await post('start', { seconds: Number(get('self-train-duration').value), threshold: ai.threshold, difficulty: get('self-train-difficulty').value })).job); }
    catch (error) { get('self-train-status').textContent = error.message; }
    finally { pending = false; }
  });
  get('human-train').addEventListener('click',async()=>{
    if (!canApply() || pending) return;
    pending=true;render(lastJob);
    try {render((await post('human',{dataset:recorder.exportDataset(),difficulty:get('difficulty').value,threshold:ai.threshold,initialization:get('human-train-initialization').value})).job);}
    catch(error){get('recording-import-status').textContent=error.message;}
    finally{pending=false;render(lastJob);}
  });
  get('self-train-stop').addEventListener('click', async () => {
    try { await post('stop', { id: lastJob.id }); get('self-train-status').textContent = 'Stopping training; the best candidate will be validated.'; }
    catch (error) { get('self-train-status').textContent = error.message; }
  });
  get('self-train-retry').addEventListener('click', async () => {
    get('self-train-retry').disabled = true;
    try { checking = null; render((await post('retry', { id: lastJob.id })).job); }
    catch (error) { get('self-train-status').textContent = error.message; }
    finally { get('self-train-retry').disabled = false; }
  });
  get('self-train-load').addEventListener('click', async () => {
    await switchModel(null, false, lastJob?.difficulty || 'normal');
  });
  get('human-train-initialization').addEventListener('change',()=>render(lastJob));
  get('self-train-promote').addEventListener('click',async()=>{
    if (!canApply() || pending || lastJob?.state!=='kept') return;
    const difficulty=lastJob.difficulty || 'normal';
    pending=true;onBusy(true);render(lastJob);
    let accepted=false;
    try {render((await post('promote',{id:lastJob.id})).job);accepted=true;}
    catch(error){get('self-train-status').textContent=error.message;}
    finally{pending=false;onBusy(false);}
    if (accepted) await switchModel(null,false,difficulty);
  });
  async function switchModel(base, asTrial = true, difficulty = get('difficulty').value) {
    if (!canApply() || pending) return;
    pending = true; onBusy(true); render(lastJob);
    try {
      if (humanOnly) ai.setSavedFamily('recordings');
      if (asTrial ? await ai.initialize(base) : await ai.loadSaved(difficulty, { force: true })) {
        ai.isTrial = asTrial; ai.reset(); onApplied(difficulty);
      }
    } finally { pending = false; onBusy(false); render(lastJob); }
    if (ai.error) get('self-train-status').textContent = ai.error;
  }
  get('self-train-try').addEventListener('click', () => switchModel(`./${learning}/${lastJob.id}/`, true, lastJob.difficulty || 'normal'));
  get('self-train-restore').addEventListener('click', () => switchModel(null, false));
  window.addEventListener('message', (event) => {
    if (event.origin === location.origin && event.source === frame?.contentWindow && event.data?.type === 'learning-check' && event.data.id === lastJob?.id) {
      waitingId = event.data.waiting ? lastJob.id : null;
      render(lastJob);
    }
  });
  poll();
}
