import { CSS_INPUT_PROPERTIES, modelProfile } from './model-profiles.js';
import { cssNumberReader } from './css-numbers.js';
const profile = modelProfile(new URL(location.href).searchParams.get('profile') || 'speed-516');
const base = document.body.dataset.modelBase || (new URL(location.href).searchParams.get('difficulty') === 'hard' ? profile.hardBase : profile.base);
const get = id => document.getElementById(id);
let report;
let sheet;
get('parity-run').addEventListener('click', async () => {
  get('parity-run').disabled = true;
  get('parity-download').disabled = true;
  get('parity-status').textContent = 'Checking…';
  try {
    const [fixtures, styles] = await Promise.all([fetch(`${base}parity-inputs.json`, {cache:'no-store'}), fetch(`${base}model.css`, {cache:'no-store'})]);
    if (!fixtures.ok || !styles.ok) throw new Error('Export this model and its reference states first.');
    const fixture = await fixtures.json();
    const css = await styles.text();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(css)))].map(v => v.toString(16).padStart(2,'0')).join('');
    if (digest !== fixture.cssSha256 || JSON.stringify(fixture.architecture) !== JSON.stringify(profile.architecture)) throw new Error('Reference and CSS profile do not match.');
    sheet?.remove(); sheet = document.createElement('style'); sheet.textContent = css; document.head.append(sheet);
    const sink = get('ai-output');
    const probe = cssNumberReader(sink)('--nn-probe');
    let maximumProbabilityError = 0, maximumLogitError = 0, maximumHiddenError = 0, decisionMismatches = 0;
    const cases = fixture.cases.map(sample => {
      CSS_INPUT_PROPERTIES.slice(0, profile.architecture[0]).forEach((name,i) => sink.style.setProperty(name, String(sample.inputs[i])));
      const numeric = cssNumberReader(sink);
      const actualProbability = numeric('--jump-probability');
      const actualLogit = numeric('--output-z');
      const actualHidden = Array.from({length:profile.architecture[1]}, (_,i) => numeric(`--h${i+1}`));
      maximumProbabilityError = Math.max(maximumProbabilityError, Math.abs(actualProbability-sample.expected.probability));
      maximumLogitError = Math.max(maximumLogitError, Math.abs(actualLogit-sample.expected.logit));
      maximumHiddenError = Math.max(maximumHiddenError, ...actualHidden.map((v,i) => Math.abs(v-sample.expected.hidden[i])));
      if (Math.abs(sample.expected.probability-fixture.threshold) >= 1e-4 && (actualProbability >= fixture.threshold) !== (sample.expected.probability >= fixture.threshold)) decisionMismatches++;
      return {id:sample.id, inputs:sample.inputs, actualProbability, actualLogit, actualHidden};
    });
    const passed = Math.abs(probe-fixture.expectedProbe)<1e-4 && Math.max(maximumProbabilityError,maximumLogitError,maximumHiddenError)<1e-4 && decisionMismatches===0;
    report = {version:1, passed, checkedAt:new Date().toISOString(), browser:navigator.userAgent, modelSha256:fixture.modelSha256,
      cssSha256:digest, caseCount:cases.length, tolerance:1e-4, probe, maximumProbabilityError, maximumLogitError, maximumHiddenError, decisionMismatches, cases};
    get('parity-status').textContent = passed ? 'PASSED' : 'FAILED';
    get('parity-summary').textContent = `${profile.architecture.join(' → ')} · ${cases.length} states · probability error ${maximumProbabilityError.toExponential(3)} · logit error ${maximumLogitError.toExponential(3)} · activation error ${maximumHiddenError.toExponential(3)}`;
    get('parity-report').textContent = JSON.stringify(report, null, 2);
    get('parity-download').disabled = !passed;
  } catch(error) { get('parity-status').textContent = error.message; }
  finally { get('parity-run').disabled = false; }
});
get('parity-download').addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
  const link = document.createElement('a'); link.href=url; link.download=`model-parity-${profile.architecture[0]}16.json`; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
