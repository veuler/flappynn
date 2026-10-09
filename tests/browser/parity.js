import { cssNumberReader } from '/js/css-numbers.js';
const get = (id) => document.getElementById(id);
const inputProperties = ['--bird-y', '--velocity-y', '--pipe-distance', '--gap-y', '--pipe-speed', '--following-gap-y'];
let lastReport = null;
let verifiedStyle = null;

async function digest(text) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(bytes)].map((value) => value.toString(16).padStart(2, '0')).join('');
}

async function run() {
  get('run').disabled = true;
  get('download').disabled = true;
  get('status').textContent = 'Calculating…';
  get('status').dataset.result = '';
  lastReport = null;
  try {
    const [fixtureResponse, cssResponse] = await Promise.all([
      fetch('./generated/parity-inputs.json', { cache: 'no-store' }),
      fetch('./generated/model.css', { cache: 'no-store' }),
    ]);
    if (!fixtureResponse.ok || !cssResponse.ok) throw new Error('Generate the test files with the Python CSS exporter first.');
    const fixture = await fixtureResponse.json();
    const cssText = await cssResponse.text();
    if (await digest(cssText) !== fixture.cssSha256) throw new Error('The CSS and Python reference do not belong to the same model export.');
    // Use exactly the bytes just verified, even if a cached link sheet was older.
    if (verifiedStyle) verifiedStyle.remove();
    verifiedStyle = document.createElement('style');
    verifiedStyle.textContent = cssText;
    document.head.append(verifiedStyle);
    if (fixture.version !== 1 || fixture.tolerance !== 1e-4 || !fixture.cases?.length) throw new Error('Invalid reference file.');
    get('model-kind').textContent = fixture.dataKind === 'synthetic-test' ?
      'Synthetic test model: validates the training pipeline; it was not learned from human gameplay.' : 'Model type: ' + fixture.dataKind;
    get('model-sha').textContent = fixture.modelSha256;
    get('browser').textContent = navigator.userAgent;
    const sink = get('ai-output');
    const probe = cssNumberReader(sink)('--nn-probe');
    if (Math.abs(probe - fixture.expectedProbe) >= fixture.tolerance) throw new Error('Native exp() or numeric @property calculations are not supported.');
    const rows = [];
    let maximumProbabilityError = 0;
    let maximumHiddenError = 0;
    let maximumLogitError = 0;
    let decisionMismatches = 0;
    let nearThresholdCases = 0;
    for (const sample of fixture.cases) {
      if (sample.inputs.length !== 4 || !sample.inputs.every(Number.isFinite)) throw new Error('Invalid test input.');
      inputProperties.forEach((property, index) => sink.style.setProperty(property, String(sample.inputs[index])));
      const readNumber = cssNumberReader(sink);
      const actual = {
        probability: readNumber('--jump-probability'),
        logit: readNumber('--output-z'),
        hidden: Array.from({ length: fixture.architecture[1] }, (_, index) => readNumber(`--h${index + 1}`)),
      };
      const probabilityError = Math.abs(actual.probability - sample.expected.probability);
      const logitError = Math.abs(actual.logit - sample.expected.logit);
      const hiddenError = Math.max(...actual.hidden.map((value, index) => Math.abs(value - sample.expected.hidden[index])));
      maximumProbabilityError = Math.max(maximumProbabilityError, probabilityError);
      maximumHiddenError = Math.max(maximumHiddenError, hiddenError);
      maximumLogitError = Math.max(maximumLogitError, logitError);
      const closeToThreshold = Math.abs(sample.expected.probability - fixture.threshold) < fixture.tolerance;
      if (closeToThreshold) nearThresholdCases += 1;
      if (!closeToThreshold && (actual.probability >= fixture.threshold) !== (sample.expected.probability >= fixture.threshold)) decisionMismatches += 1;
      rows.push({ id: sample.id, inputs: sample.inputs, expectedProbability: sample.expected.probability,
        actualProbability: actual.probability, actualLogit: actual.logit, actualHidden: actual.hidden,
        probabilityError, logitError, hiddenError });
    }
    const passed = maximumProbabilityError < fixture.tolerance && maximumHiddenError < fixture.tolerance &&
      maximumLogitError < fixture.tolerance && decisionMismatches === 0;
    lastReport = {
      version: 1, passed, checkedAt: new Date().toISOString(), browser: navigator.userAgent,
      modelSha256: fixture.modelSha256, cssSha256: fixture.cssSha256, dataKind: fixture.dataKind,
      caseCount: rows.length, tolerance: fixture.tolerance, probe,
      maximumProbabilityError, maximumHiddenError, maximumLogitError, decisionMismatches, nearThresholdCases,
      cases: rows,
    };
    get('case-count').textContent = rows.length;
    get('max-error').textContent = maximumProbabilityError.toExponential(3);
    get('status').textContent = passed ? 'PASSED' : 'FAILED';
    get('status').dataset.result = passed ? 'pass' : 'fail';
    get('summary').textContent = `Hidden-neuron error: ${maximumHiddenError.toExponential(3)} · Logit error: ${maximumLogitError.toExponential(3)} · Decision mismatches: ${decisionMismatches} · Near-threshold states: ${nearThresholdCases}`;
    get('report-json').textContent = JSON.stringify(lastReport, null, 2);
    get('download').disabled = false;
  } catch (error) {
    get('status').textContent = 'FAILED';
    get('status').dataset.result = 'fail';
    get('summary').textContent = error.message;
    get('report-json').textContent = JSON.stringify({ passed: false, error: error.message }, null, 2);
  } finally {
    get('run').disabled = false;
  }
}

get('run').addEventListener('click', run);
get('download').addEventListener('click', () => {
  if (!lastReport) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastReport, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'model-parity-report.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
