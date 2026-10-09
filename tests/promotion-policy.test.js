import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluatePromotion, scoreSummary } from '../src/js/promotion-policy.js';

const runs = scores => scores.map((score, i) => ({ seed: i + 1, score }));
const constant = value => runs(Array(20).fill(value));

test('summary uses ordinary median and interpolated tenth percentile on 20 scores', () => {
  const result = scoreSummary(runs(Array.from({ length: 20 }, (_, i) => 20 - i)));
  assert.equal(result.mean, 10.5); assert.equal(result.median, 10.5);
  assert.ok(Math.abs(result.p10 - 2.9) < 1e-12);
  assert.throws(() => scoreSummary(constant(100).slice(1)), /20/);
  const wrongSeed = constant(100); wrongSeed[2].seed = 7;
  assert.throws(() => scoreSummary(wrongSeed), /ordered/);
  assert.throws(() => scoreSummary(constant(NaN)));
});

test('each performance guard rejects independently, including exact 5% gain and zero-baseline boundaries', () => {
  const baseline = constant(100);
  assert.equal(evaluatePromotion(baseline, constant(105)).accepted, true);
  assert.deepEqual(evaluatePromotion(baseline, constant(104)).checks, { mean: false, median: true, p10: true, catastrophic: true });
  assert.deepEqual(evaluatePromotion(baseline, runs([...Array(11).fill(90), ...Array(9).fill(160)])).checks, { mean: true, median: false, p10: true, catastrophic: true });
  assert.deepEqual(evaluatePromotion(baseline, runs([80,80,80,...Array(17).fill(120)])).checks, { mean: true, median: true, p10: false, catastrophic: true });
  assert.deepEqual(evaluatePromotion(baseline, runs([40,40,...Array(18).fill(130)])).checks, { mean: true, median: true, p10: true, catastrophic: false });
  assert.equal(evaluatePromotion(constant(0), constant(0)).accepted, false);
  assert.equal(evaluatePromotion(constant(0), constant(1)).accepted, true);
});

test('severe regressions compare the same seed and need both percentage and absolute loss', () => {
  const candidate = constant(120); candidate[0].score = 40;
  const result = evaluatePromotion(constant(100), candidate);
  assert.equal(result.accepted, true);
  assert.deepEqual(result.catastrophicRegressions, [{ seed: 1, baselineScore: 100, candidateScore: 40 }]);
  const exactHalf = constant(120); exactHalf[0].score = 50;
  assert.equal(evaluatePromotion(constant(100), exactHalf).catastrophicRegressions.length, 0);
  const smallBaseline = constant(100), smallCandidate = constant(120);
  smallBaseline[0].score = 8; smallCandidate[0].score = 1;
  assert.equal(evaluatePromotion(smallBaseline, smallCandidate).catastrophicRegressions.length, 0);
});

test('recorded hard score distributions pass without requiring private training history files', () => {
  const baseline = runs([84,20,76,34,30,106,94,59,32,45,42,46,103,35,58,30,35,62,78,87]);
  const candidates = [
    [74,62,132,134,58,160,94,51,32,160,105,53,160,35,47,101,55,50,34,109],
    [74,50,109,51,58,106,60,100,62,64,81,82,88,78,58,101,59,62,115,93],
    [48,25,104,53,59,160,94,59,32,64,160,79,143,78,44,134,35,123,88,101],
  ];
  for (const [index, scores] of candidates.entries()) {
    const result = evaluatePromotion(baseline, runs(scores));
    assert.equal(result.accepted, true);
    assert.equal(result.baseline.mean, 57.8);
    if (index === 0) {
      assert.equal(result.candidate.mean, 85.3);
      assert.equal(result.candidate.median, 68);
      assert.ok(Math.abs(result.candidate.p10 - 34.9) < 1e-12);
      assert.equal(result.catastrophicRegressions[0].seed, 19);
    }
  }
});
