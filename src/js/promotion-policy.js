// Performance policy only: CSS parity and simulator agreement are checked separately.
export const PROMOTION_POLICY = Object.freeze({
  version: 2,
  minimumMeanMultiplier: 1.05,
  minimumP10Multiplier: 0.90,
  catastrophicScoreRatio: 0.50,
  catastrophicMinimumPointLoss: 10,
  maximumCatastrophicRegressions: 1,
  quantileMethod: 'linear interpolation at (n - 1) * p',
});

export function scoreSummary(runs) {
  if (!Array.isArray(runs) || runs.length !== 20 || runs.some((run, i) => run.seed !== i + 1 || !Number.isInteger(run.score) || run.score < 0)) {
    throw new Error('Acceptance requires scores for the same 20 ordered seeds.');
  }
  const scores = runs.map(run => run.score).sort((a, b) => a - b);
  const quantile = p => {
    const position = (scores.length - 1) * p;
    const low = Math.floor(position), high = Math.ceil(position);
    return scores[low] + (scores[high] - scores[low]) * (position - low);
  };
  return { mean: scores.reduce((sum, value) => sum + value, 0) / scores.length, median: quantile(.5), p10: quantile(.1) };
}

export function evaluatePromotion(baselineRuns, candidateRuns) {
  const baseline = scoreSummary(baselineRuns), candidate = scoreSummary(candidateRuns);
  const policy = PROMOTION_POLICY;
  const atLeast = (actual, required) => actual >= required || Math.abs(actual - required) <= Number.EPSILON * Math.max(1, Math.abs(actual), Math.abs(required)) * 8;
  const catastrophicRegressions = baselineRuns.flatMap((run, i) => {
    const next = candidateRuns[i].score;
    return next < run.score * policy.catastrophicScoreRatio && run.score - next >= policy.catastrophicMinimumPointLoss
      ? [{ seed: run.seed, baselineScore: run.score, candidateScore: next }] : [];
  });
  // A zero baseline must still improve; 0 >= 0 alone cannot qualify a dead model.
  const checks = {
    mean: candidate.mean > baseline.mean && atLeast(candidate.mean, baseline.mean * policy.minimumMeanMultiplier),
    median: candidate.median >= baseline.median,
    p10: atLeast(candidate.p10, baseline.p10 * policy.minimumP10Multiplier),
    catastrophic: catastrophicRegressions.length <= policy.maximumCatastrophicRegressions,
  };
  const reasons = [];
  if (!checks.mean) reasons.push('Average must improve by at least 5%.');
  if (!checks.median) reasons.push('Median must not decrease.');
  if (!checks.p10) reasons.push('P10 must retain at least 90% of its previous score.');
  if (!checks.catastrophic) reasons.push('At most one course may lose over 50% and at least 10 points.');
  return { accepted: Object.values(checks).every(Boolean), policy, baseline, candidate, checks, catastrophicRegressions, reasons };
}
