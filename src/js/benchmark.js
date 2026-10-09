import { CONFIG } from './config.js';
import { Game } from './game.js';
import { TrainingRecorder } from './training-recorder.js';
import { SessionController } from './session-controller.js';
import { AiController } from './ai-controller.js';
import { modelProfile } from './model-profiles.js';

const get = (id) => document.getElementById(id);
const profile = modelProfile(new URL(location.href).searchParams.get('profile') || 'speed-516');
const ai = new AiController(get('ai-output'), profile.base, profile.id);
let lastReport = null;

async function benchmark() {
  get('benchmark-run').disabled = true;
  get('benchmark-download').disabled = true;
  try {
    ai.setThreshold(Number(get('benchmark-threshold').value));
    const runs = [];
    const game = new Game();
    const recorder = new TrainingRecorder({profileId:profile.id});
    const session = new SessionController(game, recorder, ai);
    for (let seed = 1; seed <= 20; seed++) {
      session.start(seed, 'ai');
      for (let tick = 0; tick < 60 / CONFIG.fixedStepSeconds && game.status === 'running'; tick++) session.step();
      if (game.status === 'paused') throw new Error(ai.error || 'CSS inference stopped.');
      runs.push({ seed, score: game.score, survivalSeconds: game.elapsedSeconds,
        endedBy: game.status === 'gameover' ? 'collision' : 'time-limit', decisions: ai.decisionCount });
      session.finish();
      get('benchmark-status').textContent = `${seed} / 20 flights completed`;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const sortedScores = runs.map((run) => run.score).sort((a, b) => a - b);
    lastReport = {
      version: 1, checkedAt: new Date().toISOString(), browser: navigator.userAgent,
      modelSha256: ai.metadata.modelSha256, cssSha256: ai.metadata.cssSha256,
      dataKind: ai.metadata.dataKind, threshold: ai.threshold,
      architecture: profile.architecture, modelProfile: profile.id,
      seedRange: [1, 20], runCount: 20, maximumSimulationSeconds: 60,
      meanScore: runs.reduce((sum, run) => sum + run.score, 0) / 20,
      medianScore: (sortedScores[9] + sortedScores[10]) / 2,
      maxScore: sortedScores.at(-1),
      meanSurvivalSeconds: runs.reduce((sum, run) => sum + run.survivalSeconds, 0) / 20,
      timeLimitedRuns: runs.filter((run) => run.endedBy === 'time-limit').length,
      samplesRecorded: recorder.samples.length,
      inferenceSource: 'CSS computed numeric properties', runs,
    };
    get('benchmark-status').textContent = 'Benchmark completed';
    get('benchmark-results').textContent = `Mean score: ${lastReport.meanScore.toFixed(2)} · Median: ${lastReport.medianScore} · Best: ${lastReport.maxScore} · Mean time: ${lastReport.meanSurvivalSeconds.toFixed(1)} s · Time-limited runs: ${lastReport.timeLimitedRuns}`;
    get('benchmark-report').textContent = JSON.stringify(lastReport, null, 2);
    get('benchmark-download').disabled = false;
  } catch (error) {
    get('benchmark-status').textContent = error.message;
  } finally {
    get('benchmark-run').disabled = !ai.ready;
  }
}

get('benchmark-run').addEventListener('click', benchmark);
get('benchmark-download').addEventListener('click', () => {
  if (!lastReport) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastReport, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url; link.download = 'css-gameplay-evaluation.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
ai.initialize().then((ready) => {
  get('benchmark-status').textContent = ready ? 'Verified CSS model ready' : ai.error;
  get('benchmark-run').disabled = !ready;
  get('benchmark-threshold').value = ai.threshold;
});
