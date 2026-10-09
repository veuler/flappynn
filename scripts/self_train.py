"""Time-bounded local neuroevolution. Deployment still uses CSS inference only."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.model import load_model, validate_model, write_json
from scripts.export_css import export_bundle
from scripts.difficulties import HARD_RULES


def random_sequence(seed, count):
    """Exact unsigned Mulberry32 sequence used by src/js/pipes.js."""
    state = int(seed) & 0xffffffff
    values = []
    for _ in range(count):
        state = (state + 0x6D2B79F5) & 0xffffffff
        value = ((state ^ (state >> 15)) * (state | 1)) & 0xffffffff
        value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & 0xffffffff)) & 0xffffffff
        values.append(((value ^ (value >> 14)) & 0xffffffff) / 4294967296)
    return values


class FlightBatch:
    """Vectorized replica of Game; no rendering, changed physics or teacher policy."""
    def __init__(self, seeds, config, seconds, difficulty='normal'):
        if difficulty not in ('normal', 'hard'):
            raise ValueError('Invalid difficulty.')
        self.difficulty = difficulty
        self.config = config
        self.seeds = np.asarray(seeds, dtype=np.uint32)
        self.y = np.full(len(seeds), float(config['initialBirdY']))
        self.velocity = np.full(len(seeds), float(config['jumpVelocity']))
        self.alive = np.ones(len(seeds), dtype=bool)
        self.score = np.zeros(len(seeds), dtype=int)
        self.elapsed = np.zeros(len(seeds))
        self.last_jump = np.zeros(len(seeds), dtype=int)
        self.tick = 0
        self.speed = np.full(len(seeds), float(config['pipeSpeed']))
        self.first_x = np.full(len(seeds), float(config['firstPipeX']))
        count = math.ceil(seconds * config['pipeSpeed'] * (HARD_RULES['maxSpeedMultiplier'] if difficulty == 'hard' else 1) / config['pipeSpacing']) + 8
        low = config['gapMargin'] + config['pipeGap'] / 2
        self.gaps = np.array([random_sequence(seed, count) for seed in seeds]) * (config['height'] - 2 * low) + low

    def step(self):
        c = self.config
        dt = c['fixedStepSeconds']
        self.tick += 1
        self.elapsed[self.alive] += dt
        self.velocity[self.alive] += c['gravity'] * dt
        self.y[self.alive] += self.velocity[self.alive] * dt
        if self.difficulty == 'hard':
            target = c['pipeSpeed'] * np.minimum(1 + self.score * HARD_RULES['speedGrowthPerPoint'], HARD_RULES['maxSpeedMultiplier'])
            self.speed[self.alive] += (target[self.alive] - self.speed[self.alive]) * min(dt / HARD_RULES['speedResponseSeconds'], 1)
            self.first_x[self.alive] -= self.speed[self.alive] * dt
            first_x = self.first_x
            index = np.maximum(0, np.ceil((c['birdX'] - c['birdRadius'] - first_x - c['pipeWidth']) / c['pipeSpacing'])).astype(int)
        else:
            first_x = c['firstPipeX'] - self.tick * c['pipeSpeed'] * dt
            index = max(0, math.ceil((c['birdX'] - c['birdRadius'] - first_x - c['pipeWidth']) / c['pipeSpacing']))
        x = first_x + index * c['pipeSpacing']
        gap = self.gaps[np.arange(len(self.y)), index] if self.difficulty == 'hard' else self.gaps[:, index]
        self.following_gap = self.gaps[np.arange(len(self.y)), index + 1] if self.difficulty == 'hard' else self.gaps[:, index + 1]
        nearest_x = np.clip(c['birdX'], x, x + c['pipeWidth'])
        horizontal = (c['birdX'] - nearest_x) ** 2
        top = gap - c['pipeGap'] / 2
        bottom = gap + c['pipeGap'] / 2
        top_y = np.clip(self.y, 0, top)
        bottom_y = np.clip(self.y, bottom, c['height'])
        hit = ((self.y - c['birdRadius'] <= 0) | (self.y + c['birdRadius'] >= c['height']) |
               (horizontal + (self.y - top_y) ** 2 <= c['birdRadius'] ** 2) |
               (horizontal + (self.y - bottom_y) ** 2 <= c['birdRadius'] ** 2))
        self.alive &= ~hit
        self.score[self.alive] = index[self.alive] if self.difficulty == 'hard' else index
        return np.column_stack((np.clip(self.y / c['height'], 0, 1),
                                np.clip(self.velocity / 1200, -1, 1),
                                np.broadcast_to(np.clip((x - c['birdX']) / 600, 0, 1), (len(self.y),)),
                                gap / c['height']))

    def jump(self, requested):
        cooled = (self.tick - self.last_jump) * self.config['fixedStepSeconds'] * 1000 >= self.config['minimumJumpIntervalMs'] - 1e-8
        action = requested & cooled & self.alive
        self.velocity[action] = self.config['jumpVelocity']
        self.last_jump[action] = self.tick


def pack(model):
    return np.concatenate((np.array(model['w1']).ravel(), model['b1'], model['w2'], [model['b2']]))


def unpack(parameters, template):
    model = copy.deepcopy(template)
    inputs, hidden, _ = template['architecture']
    matrix = inputs * hidden
    model.update(w1=parameters[:matrix].reshape(hidden, inputs).tolist(), b1=parameters[matrix:matrix + hidden].tolist(),
                 w2=parameters[matrix + hidden:matrix + hidden * 2].tolist(), b2=float(parameters[matrix + hidden * 2]))
    return validate_model(model)


def rollout(parameters, seeds, config, threshold, seconds=60, interrupted=None, difficulty='normal', architecture=None):
    parameters = np.atleast_2d(parameters)
    if architecture is None:
        architecture = {113: [5, 16, 1], 129: [6, 16, 1]}.get(parameters.shape[1])
    if architecture not in ([5, 16, 1], [6, 16, 1]):
        raise ValueError('Unsupported architecture or parameter count.')
    inputs_count, hidden_count, _ = architecture
    matrix = inputs_count * hidden_count
    if parameters.shape[1] != matrix + hidden_count * 2 + 1:
        raise ValueError('Parameter count disagrees with architecture.')
    population, count = len(parameters), len(seeds)
    parameters = np.repeat(parameters, count, axis=0)
    flight = FlightBatch(np.tile(seeds, population), config, seconds, difficulty)
    w1 = parameters[:, :matrix].reshape(-1, hidden_count, inputs_count)
    b1, w2, b2 = parameters[:, matrix:matrix + hidden_count], parameters[:, matrix + hidden_count:matrix + hidden_count * 2], parameters[:, -1]
    cutoff = math.log(threshold / (1 - threshold))
    interval = round(config['decisionIntervalMs'] / 1000 / config['fixedStepSeconds'])
    for tick in range(1, round(seconds / config['fixedStepSeconds']) + 1):
        if tick % 60 == 0 and interrupted and interrupted():
            return None
        inputs = flight.step()
        if inputs_count >= 5:
            inputs = np.column_stack((inputs, np.clip(flight.speed / 270, 0, 1)))
        if inputs_count == 6:
            inputs = np.column_stack((inputs, flight.following_gap / config['height']))
        if not flight.alive.any():
            break
        if tick % interval == 0:
            hidden = np.maximum(0, np.einsum('nhi,ni->nh', w1, inputs) + b1)
            logit = np.einsum('nh,nh->n', hidden, w2) + b2
            flight.jump(logit >= cutoff)
    return flight.score.reshape(population, count), flight.elapsed.reshape(population, count)


def fitness(result):
    scores, survival = result
    return scores.mean(axis=1) + 0.01 * survival.mean(axis=1) + 0.05 * scores.min(axis=1)


def summarize(result, seeds):
    scores, seconds = result
    return {'meanScore': float(scores[0].mean()), 'minimumScore': int(scores[0].min()),
            'meanSurvivalSeconds': float(seconds[0].mean()),
            'runs': [{'seed': int(seed), 'score': int(score), 'survivalSeconds': float(duration)}
                     for seed, score, duration in zip(seeds, scores[0], seconds[0])]}


def train(model_path, output, seconds=300, threshold=0.62, seed=20261007, difficulty='normal'):
    model_path, output = Path(model_path), Path(output)
    if not 1 <= seconds <= 1800 or not 0 < threshold < 1:
        raise ValueError('Invalid duration or threshold.')
    if difficulty not in ('normal', 'hard'):
        raise ValueError('Invalid difficulty.')
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path)
    if model['gameConfig']['fixedStepSeconds'] * model['gameConfig']['pipeSpeed'] != 1.5:
        raise ValueError('This simulator requires the current 1.5 pixel pipe step.')
    model['threshold'] = threshold
    original = pack(model)
    champion, center = original.copy(), original.copy()
    rng = np.random.default_rng(seed)
    validation_seeds = [12345] + rng.integers(100000, 0xffffffff, size=7, dtype=np.uint32).tolist()
    seen = set(validation_seeds + list(range(1, 21)))
    start = time.monotonic()
    deadline = start + seconds
    stop = output / 'stop'
    interrupted = lambda: time.monotonic() >= deadline or stop.exists()
    validation = rollout(champion, validation_seeds, model['gameConfig'], threshold, seconds=180, difficulty=difficulty)
    baseline_validation = summarize(validation, validation_seeds)
    champion_fitness = float(fitness(validation)[0])
    generation = evaluations = accepted = 0
    history = []

    def progress(state):
        record = {'state': state, 'difficulty': difficulty, 'elapsedSeconds': min(time.monotonic() - start, seconds),
                  'durationSeconds': seconds, 'generation': generation, 'episodes': evaluations,
                  'acceptedUpdates': accepted, 'validationMeanScore': float(validation[0].mean()),
                  'baselineValidationMeanScore': baseline_validation['meanScore']}
        write_json(output / 'status.json', record)
        print(json.dumps(record), flush=True)

    progress('running')
    while not interrupted():
        seeds = []
        while len(seeds) < 6:
            candidate_seed = int(rng.integers(100000, 0xffffffff))
            if candidate_seed not in seen:
                seen.add(candidate_seed)
                seeds.append(candidate_seed)
        scale = [0.04, 0.02, 0.01, 0.006][generation % 4]
        population = np.repeat(center[None, :], 26, axis=0)
        population[13:] = champion
        population += rng.normal(size=population.shape) * scale * (0.5 + np.abs(population))
        population[0], population[1] = champion, center
        result = rollout(population, seeds, model['gameConfig'], threshold, seconds=120, interrupted=interrupted, difficulty=difficulty)
        if result is None:
            break
        generation += 1
        evaluations += len(population) * len(seeds)
        ranking = np.argsort(fitness(result))[::-1]
        center = population[ranking[:4]].mean(axis=0)
        # Validation selects checkpoints; seeds 1..20 are never used to search.
        candidates = np.vstack((champion, population[ranking[:3]], center))
        checked = rollout(candidates, validation_seeds, model['gameConfig'], threshold, seconds=180, interrupted=interrupted, difficulty=difficulty)
        if checked is None:
            break
        evaluations += len(candidates) * len(validation_seeds)
        best = int(np.argmax(fitness(checked)))
        if fitness(checked)[best] > champion_fitness + 1e-8:
            champion, champion_fitness = candidates[best].copy(), float(fitness(checked)[best])
            validation = (checked[0][best:best + 1], checked[1][best:best + 1])
            accepted += 1
            write_json(output / 'checkpoint.json', unpack(champion, model))
        history.append({'generation': generation, 'meanScore': float(validation[0].mean()), 'elapsedSeconds': time.monotonic() - start})
        progress('running')
    training_elapsed = time.monotonic() - start
    progress('evaluating')
    test_seeds = list(range(1, 21))
    base_test = rollout(original, test_seeds, model['gameConfig'], threshold, seconds=180, difficulty=difficulty)
    candidate_test = rollout(champion, test_seeds, model['gameConfig'], threshold, seconds=180, difficulty=difficulty)
    candidate = unpack(champion, model)
    candidate['createdAt'] = datetime.now(timezone.utc).isoformat()
    candidate['training'] = {'dataKind': 'self-play-neuroevolution', 'method': 'elitist Gaussian parameter search',
        'parentModelSha256': hashlib.sha256(model_path.read_bytes()).hexdigest(), 'seed': seed,
        'trainingSeconds': training_elapsed, 'requestedSeconds': seconds, 'generations': generation,
        'episodes': evaluations, 'acceptedUpdates': accepted, 'threshold': threshold, 'difficulty': difficulty,
        'validationSeeds': validation_seeds, 'trainingSeedsCount': len(seen) - 28,
        'reward': 'mean pipes + 0.01 * mean survival seconds + 0.05 * minimum pipes',
        'checkpointSelection': 'validation fitness; final test seeds excluded from search',
        'numpy': np.__version__}
    if difficulty == 'hard':
        candidate['training']['difficultyConfig'] = dict(HARD_RULES)
    report = {'version': 1, 'state': 'awaiting-validation', 'difficulty': difficulty, 'stoppedEarly': stop.exists(),
              'training': candidate['training'], 'baselineValidation': baseline_validation,
              'candidateValidation': summarize(validation, validation_seeds),
              'baselineTest': summarize(base_test, test_seeds), 'candidateTest': summarize(candidate_test, test_seeds),
              'history': history, 'maximumSimulationSeconds': 180,
              'limitations': ['CPU simulation must match the browser; browser CSS gameplay is the deployment gate.',
                             'Validation selected the checkpoint and is not an untouched test set.',
                             'Five minutes need not improve the model; only a better verified model is deployed.']}
    write_json(output / 'candidate.json', candidate)
    write_json(output / 'report.json', report)
    export_bundle(output / 'candidate.json', output / 'model.css', output / 'parity-inputs.json', output / 'model.meta.json')
    progress('awaiting-validation')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--threshold', type=float, default=0.62)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--difficulty', choices=['normal', 'hard'], default='normal')
    args = parser.parse_args()
    try:
        train(args.model, args.output, args.seconds, args.threshold, args.seed, args.difficulty)
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
