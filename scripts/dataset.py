"""Validate already-normalized browser exports and split entire seed/session groups."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from .difficulties import HARD_RULES
from .profiles import profile_for_id


class DataError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DataError(message)


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def reject_constant(value):
    raise DataError(f'Non-finite JSON value: {value}')


def read_json(path):
    path = Path(path)
    try:
        require(path.stat().st_size <= 64 * 1024 * 1024, 'JSON exceeds the 64 MiB limit.')
        raw = path.read_bytes()
        data = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f'Cannot read {path}: {error}') from error
    return data, hashlib.sha256(raw).hexdigest()


@dataclass
class Dataset:
    data: dict
    sha256: str
    path: str

    @property
    def samples(self):
        return self.data['samples']

    @property
    def sessions(self):
        return self.data['sessions']


def validate_dataset(data):
    require(isinstance(data, dict), 'Dataset must be a JSON object.')
    require(type(data.get('version')) is int and data['version'] in (3, 4), 'Unsupported dataset version. Record a flight in a supported lab.')
    expected_id = 'lookahead-616' if data['version'] == 4 else 'speed-516'
    require(data.get('modelProfile') == expected_id, 'Dataset belongs to another model profile.')
    profile = profile_for_id(expected_id)
    input_names = profile['inputNames']
    require(data.get('difficultyRules') == {'hard': HARD_RULES}, 'Dataset difficulty rules do not match this simulator.')
    require(data.get('inputNames') == input_names, 'Unexpected dataset input order.')
    require(data.get('clock') == 'simulation' and data.get('sampleTiming') == 'before-action',
            'Expected simulation-clock, before-action samples.')
    require(data.get('sampleIntervalMs') == 50 and type(data['sampleIntervalMs']) is int,
            'V1 requires a 50 ms sample interval.')
    require(data.get('minimumJumpIntervalMs') == 100 and type(data['minimumJumpIntervalMs']) is int,
            'V1 requires a 100 ms jump cooldown.')
    normalization = data.get('normalization')
    require(isinstance(normalization, dict) and type(normalization.get('version')) is int and
            normalization['version'] == profile['version'], 'Invalid normalization version.')
    require(normalization.get('maxPipeSpeed') == 270, 'Invalid pipe speed normalization.')
    for key in ('height', 'velocityScale', 'maxPipeDistance'):
        require(finite_number(normalization.get(key)) and normalization[key] > 0, f'Invalid normalization.{key}')
    config = data.get('gameConfig')
    require(isinstance(config, dict) and bool(config), 'Missing gameConfig.')
    require(all(finite_number(value) for value in config.values()), 'gameConfig must contain finite numbers.')
    for key in ('width', 'height', 'birdRadius', 'gravity', 'pipeSpeed', 'pipeWidth', 'pipeGap',
                'pipeSpacing', 'fixedStepSeconds', 'maxTrainingSamples'):
        require(config.get(key, 0) > 0, f'Invalid gameConfig.{key}')
    require(config['height'] == normalization['height'], 'Game height and normalization disagree.')
    require(config.get('decisionIntervalMs') == data['sampleIntervalMs'] and
            config.get('minimumJumpIntervalMs') == data['minimumJumpIntervalMs'], 'Game timing and dataset timing disagree.')
    require(config.get('jumpVelocity', 0) < 0, 'Jump velocity must be negative.')
    require(integer(config['maxTrainingSamples'], 1), 'Invalid sample limit.')
    samples = data.get('samples')
    require(isinstance(samples, list) and 0 < len(samples) <= min(config['maxTrainingSamples'], 100000),
            'Dataset must contain 1 to 100000 samples within its configured limit.')
    for index, row in enumerate(samples):
        require(isinstance(row, list) and len(row) == len(input_names) + 1, f'Sample {index}: unexpected input/action count.')
        require(all(finite_number(value) for value in row), f'Sample {index}: invalid numeric value.')
        require(0 <= row[0] <= 1 and -1 <= row[1] <= 1 and 0 <= row[2] <= 1 and 0 <= row[3] <= 1,
                f'Sample {index}: input outside normalized range.')
        require(0 <= row[4] <= 1, f'Sample {index}: invalid normalized pipe speed.')
        if data['version'] == 4:
            require(0 <= row[5] <= 1, f'Sample {index}: invalid following pipe gap.')
        require(type(row[-1]) is int and row[-1] in (0, 1), f'Sample {index}: action must be integer 0 or 1.')
    counts = data.get('counts')
    jumps = sum(row[-1] for row in samples)
    require(isinstance(counts, dict) and all(integer(counts.get(k)) for k in ('total', 'jump', 'wait')),
            'Missing or invalid counters.')
    require(counts['total'] == len(samples) and counts['jump'] == jumps and counts['wait'] == len(samples) - jumps,
            'Counters disagree with samples.')
    sessions = data.get('sessions')
    require(isinstance(sessions, list) and bool(sessions), 'Missing sessions; frame-level random splitting is not supported.')
    offset, ids = 0, set()
    for session in sessions:
        require(isinstance(session, dict), 'Invalid session object.')
        difficulty = session.get('difficulty', 'normal')
        require(difficulty in ('normal', 'hard'), 'Unknown recording difficulty.')
        require('difficulty' in session, 'Sessions must declare their difficulty.')
        session_id = session.get('id')
        require(integer(session_id, 1) and session_id not in ids, 'Session IDs must be unique positive integers.')
        ids.add(session_id)
        require(integer(session.get('seed')) and session['seed'] <= 4294967295, f'Session {session_id}: invalid seed.')
        require(integer(session.get('startIndex')) and session['startIndex'] == offset,
                f'Session {session_id}: sample ranges must be contiguous and non-overlapping.')
        require(integer(session.get('sampleCount'), 1), f'Session {session_id}: empty or invalid range.')
        require(integer(session.get('score')), f'Session {session_id}: invalid score.')
        duration = session.get('elapsedSeconds')
        require(finite_number(duration) and duration >= 0, f'Session {session_id}: invalid duration.')
        require(session['sampleCount'] <= math.floor((duration + 1e-7) * 1000 / data['sampleIntervalMs']) + 1,
                f'Session {session_id}: too many samples for its duration.')
        require(session.get('endReason') in ('collision', 'restart', 'stopped', 'in-progress'),
                f'Session {session_id}: unfinished or unknown end reason.')
        offset += session['sampleCount']
    require(offset == len(samples), 'Session ranges must cover all samples exactly.')
    require(data.get('dataKind', 'human-gameplay-unverified') in ('human-gameplay-unverified', 'synthetic-test'),
            'Unknown dataKind.')
    return data


def load_dataset(path):
    data, digest = read_json(path)
    return Dataset(validate_dataset(data), digest, str(Path(path).resolve()))


def select_difficulty(dataset, difficulty=None):
    """Filter entire flights, preserving IDs and the source file fingerprint."""
    available = {session.get('difficulty', 'normal') for session in dataset.sessions}
    if difficulty is None:
        require(len(available) == 1, 'Mixed recordings: choose --difficulty normal or --difficulty hard.')
        difficulty = next(iter(available))
    require(difficulty in ('normal', 'hard'), 'Unknown training difficulty.')
    require(difficulty in available, f'No {difficulty} recordings in this dataset.')
    samples, sessions = [], []
    for session in dataset.sessions:
        if session.get('difficulty', 'normal') != difficulty:
            continue
        rows = dataset.samples[session['startIndex']:session['startIndex'] + session['sampleCount']]
        sessions.append({**session, 'difficulty': difficulty, 'startIndex': len(samples)})
        samples.extend(rows)
    jumps = sum(row[-1] for row in samples)
    data = {**dataset.data, 'samples': samples, 'sessions': sessions,
            'counts': {'total': len(samples), 'jump': jumps, 'wait': len(samples) - jumps}}
    return Dataset(data, dataset.sha256, dataset.path), difficulty


def session_indices(dataset, ids):
    selected = set(ids)
    require(selected <= {session['id'] for session in dataset.sessions}, 'Unknown session in split.')
    return [index for session in dataset.sessions if session['id'] in selected
            for index in range(session['startIndex'], session['startIndex'] + session['sampleCount'])]


def split_dataset(dataset, validation_fraction=0.2, seed=12345, split_by='seed', min_samples=200):
    require(len(dataset.samples) >= min_samples,
            f'At least {min_samples} samples required; got {len(dataset.samples)}. Collect more human gameplay.')
    require(0 < validation_fraction < 1, 'Validation fraction must be between 0 and 1.')
    require(split_by in ('seed', 'session'), 'split_by must be seed or session.')
    groups = {}
    for session in dataset.sessions:
        key = session['seed'] if split_by == 'seed' else session['id']
        groups.setdefault(key, []).append(session['id'])
    require(len(groups) >= 2, f'Need at least two distinct {split_by} groups. Collect additional runs with different seeds.')
    stats = {}
    for session in dataset.sessions:
        key = session['seed'] if split_by == 'seed' else session['id']
        rows = dataset.samples[session['startIndex']:session['startIndex'] + session['sampleCount']]
        count, jumps = stats.get(key, (0, 0))
        stats[key] = (count + len(rows), jumps + sum(row[-1] for row in rows))
    total, total_jumps = len(dataset.samples), dataset.data['counts']['jump']
    require(total_jumps > 0 and total_jumps < total, 'Both JUMP and WAIT samples are required.')
    validation_count = max(1, min(len(groups) - 1, math.ceil(len(groups) * validation_fraction)))
    randomizer = random.Random(seed)
    best = None
    for _ in range(1000):
        candidate = randomizer.sample(sorted(groups), validation_count)
        size = sum(stats[key][0] for key in candidate)
        jumps = sum(stats[key][1] for key in candidate)
        if min(jumps, size - jumps, total_jumps - jumps, total - size - total_jumps + jumps) < 2:
            continue
        deviation = abs(size / total - validation_fraction)
        if best is None or deviation < best[0]:
            best = (deviation, candidate)
    require(best is not None, 'Cannot split groups with at least two JUMP and two WAIT samples on each side. Collect more varied sessions.')
    validation_ids = sorted(session_id for key in best[1] for session_id in groups[key])
    train_ids = sorted(session_id for key in groups if key not in best[1] for session_id in groups[key])
    return {
        'by': split_by, 'seed': seed, 'requestedValidationFraction': validation_fraction,
        'trainSessionIds': train_ids, 'validationSessionIds': validation_ids,
        'actualValidationFraction': sum(stats[key][0] for key in best[1]) / total,
    }
