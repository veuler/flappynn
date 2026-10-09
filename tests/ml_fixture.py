"""Artificial separable data for pipeline tests, never a gameplay demonstration."""
import argparse
import json
from pathlib import Path
import random


def make_fixture():
    # Read the actual JS-exported contract instead of copying game constants into tests.
    # A compact contract fixture is supplied by export_contract.js.
    contract = json.loads((Path(__file__).parent / 'fixtures/training-contract.json').read_text(encoding='utf-8'))
    randomizer = random.Random(73)
    samples, sessions = [], []
    for session_id in range(1, 9):
        start = len(samples)
        for _ in range(80):
            y = randomizer.uniform(0.15, 0.85)
            velocity = randomizer.uniform(-0.5, 0.7)
            distance = randomizer.uniform(0, 1)
            gap = randomizer.uniform(0.25, 0.75)
            action = int(y - gap + 0.25 * velocity > 0.2)
            samples.append([y, velocity, distance, gap, 2/3, action])
        sessions.append({'id': session_id, 'seed': 100 + session_id, 'difficulty': 'normal', 'startIndex': start,
                         'sampleCount': 80, 'elapsedSeconds': 4.0, 'score': 0, 'endReason': 'stopped',
                         'createdAt': '2026-10-07T00:00:00Z'})
    return {
        **contract, 'dataKind': 'synthetic-test', 'samples': samples, 'sessions': sessions,
        'counts': {'total': len(samples), 'jump': sum(row[-1] for row in samples),
                   'wait': sum(row[-1] == 0 for row in samples)},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic pipeline-test data; not human gameplay.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(make_fixture(), indent=2) + '\n', encoding='utf-8')
    print(f'Synthetic test data written: {args.output}')
