"""Evaluate saved JSON weights offline; this is not a gameplay benchmark."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from scripts.dataset import DataError, finite_number, load_dataset, require, select_difficulty, session_indices
from scripts.metrics import binary_losses, classification_metrics
from scripts.model import load_model, predict, write_json


def evaluate_model(model, dataset, partition='validation', threshold=None):
    require(model['normalization'] == dataset.data['normalization'], 'Model and dataset normalization disagree.')
    require(model['gameConfig'] == dataset.data['gameConfig'], 'Model and dataset game configuration disagree.')
    require(model['inputNames'] == dataset.data['inputNames'], 'Model and dataset input profiles disagree.')
    require(partition in ('train', 'validation', 'all'), 'Unknown partition.')
    training = model.get('training', {})
    dataset, difficulty = select_difficulty(dataset, training.get('difficulty', 'normal'))
    if partition != 'all':
        require(training.get('dataSha256') == dataset.sha256,
                'Dataset fingerprint differs from the training file. Use --partition all for separate external data.')
        split = training.get('split', {})
        ids = split.get('trainSessionIds' if partition == 'train' else 'validationSessionIds')
        require(isinstance(ids, list) and bool(ids), 'Missing saved split.')
        indices = session_indices(dataset, ids)
    else:
        indices = list(range(len(dataset.samples)))
    predictions = [predict(model, dataset.samples[i][:-1]) for i in indices]
    labels = [dataset.samples[i][-1] for i in indices]
    decision_threshold = model['threshold'] if threshold is None else threshold
    weight = training.get('positiveWeight', 1.0)
    require(finite_number(weight) and weight > 0, 'Invalid saved positive weight.')
    return {
        'version': 1, 'partition': partition, 'dataSha256': dataset.sha256,
        'difficulty': difficulty,
        'dataKind': dataset.data.get('dataKind', 'human-gameplay-unverified'),
        'metrics': {
            **classification_metrics(labels, [p['probability'] for p in predictions], decision_threshold),
            **binary_losses(labels, [p['logit'] for p in predictions], weight),
        },
        'baselineAlwaysWait': classification_metrics(labels, [0.0] * len(labels), decision_threshold),
        'note': 'Offline classification only; all-data evaluation may include training samples. No gameplay benchmark performed.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a JSON model on saved or external normalized states.')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/training-data.json')
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts/model-516.json')
    parser.add_argument('--partition', choices=['train', 'validation', 'all'], default='validation')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output:
            require(args.output.resolve() not in {args.data.resolve(), args.model.resolve()}, 'Output cannot overwrite the model or dataset.')
        result = evaluate_model(load_model(args.model), load_dataset(args.data), args.partition, args.threshold)
        if args.output:
            write_json(args.output, result)
        metrics = result['metrics']
        print(f"JUMP precision={metrics['jumpPrecision']:.4f}, recall={metrics['jumpRecall']:.4f}, F1={metrics['jumpF1']:.4f}")
        print(f"BCE={metrics['bce']:.6f}; weighted BCE={metrics['weightedBce']:.6f}; confusion={metrics['confusionMatrix']}")
        print(result['note'])
    except (DataError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
