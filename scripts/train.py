"""Train the v1 MLP on CPU; browser inference remains entirely outside Python."""
import argparse
import copy
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from scripts.dataset import DataError, finite_number, load_dataset, require, select_difficulty, session_indices, split_dataset
from scripts.difficulties import HARD_RULES
from scripts.profiles import profile_for_id
from scripts.metrics import binary_losses, classification_metrics
from scripts.model import load_model, predict, validate_model, write_json


@dataclass(frozen=True)
class TrainingOptions:
    epochs: int = 400
    patience: int = 40
    learning_rate: float = 0.01
    batch_size: int = 128
    seed: int = 12345
    initialization_seed: int | None = None
    validation_fraction: float = 0.2
    split_by: str = 'seed'
    min_samples: int = 200
    threshold: float = 0.5
    difficulty: str | None = None


def train_dataset(dataset, options=TrainingOptions(), progress=None, initial_model=None, should_stop=None, progress_interval=25):
    dataset, difficulty = select_difficulty(dataset, options.difficulty)
    profile = profile_for_id(dataset.data['modelProfile'])
    input_count, hidden_count, _ = profile['architecture']
    for key in ('epochs', 'patience', 'batch_size', 'min_samples'):
        require(type(getattr(options, key)) is int and getattr(options, key) > 0, f'{key} must be a positive integer.')
    require(type(options.seed) is int and 0 <= options.seed <= 4294967295, 'Seed must be an unsigned 32-bit integer.')
    initialization_seed = options.seed if options.initialization_seed is None else options.initialization_seed
    require(type(initialization_seed) is int and 0 <= initialization_seed <= 4294967295,
            'Initialization seed must be an unsigned 32-bit integer.')
    require(finite_number(options.learning_rate) and options.learning_rate > 0, 'Learning rate must be positive and finite.')
    require(finite_number(options.threshold) and 0 < options.threshold < 1, 'Threshold must be between 0 and 1.')
    split = split_dataset(dataset, options.validation_fraction, options.seed, options.split_by, options.min_samples)
    try:
        import numpy as np
        import torch
        from torch import nn
    except ImportError as error:
        raise DataError('Install offline training dependencies with .venv/Scripts/python -m pip install -r requirements.txt.') from error

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(initialization_seed)
    data = np.asarray(dataset.samples, dtype=np.float32)
    train_indices = session_indices(dataset, split['trainSessionIds'])
    validation_indices = session_indices(dataset, split['validationSessionIds'])
    train_x = torch.from_numpy(data[train_indices, :input_count])
    train_y = torch.from_numpy(data[train_indices, -1])
    validation_x = torch.from_numpy(data[validation_indices, :input_count])
    validation_y = torch.from_numpy(data[validation_indices, -1])
    positives = int(train_y.sum().item())
    positive_weight = (len(train_y) - positives) / positives
    network = nn.Sequential(nn.Linear(input_count, hidden_count), nn.ReLU(), nn.Linear(hidden_count, 1))
    initialization = {'initialization': 'random'}
    if initial_model is not None:
        source = load_model(initial_model)
        require(source['architecture'] == profile['architecture'] and source['inputNames'] == dataset.data['inputNames'] and
                source['normalization'] == dataset.data['normalization'] and source['gameConfig'] == dataset.data['gameConfig'],
                'Initial model and recording profiles or physics disagree.')
        random_inputs = network[0].weight.detach().clone()
        dormant = [i for i in range(hidden_count) if source['w2'][i] == 0 and source['b1'][i] == 0 and all(v == 0 for v in source['w1'][i])]
        with torch.no_grad():
            network[0].weight.copy_(torch.tensor(source['w1']))
            network[0].bias.copy_(torch.tensor(source['b1']))
            network[2].weight.copy_(torch.tensor([source['w2']]))
            network[2].bias.copy_(torch.tensor([source['b2']]))
            # Zero output weights preserve the starting function while making unused ReLU units learnable.
            for i in dormant:
                network[0].weight[i].copy_(random_inputs[i] * 0.1)
                network[0].bias[i] = 0.05
        initialization = {'initialization': 'fine-tune', 'parentModelSha256': hashlib.sha256(Path(initial_model).read_bytes()).hexdigest(),
                          'activatedDormantNeurons': dormant}
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.Adam(network.parameters(), lr=options.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(options.seed)
    best_loss, best_epoch, best_state = float('inf'), 0, None
    history = []
    for epoch in range(1, options.epochs + 1):
        network.train()
        order = torch.randperm(len(train_y), generator=generator)
        for indices in order.split(options.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = network(train_x[indices]).squeeze(1)
            loss = loss_function(logits, train_y[indices])
            require(torch.isfinite(loss).item(), 'Non-finite training loss; reduce the learning rate.')
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            train_loss = loss_function(network(train_x).squeeze(1), train_y).item()
            validation_loss = loss_function(network(validation_x).squeeze(1), validation_y).item()
        require(finite_number(train_loss) and finite_number(validation_loss), 'Non-finite loss; model not exported.')
        history.append({'epoch': epoch, 'trainingWeightedBce': train_loss, 'validationWeightedBce': validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(network.state_dict())
        if progress and (epoch == 1 or epoch % progress_interval == 0):
            progress(epoch, train_loss, validation_loss)
        if epoch - best_epoch >= options.patience:
            break
        if should_stop and should_stop():
            break

    network.load_state_dict(best_state)
    created = datetime.now(timezone.utc).isoformat()
    provenance = {
        **initialization,
        'dataSha256': dataset.sha256, 'dataFile': Path(dataset.path).name,
        'dataKind': dataset.data.get('dataKind', 'human-gameplay-unverified'),
        'difficulty': difficulty,
        'seed': options.seed, 'initializationSeed': initialization_seed, 'epochsRun': epoch, 'bestEpoch': best_epoch,
        'optimizer': 'Adam', 'learningRate': options.learning_rate, 'weightDecay': 1e-4,
        'batchSize': options.batch_size, 'patience': options.patience,
        'loss': 'BCEWithLogitsLoss', 'positiveWeight': positive_weight,
        'checkpointSelection': 'minimum validation weighted BCE',
        'framework': {'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version.split()[0]},
        'split': split,
    }
    if difficulty == 'hard':
        provenance['difficultyConfig'] = dict(HARD_RULES)
    model = validate_model({
        'version': profile['version'], 'createdAt': created, 'architecture': profile['architecture'],
        'modelProfile': profile['id'],
        'activation': 'relu', 'outputActivation': 'sigmoid',
        'inputNames': dataset.data['inputNames'],
        'w1': network[0].weight.detach().tolist(), 'b1': network[0].bias.detach().tolist(),
        'w2': network[2].weight.detach().tolist()[0], 'b2': network[2].bias.detach().item(),
        'threshold': options.threshold, 'normalization': dataset.data['normalization'],
        'gameConfig': dataset.data['gameConfig'], 'sampleIntervalMs': dataset.data['sampleIntervalMs'],
        'minimumJumpIntervalMs': dataset.data['minimumJumpIntervalMs'], 'training': provenance,
    })

    def summarize(indices):
        rows = [dataset.samples[i] for i in indices]
        labels = [row[-1] for row in rows]
        predictions = [predict(model, row[:input_count]) for row in rows]
        return {
            **classification_metrics(labels, [p['probability'] for p in predictions], options.threshold),
            **binary_losses(labels, [p['logit'] for p in predictions], positive_weight),
        }

    training_metrics, validation_metrics = summarize(train_indices), summarize(validation_indices)
    validation_labels = [dataset.samples[i][-1] for i in validation_indices]
    warnings = []
    train_seeds = {s['seed'] for s in dataset.sessions if s['id'] in split['trainSessionIds']}
    validation_seeds = {s['seed'] for s in dataset.sessions if s['id'] in split['validationSessionIds']}
    if train_seeds & validation_seeds:
        warnings.append('Identical pipe seeds occur in training and validation; generalization may be overstated.')
    if len(dataset.samples) < 1000 or validation_metrics['jumpCount'] < 20:
        warnings.append('Small dataset or few validation jumps: metrics are unstable; collect more varied human sessions.')
    if validation_metrics['predictedJumps'] in (0, validation_metrics['sampleCount']):
        warnings.append('Validation predicts only one action at this threshold.')
    if validation_metrics['predictionMax'] - validation_metrics['predictionMin'] < 1e-3:
        warnings.append('Output is nearly constant on validation states.')
    if provenance['dataKind'] == 'synthetic-test':
        warnings.append('Synthetic test data proves pipeline operation only; it is not human gameplay or agent performance.')
    if any(s['endReason'] == 'in-progress' for s in dataset.sessions):
        warnings.append('Dataset includes a snapshot of an incomplete session.')
    report = {
        'version': 1, 'createdAt': created, 'architecture': profile['architecture'], 'parameterCount': input_count * hidden_count + 2 * hidden_count + 1,
        'training': provenance, 'train': training_metrics, 'validation': validation_metrics,
        'baselines': {
            'alwaysWait': classification_metrics(validation_labels, [0.0] * len(validation_labels), options.threshold),
            'alwaysJump': classification_metrics(validation_labels, [1.0] * len(validation_labels), options.threshold),
        },
        'history': history, 'warnings': warnings,
        'limitations': [
            'Validation was used for checkpoint selection and is not an untouched final test set.',
            'Weighted-loss sigmoid output is a decision score, not a calibrated jump probability.',
            'Offline action classification does not measure survival or pipes passed by a closed-loop agent.',
        ],
    }
    return model, report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train the matching CPU model profile from normalized human gameplay.')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/training-data.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--difficulty', choices=['normal', 'hard'], help='Required for mixed recordings; otherwise inferred from the flights.')
    parser.add_argument('--initial-model', type=Path, help='Fine-tune compatible saved weights instead of starting randomly.')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--initialization-seed', type=int, help='Network initialization seed; defaults to --seed without changing the data split.')
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--split-by', choices=['seed', 'session'], default='seed')
    parser.add_argument('--min-samples', type=int, default=200, help='Minimum count; lowering it does not establish data quality.')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.data)
        _, difficulty = select_difficulty(dataset, args.difficulty)
        suffix = '616' if dataset.data['version'] == 4 else '516'
        prefix = f"human-{suffix}{'-hard' if difficulty == 'hard' else ''}-"
        args.output = args.output or ROOT / f'artifacts/{prefix}model.json'
        args.report = args.report or ROOT / f'artifacts/{prefix}training-report.json'
        require(len({args.data.resolve(), args.output.resolve(), args.report.resolve()}) == 3,
                'Dataset, model and report must use different paths.')
        if args.initial_model:
            require(args.initial_model.resolve() not in {args.output.resolve(), args.report.resolve()}, 'Fine-tuning cannot overwrite its starting model.')
        options = TrainingOptions(**{key: getattr(args, key) for key in TrainingOptions.__dataclass_fields__})
        model, report = train_dataset(dataset, options,
            progress=lambda epoch, train, validation: print(f'Epoch {epoch:4d} | train weighted BCE {train:.6f} | validation {validation:.6f}'),
            initial_model=args.initial_model)
        write_json(args.output, model)
        write_json(args.report, report)
        metrics = report['validation']
        print(f"Validation JUMP: precision={metrics['jumpPrecision']:.4f}, recall={metrics['jumpRecall']:.4f}, F1={metrics['jumpF1']:.4f}")
        print(f"Confusion matrix (rows actual WAIT/JUMP; columns predicted WAIT/JUMP): {metrics['confusionMatrix']}")
        print(f"Best epoch: {report['training']['bestEpoch']}; model: {args.output}; report: {args.report}")
        for warning in report['warnings']:
            print(f'WARNING: {warning}')
    except (DataError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
