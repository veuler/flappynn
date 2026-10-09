"""Export explicit CSS arithmetic and matching offline reference states."""
import argparse
import hashlib
import itertools
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from scripts.dataset import DataError, read_json, require
from scripts.model import predict, validate_model, write_json, write_text

CSS_INPUTS = ['--bird-y', '--velocity-y', '--pipe-distance', '--gap-y', '--pipe-speed', '--following-gap-y']


def css_number(value):
    # Round-trip serialization preserves tiny coefficients; no decimal-place rounding.
    return repr(float(value))


def registered_number(name):
    return f'@property {name} {{\n  syntax: "<number>";\n  inherits: false;\n  initial-value: 0;\n}}\n'


def render_css(model, model_sha256):
    validate_model(model)
    css_inputs = CSS_INPUTS[:model['architecture'][0]]
    require(len(model_sha256) == 64 and all(c in '0123456789abcdef' for c in model_sha256), 'Invalid model digest.')
    lines = [f'/* Generated offline. Model SHA-256: {model_sha256} */',
             '/* All weighted sums, ReLU and sigmoid are evaluated by CSS. */']
    for name in ['--nn-probe', '--output-z', '--jump-probability'] + [f'--h{i}' for i in range(1, model['architecture'][1] + 1)]:
        lines.append(registered_number(name))
    lines.append('.css-neural-model {')
    for name in css_inputs:
        lines.append(f'  {name}: 0;')
    lines.append('  --nn-probe: calc(1 / (1 + exp(-1)));')
    for index, (weights, bias) in enumerate(zip(model['w1'], model['b1']), start=1):
        terms = [f'var({name}) * ({css_number(weight)})' for name, weight in zip(css_inputs, weights)]
        terms.append(f'({css_number(bias)})')
        lines.append(f'  --h{index}-z: calc(' + ' + '.join(terms) + ');')
        lines.append(f'  --h{index}: max(0, var(--h{index}-z));')
    # Registered numbers can serialize to six significant digits when substituted
    # into another property. Keep the forward pass as unregistered expressions;
    # registered hidden/logit properties remain precise diagnostic outputs only.
    terms = [f'max(0, var(--h{index}-z)) * ({css_number(weight)})' for index, weight in enumerate(model['w2'], start=1)]
    terms.append(f"({css_number(model['b2'])})")
    lines.append('  --output-z-raw: calc(' + ' + '.join(terms) + ');')
    lines.append('  --output-z: var(--output-z-raw);')
    lines.append('  --jump-probability: calc(1 / (1 + exp(-1 * var(--output-z-raw))));')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def parity_cases(model, seed=12345, random_count=500):
    require(type(random_count) is int and 0 <= random_count <= 10000, 'Random case count must be 0 to 10000.')
    states = [[0.5, 0.0, 0.5, 0.5], [0.62, -0.18, 0.31, 0.48]]
    states.extend([list(values) for values in itertools.product((0.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (0.0, 1.0))])
    generator = random.Random(seed)
    states.extend([[generator.random(), generator.uniform(-1, 1), generator.random(), generator.random()]
                   for _ in range(random_count)])
    # Repeat early inputs at the end to detect stale computed values after large state changes.
    states.extend([list(states[0]), list(states[1])])
    states = [values + [speed] for values in states for speed in (0.0, 2 / 3, 1.0)]
    if model['architecture'][0] == 6:
        states = [values + [gap] for values in states for gap in (0.0, 0.5, 1.0)]
    return [{'id': index, 'inputs': values, 'expected': predict(model, values)} for index, values in enumerate(states)]


def export_bundle(model_path, css_path, fixture_path, metadata_path, random_count=500, seed=12345):
    paths = [Path(p).resolve() for p in (model_path, css_path, fixture_path, metadata_path)]
    require(len(set(paths)) == 4, 'Model, CSS, fixture and metadata must use different paths.')
    raw_model, digest = read_json(model_path)
    model = validate_model(raw_model)
    css = render_css(model, digest)
    css_digest = hashlib.sha256(css.encode('utf-8')).hexdigest()
    metadata = {
        'version': model['version'], 'modelSha256': digest, 'cssSha256': css_digest,
        'architecture': model['architecture'], 'inputNames': model['inputNames'],
        'threshold': model['threshold'], 'normalization': model['normalization'], 'gameConfig': model['gameConfig'],
        'sampleIntervalMs': model['sampleIntervalMs'], 'minimumJumpIntervalMs': model['minimumJumpIntervalMs'],
        'dataKind': model.get('training', {}).get('dataKind', 'unverified'),
        'trainingDifficulty': model.get('training', {}).get('difficulty', 'normal'),
        'parityVerified': False,
    }
    metadata['modelProfile'] = model['modelProfile']
    if metadata['trainingDifficulty'] == 'hard':
        metadata['difficultyConfig'] = model['training']['difficultyConfig']
    fixture = {
        'version': 1, 'modelSha256': digest, 'cssSha256': css_digest,
        'dataKind': metadata['dataKind'], 'threshold': model['threshold'],
        'inputNames': model['inputNames'], 'tolerance': 1e-4, 'seed': seed,
        'architecture': model['architecture'],
        'expectedProbe': 0.7310585786300049,
        'cases': parity_cases(model, seed, random_count),
    }
    # This command cannot certify browser parity; the isolated browser page must pass.
    write_text(css_path, css)
    write_json(metadata_path, metadata)
    write_json(fixture_path, fixture)
    return metadata, fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description='Export CSS arithmetic and Python references for either supported model profile.')
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts/model-516.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--random-count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args(argv)
    try:
        model = validate_model(read_json(args.model)[0])
        base = ROOT / 'src/models' / model['modelProfile']
        if model.get('training', {}).get('difficulty') == 'hard':
            base /= 'hard'
        args.output = args.output or base / 'model.css'
        args.metadata = args.metadata or base / 'model.meta.json'
        args.fixture = args.fixture or base / 'parity-inputs.json'
        metadata, fixture = export_bundle(args.model, args.output, args.fixture, args.metadata, args.random_count, args.seed)
        print(f"CSS: {args.output}; reference cases: {len(fixture['cases'])}; model SHA-256: {metadata['modelSha256']}")
        print('Browser parity is NOT verified yet. Use the isolated test page before AI inference.')
    except (DataError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
