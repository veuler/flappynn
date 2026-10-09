"""Create a separate six-input starting model, preserving the five-input behavior."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
from scripts.model import load_model, validate_model, write_json
from scripts.export_css import export_bundle


def transfer(source):
    validate_model(source)
    if source['architecture'] != [5, 16, 1]:
        raise ValueError('Transfer requires a five-input source.')
    model = copy.deepcopy(source)
    difficulty = source.get('training', {}).get('difficulty', 'normal')
    model.update(version=3, modelProfile='lookahead-616', architecture=[6, 16, 1],
        inputNames=source['inputNames'] + ['followingGapY'],
        normalization={**source['normalization'], 'version': 3},
        w1=[row + [0.0] for row in source['w1']],
        createdAt=datetime.now(timezone.utc).isoformat(),
        training={'dataKind': 'warm-start-lookahead', 'difficulty': difficulty,
            **({'difficultyConfig': source['training']['difficultyConfig']} if difficulty == 'hard' else {}),
            'method': 'Function-preserving transfer; the sixth input starts with zero weights.',
            'note': 'Train with new six-input observations to learn lookahead behavior.'})
    return validate_model(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or any((args.bundle / name).exists() for name in ('model.css', 'model.meta.json', 'parity-inputs.json')):
        raise SystemExit('The six-input slot already exists; its training will not be overwritten.')
    model = transfer(load_model(args.source))
    model['training']['parentModelSha256'] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    write_json(args.output, model)
    export_bundle(args.output, args.bundle/'model.css', args.bundle/'parity-inputs.json', args.bundle/'model.meta.json')
    print('Created six-input starting model. Browser parity is required before playing.')


if __name__ == '__main__':
    main()
