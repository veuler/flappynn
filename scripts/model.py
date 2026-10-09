"""Framework-independent JSON contract and double-precision offline reference."""
import json
import math
import os
from pathlib import Path
import tempfile
import time

from .dataset import finite_number, read_json, require
from .difficulties import HARD_RULES
from .profiles import profile_for_architecture


def validate_model(model):
    require(isinstance(model, dict), 'Model must be an object.')
    require(model.get('architecture') in ([5, 16, 1], [6, 16, 1]) and all(type(v) is int for v in model['architecture']), 'Unsupported model architecture.')
    profile = profile_for_architecture(model['architecture'])
    inputs, hidden, _ = profile['architecture']
    require(type(model.get('version')) is int and model['version'] == profile['version'], 'Unsupported model version.')
    require(model.get('modelProfile') == profile['id'], 'Model profile disagrees with architecture.')
    require(model.get('activation') == 'relu' and model.get('outputActivation') == 'sigmoid', 'Unsupported activations.')
    require(model.get('inputNames') == profile['inputNames'], 'Unexpected model input order.')
    for key, length in (('w1', hidden), ('b1', hidden), ('w2', hidden)):
        require(isinstance(model.get(key), list) and len(model[key]) == length, f'Invalid {key} shape.')
    require(all(isinstance(row, list) and len(row) == inputs and all(finite_number(v) for v in row)
                for row in model['w1']), 'Invalid finite w1 matrix.')
    require(all(finite_number(v) for v in model['b1'] + model['w2']) and finite_number(model.get('b2')),
            'Non-finite weights or biases.')
    require(finite_number(model.get('threshold')) and 0 < model['threshold'] < 1, 'Threshold must be between 0 and 1.')
    normalization = model.get('normalization')
    require(isinstance(normalization, dict) and normalization.get('version') == profile['version'], 'Missing model normalization.')
    if inputs >= 5:
        require(normalization.get('maxPipeSpeed') == 270, 'Invalid pipe speed normalization.')
    require(all(finite_number(normalization.get(k)) and normalization[k] > 0
                for k in ('height', 'velocityScale', 'maxPipeDistance')), 'Invalid model normalization constants.')
    require(isinstance(model.get('gameConfig'), dict), 'Missing model game configuration.')
    require(model['gameConfig'].get('height') == normalization['height'], 'Model game height and normalization disagree.')
    require(model.get('sampleIntervalMs') == 50 and model.get('minimumJumpIntervalMs') == 100, 'Invalid model timing.')
    if model.get('training', {}).get('difficulty') == 'hard':
        require(model['training'].get('difficultyConfig') == HARD_RULES, 'Hard model difficulty rules do not match this simulator.')
    return model


def load_model(path):
    return validate_model(read_json(path)[0])


def predict(model, inputs):
    require(len(inputs) == model['architecture'][0] and all(finite_number(v) for v in inputs), 'Unexpected input count or nonfinite input.')
    hidden = [max(0.0, sum(weight * value for weight, value in zip(row, inputs)) + bias)
              for row, bias in zip(model['w1'], model['b1'])]
    logit = sum(weight * value for weight, value in zip(model['w2'], hidden)) + model['b2']
    require(all(finite_number(value) for value in hidden) and finite_number(logit),
            'Model arithmetic overflowed on these inputs.')
    if logit >= 0:
        probability = 1 / (1 + math.exp(-logit))
    else:
        exponential = math.exp(logit)
        probability = exponential / (1 + exponential)
    return {'hidden': hidden, 'logit': logit, 'probability': probability}


def write_json(path, data):
    """Atomic replacement prevents leaving a partial model after interruption."""
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    write_text(path, text)


def replace_file_with_retry(source, destination):
    """Keep the complete destination while Windows briefly denies replacement."""
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32) or attempt == 7:
                raise
            time.sleep(min(0.025 * 2 ** attempt, 0.5))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        replace_file_with_retry(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
