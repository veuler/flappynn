"""Bind a browser report to exact CSS/fixture bytes and audit every returned value."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from scripts.dataset import DataError, finite_number, read_json, require
from scripts.model import write_json


def verify_report(report, fixture, metadata, css_bytes):
    require(isinstance(report, dict) and report.get('version') == 1 and report.get('passed') is True,
            'Browser report did not pass.')
    css_digest = hashlib.sha256(css_bytes).hexdigest()
    for document in (report, fixture, metadata):
        require(document.get('cssSha256') == css_digest, 'CSS digest mismatch; rerun browser parity after export.')
        require(document.get('modelSha256') == metadata.get('modelSha256'), 'Model digest mismatch.')
    require(fixture.get('version') == 1 and fixture.get('tolerance') == 1e-4 and report.get('tolerance') == 1e-4,
            'V1 parity requires an absolute tolerance of 1e-4.')
    require(finite_number(report.get('probe')) and abs(report['probe'] - fixture['expectedProbe']) < 1e-4,
            'CSS numeric registration/exp probe failed.')
    references, results = fixture.get('cases'), report.get('cases')
    require(isinstance(references, list) and len(references) >= 20, 'Parity fixture must include at least 20 cases.')
    require(isinstance(results, list) and len(results) == len(references) == report.get('caseCount'),
            'Report does not cover all reference cases.')
    maxima = [0.0, 0.0, 0.0]
    mismatches = near_threshold = 0
    for reference, actual in zip(references, results):
        require(actual.get('id') == reference['id'] and actual.get('inputs') == reference['inputs'],
                'Report case order or inputs differ from the Python reference.')
        probability, logit, hidden = actual.get('actualProbability'), actual.get('actualLogit'), actual.get('actualHidden')
        require(finite_number(probability) and 0 <= probability <= 1 and finite_number(logit), 'Invalid browser output.')
        require(isinstance(hidden, list) and len(hidden) == metadata['architecture'][1] and all(finite_number(v) and v >= 0 for v in hidden),
                'Missing or invalid browser hidden activations.')
        expected = reference['expected']
        errors = [abs(probability - expected['probability']), abs(logit - expected['logit']),
                  max(abs(v - e) for v, e in zip(hidden, expected['hidden']))]
        require(all(error < 1e-4 for error in errors), f"Case {reference['id']} exceeds parity tolerance.")
        maxima = [max(before, current) for before, current in zip(maxima, errors)]
        close = abs(expected['probability'] - fixture['threshold']) < 1e-4
        near_threshold += int(close)
        mismatches += int(not close and (probability >= fixture['threshold']) != (expected['probability'] >= fixture['threshold']))
    require(mismatches == 0, 'CSS/Python decisions disagree outside the threshold tolerance.')
    require(isinstance(report.get('browser'), str) and report['browser'] and isinstance(report.get('checkedAt'), str),
            'Missing browser audit information.')
    return {**metadata, 'parityVerified': True, 'parity': {
        'browser': report['browser'], 'checkedAt': report['checkedAt'], 'caseCount': len(results),
        'tolerance': 1e-4, 'maximumProbabilityError': maxima[0], 'maximumLogitError': maxima[1],
        'maximumHiddenError': maxima[2], 'decisionMismatches': mismatches, 'nearThresholdCases': near_threshold,
    }}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Audit a real browser parity report and mark its matching metadata verified.')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/browser/generated/parity-inputs.json')
    parser.add_argument('--css', type=Path, default=ROOT / 'tests/browser/generated/model.css')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'tests/browser/generated/model.meta.json')
    args = parser.parse_args(argv)
    try:
        require(len({args.report.resolve(), args.fixture.resolve(), args.css.resolve(), args.metadata.resolve()}) == 4,
                'Report, fixture, CSS and metadata must use different paths.')
        result = verify_report(read_json(args.report)[0], read_json(args.fixture)[0], read_json(args.metadata)[0], args.css.read_bytes())
        write_json(args.metadata, result)
        print(f"Verified {result['parity']['caseCount']} browser cases; max probability error {result['parity']['maximumProbabilityError']:.8g}")
        print(f'Metadata marked verified: {args.metadata}')
    except (DataError, OSError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
