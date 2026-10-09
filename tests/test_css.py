import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.dataset import DataError, read_json
from scripts.export_css import export_bundle, parity_cases, render_css
from scripts.model import predict, write_json
from scripts.verify_parity import verify_report
from tests.ml_fixture import make_fixture


class CssExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contract = make_fixture()
        cls.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / 'artifacts')
        cls.model_path = Path(cls.directory.name) / 'unit-model.json'
        cls.model = {
            'version': 2, 'modelProfile': 'speed-516', 'architecture': [5, 16, 1], 'activation': 'relu', 'outputActivation': 'sigmoid',
            'inputNames': contract['inputNames'], 'threshold': 0.5,
            'w1': [[0.72 + i * 0.1, -0.31, 0.18, -0.44, 0.2] for i in range(16)],
            'b1': [0.1] * 16, 'w2': [0.8 if i % 2 == 0 else -0.6 for i in range(16)], 'b2': -0.19,
            'normalization': contract['normalization'], 'gameConfig': contract['gameConfig'],
            'sampleIntervalMs': 50, 'minimumJumpIntervalMs': 100,
        }
        cls.model['w1'][2][0] = 1e-20
        write_json(cls.model_path, cls.model)
        cls.digest = read_json(cls.model_path)[1]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_tiny_coefficients_survive_export_and_model_errors_are_rejected(self):
        css = render_css(self.model, self.digest)
        self.assertIn(repr(self.model['w1'][2][0]), css)
        self.assertIn('exp(-1 * var(--output-z-raw))', css)
        model = copy.deepcopy(self.model)
        model['w1'][0].pop()
        with self.assertRaises(DataError):
            render_css(model, self.digest)

    def test_parity_states_include_boundaries_repeats_and_python_reference(self):
        cases = parity_cases(self.model)
        self.assertEqual(len(cases), 1560)
        self.assertEqual(cases[0]['inputs'], cases[-6]['inputs'])
        self.assertEqual(cases[3]['inputs'], cases[-3]['inputs'])
        self.assertEqual(cases[0]['expected'], predict(self.model, cases[0]['inputs']))
        self.assertEqual(cases, parity_cases(self.model))

    def test_export_is_unverified_until_report_covers_the_matching_model_and_css(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / 'artifacts') as directory:
            root = Path(directory)
            metadata, fixture = export_bundle(self.model_path, root / 'model.css', root / 'fixture.json', root / 'meta.json', 4)
            css = (root / 'model.css').read_bytes()
            self.assertEqual(metadata['cssSha256'], hashlib.sha256(css).hexdigest())
            self.assertFalse(metadata['parityVerified'])
            # A constructed report exercises the verifier; this is not evidence of browser support.
            report = {'version': 1, 'passed': True, 'cssSha256': metadata['cssSha256'],
                      'modelSha256': metadata['modelSha256'], 'tolerance': 1e-4, 'probe': fixture['expectedProbe'],
                      'caseCount': len(fixture['cases']), 'browser': 'unit-test-only', 'checkedAt': 'unit-test-only',
                      'cases': [{'id': c['id'], 'inputs': c['inputs'], 'actualProbability': c['expected']['probability'],
                                 'actualLogit': c['expected']['logit'], 'actualHidden': c['expected']['hidden']}
                                for c in fixture['cases']]}
            verified = verify_report(report, fixture, metadata, css)
            self.assertTrue(verified['parityVerified'])
            self.assertFalse(metadata['parityVerified'])
            for mutate in [lambda r: r['cases'].pop(),
                           lambda r: r.__setitem__('modelSha256', 'stale'),
                           lambda r: r['cases'][0].__setitem__('actualProbability', -1),
                           lambda r: r['cases'][0].__setitem__('actualHidden', [0] * 8),
                           lambda r: r['cases'][0].__setitem__('inputs', [0] * 4)]:
                bad = copy.deepcopy(report)
                mutate(bad)
                with self.subTest(mutation=mutate), self.assertRaises(DataError):
                    verify_report(bad, fixture, metadata, css)
            with self.assertRaisesRegex(DataError, 'CSS digest'):
                verify_report(report, fixture, metadata, css + b'\n')


if __name__ == '__main__':
    unittest.main()
