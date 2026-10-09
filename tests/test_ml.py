import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import DEFAULT, patch

from scripts.dataset import Dataset, DataError, load_dataset, select_difficulty, session_indices, split_dataset, validate_dataset
from scripts.difficulties import HARD_RULES
from scripts.evaluate import evaluate_model
from scripts.metrics import binary_losses, classification_metrics
from scripts.model import load_model, predict, validate_model, write_json
from scripts.train import TrainingOptions, train_dataset
from tests.ml_fixture import make_fixture


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts')
        cls.root = Path(cls.directory.name)
        cls.path = cls.root / 'synthetic.json'
        write_json(cls.path, make_fixture())
        cls.dataset = load_dataset(cls.path)
        cls.options = TrainingOptions(epochs=250, patience=40)
        cls.model, cls.report = train_dataset(cls.dataset, cls.options)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_loader_accepts_real_js_contract_without_renormalizing(self):
        fixture = make_fixture()
        self.assertEqual(validate_dataset(fixture)['samples'], fixture['samples'])
        self.assertEqual(self.dataset.data['normalization']['height'], 640)

    def test_actual_js_mixed_recordings_train_and_evaluate_a_hard_model(self):
        dataset = load_dataset(Path(__file__).parent / 'fixtures/recording-flights.json')
        self.assertEqual(dataset.data['version'], 3)
        self.assertEqual([flight['difficulty'] for flight in dataset.sessions], ['normal', 'hard', 'normal', 'hard'])
        self.assertTrue(all(flight['score'] > 0 for flight in dataset.sessions))
        model, report = train_dataset(dataset, TrainingOptions(epochs=2, patience=2, difficulty='hard'))
        self.assertEqual(model['training']['difficultyConfig'], HARD_RULES)
        self.assertEqual(report['train']['sampleCount'] + report['validation']['sampleCount'], 242)
        self.assertEqual(evaluate_model(model, dataset, partition='all')['metrics']['sampleCount'], 242)

    def test_legacy_recordings_and_mixed_difficulty_filter_preserve_source_and_ranges(self):
        legacy = make_fixture()
        legacy['version'] = 1
        legacy.pop('difficultyRules')
        for flight in legacy['sessions']:
            flight.pop('difficulty')
        with self.assertRaises(DataError): validate_dataset(legacy)
        mixed = copy.deepcopy(self.dataset)
        for flight in mixed.sessions[1::2]:
            flight['difficulty'] = 'hard'
        validate_dataset(mixed.data)
        with self.assertRaisesRegex(DataError, 'Mixed recordings'):
            select_difficulty(mixed)
        selected, difficulty = select_difficulty(mixed, 'hard')
        self.assertEqual(difficulty, 'hard')
        self.assertEqual(selected.sha256, mixed.sha256)
        self.assertEqual([flight['id'] for flight in selected.sessions], [2, 4, 6, 8])
        self.assertEqual([flight['startIndex'] for flight in selected.sessions], [0, 80, 160, 240])
        self.assertEqual(selected.samples[:80], mixed.samples[80:160])
        self.assertEqual(selected.data['counts']['total'], 320)
        self.assertEqual(len(mixed.samples), 640)
        validate_dataset(selected.data)
        with self.assertRaisesRegex(DataError, 'No hard recordings'):
            select_difficulty(self.dataset, 'hard')
        mixed.data['difficultyRules']['hard']['maxSpeedMultiplier'] = 2
        with self.assertRaisesRegex(DataError, 'difficulty rules'):
            validate_dataset(mixed.data)

    def test_hard_human_training_exports_compatible_model_and_evaluates_only_hard_flights(self):
        mixed = copy.deepcopy(self.dataset)
        for flight in mixed.sessions[1::2]:
            flight['difficulty'] = 'hard'
        options = TrainingOptions(epochs=5, patience=5, difficulty='hard')
        model, report = train_dataset(mixed, options)
        self.assertEqual(model['training']['difficulty'], 'hard')
        self.assertEqual(model['training']['difficultyConfig'], HARD_RULES)
        self.assertTrue(set(model['training']['split']['trainSessionIds']) <= {2, 4, 6, 8})
        result = evaluate_model(model, mixed, partition='all')
        self.assertEqual(result['difficulty'], 'hard')
        self.assertEqual(result['metrics']['sampleCount'], 320)
        validation = evaluate_model(model, mixed)
        self.assertEqual(validation['metrics']['sampleCount'], report['validation']['sampleCount'])

    def test_atomic_json_write_retries_transient_windows_locks(self):
        destination = self.root / 'locked-status.json'
        write_json(destination, {'generation': 1})
        blocked = PermissionError('temporary file lock')
        blocked.winerror = 5
        with patch('scripts.model.os.replace', wraps=os.replace, side_effect=[blocked, blocked, DEFAULT]) as replacement, patch('scripts.model.time.sleep') as sleep:
            write_json(destination, {'generation': 2})
        self.assertEqual(replacement.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(destination.read_text().strip(), '{\n  "generation": 2\n}')
        self.assertEqual(list(self.root.glob('.locked-status.json.*.tmp')), [])

    def test_atomic_json_write_keeps_previous_data_on_persistent_or_non_windows_denial(self):
        destination = self.root / 'denied-status.json'
        write_json(destination, {'generation': 1})
        original = destination.read_bytes()
        for windows_code, attempts in [(32, 8), (None, 1)]:
            blocked = PermissionError('file is not writable')
            if windows_code is not None:
                blocked.winerror = windows_code
            with patch('scripts.model.os.replace', side_effect=blocked) as replacement, patch('scripts.model.time.sleep'):
                with self.assertRaises(PermissionError):
                    write_json(destination, {'generation': 2})
            self.assertEqual(replacement.call_count, attempts)
            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(list(self.root.glob('.denied-status.json.*.tmp')), [])

    def test_loader_rejects_bad_rows_metadata_counts_and_session_ranges(self):
        mutations = [
            lambda d: d['samples'][0].__setitem__(0, True),
            lambda d: d['samples'][0].__setitem__(1, float('nan')),
            lambda d: d['samples'][0].__setitem__(1, 10 ** 400),
            lambda d: d['samples'][0].__setitem__(3, 1.1),
            lambda d: d['samples'][0].__setitem__(-1, 1.0),
            lambda d: d['samples'][0].pop(),
            lambda d: d['counts'].__setitem__('total', 1),
            lambda d: d['sessions'][1].__setitem__('startIndex', 0),
            lambda d: d['sessions'][0].__setitem__('elapsedSeconds', 0),
            lambda d: d['sessions'][1].__setitem__('id', 1),
            lambda d: d['normalization'].__setitem__('height', 999),
            lambda d: d.__setitem__('sampleTiming', 'after-action'),
            lambda d: d.__setitem__('inputNames', list(reversed(d['inputNames']))),
        ]
        for mutate in mutations:
            data = make_fixture()
            mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises(DataError):
                validate_dataset(data)

    def test_json_reader_rejects_duplicates_constants_and_bad_syntax(self):
        path = self.root / 'bad.json'
        for content in ('{"version":1,"version":1}', '{"value":NaN}', '{"value":Infinity}', '{invalid}'):
            path.write_text(content, encoding='utf-8')
            with self.subTest(content=content), self.assertRaises(DataError):
                load_dataset(path)

    def test_split_is_deterministic_disjoint_and_keeps_repeated_seeds_together(self):
        data = copy.deepcopy(self.dataset)
        data.sessions[1]['seed'] = data.sessions[0]['seed']
        split = split_dataset(data)
        self.assertEqual(split, split_dataset(data))
        train = set(session_indices(data, split['trainSessionIds']))
        validation = set(session_indices(data, split['validationSessionIds']))
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(range(len(data.samples))))
        self.assertEqual(1 in split['trainSessionIds'], 2 in split['trainSessionIds'])
        train_seeds = {s['seed'] for s in data.sessions if s['id'] in split['trainSessionIds']}
        validation_seeds = {s['seed'] for s in data.sessions if s['id'] in split['validationSessionIds']}
        self.assertFalse(train_seeds & validation_seeds)

    def test_split_rejects_one_group_small_data_and_single_class(self):
        data = copy.deepcopy(self.dataset)
        for s in data.sessions:
            s['seed'] = 1
        with self.assertRaisesRegex(DataError, 'two distinct seed'):
            split_dataset(data)
        with self.assertRaisesRegex(DataError, 'At least'):
            split_dataset(self.dataset, min_samples=1000)
        for row in data.samples:
            row[-1] = 0
        data.data['counts']['jump'] = 0
        with self.assertRaises(DataError):
            split_dataset(data, split_by='session')

    def test_metrics_match_hand_checked_confusion_matrix_and_stable_extreme_loss(self):
        metrics = classification_metrics([0, 0, 1, 1, 1], [0.1, 0.9, 0.4, 0.8, 0.7])
        self.assertEqual(metrics['confusionMatrix'], [[1, 1], [1, 2]])
        self.assertAlmostEqual(metrics['jumpPrecision'], 2 / 3)
        self.assertAlmostEqual(metrics['jumpRecall'], 2 / 3)
        self.assertAlmostEqual(metrics['jumpF1'], 2 / 3)
        self.assertEqual(classification_metrics([0, 1], [0, 0])['jumpF1'], 0)
        self.assertEqual(binary_losses([0, 1], [1000, -1000], 3), {'bce': 1000, 'weightedBce': 2000})

    def test_model_learns_nonconstant_behavior_and_uses_training_only_class_weight(self):
        validation = self.report['validation']
        self.assertGreater(validation['jumpF1'], 0.8)
        self.assertGreater(validation['predictionMax'] - validation['predictionMin'], 0.5)
        self.assertGreater(validation['jumpF1'], self.report['baselines']['alwaysWait']['jumpF1'])
        train = self.report['train']
        self.assertEqual(self.model['training']['positiveWeight'], train['waitCount'] / train['jumpCount'])
        self.assertEqual(self.model['training']['dataKind'], 'synthetic-test')

    def test_identical_seed_repeats_weights_and_history(self):
        second_model, second_report = train_dataset(self.dataset, self.options)
        for key in ('w1', 'b1', 'w2', 'b2'):
            self.assertEqual(self.model[key], second_model[key])
        self.assertEqual(self.report['history'], second_report['history'])

    def test_json_roundtrip_contains_all_113_parameters_and_reproduces_metrics(self):
        path = self.root / 'model.json'
        write_json(path, self.model)
        model = load_model(path)
        self.assertEqual(sum(len(row) for row in model['w1']) + len(model['b1']) + len(model['w2']) + 1, 113)
        result = evaluate_model(model, self.dataset)
        self.assertEqual(result['metrics'], self.report['validation'])

    def test_python_reference_matches_double_precision_torch_and_handles_extreme_logits(self):
        import torch
        row = self.dataset.samples[10][:-1]
        x = torch.tensor(row, dtype=torch.float64)
        hidden = torch.relu(torch.tensor(self.model['w1'], dtype=torch.float64) @ x +
                            torch.tensor(self.model['b1'], dtype=torch.float64))
        logit = torch.tensor(self.model['w2'], dtype=torch.float64) @ hidden + self.model['b2']
        expected = torch.sigmoid(logit).item()
        self.assertAlmostEqual(predict(self.model, row)['probability'], expected, places=12)
        for bias, expected in [(1000, 1), (-1000, 0)]:
            model = copy.deepcopy(self.model)
            model['w2'] = [0] * 16
            model['b2'] = bias
            self.assertEqual(predict(model, row)['probability'], expected)

    def test_invalid_model_or_incompatible_dataset_is_rejected(self):
        model = copy.deepcopy(self.model)
        model['w1'][0].pop()
        with self.assertRaises(DataError):
            validate_model(model)
        data = copy.deepcopy(self.dataset)
        data.sha256 = 'wrong'
        with self.assertRaisesRegex(DataError, 'fingerprint'):
            evaluate_model(self.model, data)
        self.assertEqual(evaluate_model(self.model, data, 'all')['metrics']['sampleCount'], 640)
        data.data['normalization']['velocityScale'] = 999
        with self.assertRaisesRegex(DataError, 'normalization'):
            evaluate_model(self.model, data, 'all')

    def test_overflow_and_failed_serialization_do_not_produce_partial_model_files(self):
        model = copy.deepcopy(self.model)
        model['w1'][0] = [1e308] * 5
        with self.assertRaisesRegex(DataError, 'overflowed'):
            predict(model, [1, 1, 1, 1, 1])
        path = self.root / 'preserved.json'
        write_json(path, {'value': 'original'})
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            write_json(path, {'value': float('nan')})
        self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
