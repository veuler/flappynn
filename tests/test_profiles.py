import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.create_lookahead_model import transfer
from scripts.model import load_model, validate_model, predict, write_json
from scripts.dataset import DataError, Dataset, load_dataset, validate_dataset
from scripts.train import train_dataset, TrainingOptions
from scripts.evaluate import evaluate_model
from scripts.self_train import pack, unpack, rollout, train
from scripts.export_css import export_bundle, parity_cases, main as export_main
from scripts.verify_parity import verify_report
from scripts.human_train import prepare

ROOT=Path(__file__).resolve().parents[1]
class ProfileTests(unittest.TestCase):
    def test_human_training_learns_decisions_from_the_following_gap(self):
        from tests.ml_fixture import make_fixture
        data=make_fixture()
        data.update(version=4,modelProfile='lookahead-616',inputNames=self.speed['inputNames'],
                    normalization=self.speed['normalization'])
        rng=np.random.default_rng(55)
        for row in data['samples']:
            following=float(rng.uniform(.15,.85))
            row.insert(5,following)
            row[-1]=int(following<.5)
        jumps=sum(row[-1] for row in data['samples'])
        data['counts']={'total':len(data['samples']),'jump':jumps,'wait':len(data['samples'])-jumps}
        dataset=Dataset(validate_dataset(data),'synthetic-following','synthetic-following.json')
        model,report=train_dataset(dataset,TrainingOptions(epochs=180,patience=40))
        self.assertGreater(report['validation']['jumpF1'],.9)
        low=predict(model,[.5,0,.5,.5,2/3,.2])['probability']
        high=predict(model,[.5,0,.5,.5,2/3,.8])['probability']
        self.assertGreater(low,.8);self.assertLess(high,.2)
    @classmethod
    def setUpClass(cls):
        cls.baseline=load_model(ROOT/'tests/fixtures/speed-model.json')
        cls.speed=transfer(cls.baseline)

    def test_transfer_preserves_baseline_predictions_for_every_following_gap_and_does_not_mutate_source(self):
        original=copy.deepcopy(self.baseline)
        for sample in parity_cases(self.baseline,random_count=20):
            for speed in (0,2/3,1):
                before=predict(self.baseline,sample['inputs'])
                after=predict(self.speed,sample['inputs']+[speed])
                self.assertEqual(before['probability'],after['probability'])
                self.assertEqual(before['hidden'],after['hidden'])
        self.assertEqual(self.baseline,original)
        self.assertEqual(len(pack(self.speed)),129)
        self.assertEqual(unpack(pack(self.speed),self.speed),self.speed)

    def test_sixth_input_affect_predictions_and_rollout(self):
        model=copy.deepcopy(self.speed)
        model['w1'][8][5]=2;model['b1'][8]=0;model['w2'][8]=3
        self.assertGreater(predict(model,[.5,0,.5,.5,2/3,1])['logit'],predict(model,[.5,0,.5,.5,2/3,0])['logit'])
        for difficulty in ('normal','hard'):
            before=rollout(pack(self.baseline),[2,5],self.baseline['gameConfig'],.5,seconds=40,difficulty=difficulty)
            after=rollout(pack(self.speed),[2,5],self.speed['gameConfig'],.5,seconds=40,difficulty=difficulty)
            np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
        bad=copy.deepcopy(model);bad['inputNames'].pop()
        with self.assertRaises(DataError):validate_model(bad)

    def test_actual_v4_recordings_train_129_parameters_and_reject_speed_evaluation(self):
        dataset=load_dataset(ROOT/'tests/fixtures/recording-flights-616.json')
        self.assertEqual(dataset.data['version'],4)
        model,report=train_dataset(dataset,TrainingOptions(epochs=2,patience=2,difficulty='hard'))
        self.assertEqual(model['architecture'],[6,16,1]);self.assertEqual(report['parameterCount'],129)
        self.assertEqual(evaluate_model(model,dataset,'all')['metrics']['sampleCount'],242)
        with self.assertRaises(DataError):evaluate_model(self.baseline,dataset,'all')
        bad=copy.deepcopy(dataset.data);bad['samples'][0][4]=1.1
        with self.assertRaises(DataError):validate_dataset(bad)
        bad=copy.deepcopy(dataset.data);bad['samples'][0][5]=1.1
        with self.assertRaises(DataError):validate_dataset(bad)

    def test_human_fine_tuning_preserves_source_and_learns_sixth_input(self):
        dataset=load_dataset(ROOT/'tests/fixtures/recording-flights-616.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            source=Path(directory)/'starting.json';write_json(source,self.speed);before=source.read_bytes()
            model,report=train_dataset(dataset,TrainingOptions(epochs=2,patience=2,difficulty='hard',learning_rate=.001),initial_model=source)
            self.assertEqual(model['training']['initialization'],'fine-tune')
            self.assertTrue(any(row[5] != 0 for row in model['w1']))
            self.assertTrue(any(v != 0 for v in model['w2'][8:]))
            self.assertEqual(source.read_bytes(),before)
            write_json(source,self.baseline)
            with self.assertRaisesRegex(DataError,'profiles or physics'):
                train_dataset(dataset,TrainingOptions(epochs=1,difficulty='hard'),initial_model=source)

    def test_one_click_worker_exports_human_candidate_with_comparable_hard_reports(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);source=root/'starting.json';write_json(source,self.speed);before=source.read_bytes()
            output=root/'job';output.mkdir();(output/'stop').touch()
            report=prepare(source,ROOT/'tests/fixtures/recording-flights-616.json',output,'hard')
            self.assertEqual(report['kind'],'human');self.assertEqual(report['difficulty'],'hard')
            self.assertEqual(len(report['baselineTest']['runs']),20)
            self.assertEqual(len(report['candidateTest']['runs']),20)
            self.assertEqual(load_model(output/'candidate.json')['training']['initialization'],'fine-tune')
            self.assertEqual(source.read_bytes(),before)

    def test_human_worker_random_start_does_not_use_or_overwrite_saved_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);source=root/'starting.json';write_json(source,self.speed);before=source.read_bytes()
            output=root/'job';output.mkdir();(output/'stop').touch()
            with patch('scripts.human_train.secrets.randbits',return_value=987654321) as fresh_seed:
                report=prepare(source,ROOT/'tests/fixtures/recording-flights-616.json',output,'hard',initialization='random')
                fresh_seed.assert_called_once_with(32)
            candidate=load_model(output/'candidate.json')
            self.assertEqual(candidate['training']['initialization'],'random')
            self.assertEqual(candidate['training']['initializationSeed'],987654321)
            self.assertEqual(candidate['training']['seed'],12345)
            self.assertNotIn('parentModelSha256',candidate['training'])
            self.assertNotEqual(candidate['w1'],self.speed['w1'])
            self.assertTrue(any(value != 0 for value in candidate['w2'][8:]))
            self.assertEqual(candidate['training']['learningRate'],.01)
            self.assertEqual(report['initialization'],'random')
            self.assertEqual(len(report['candidateTest']['runs']),20)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(__import__('json').loads((output/'status.json').read_text())['initialization'],'random')
            with self.assertRaisesRegex(ValueError,'initialization'):
                prepare(source,ROOT/'tests/fixtures/recording-flights-616.json',root/'bad','hard',initialization='zeros')

    def test_initialization_seed_changes_weights_preserves_split_and_replays_exactly(self):
        dataset=load_dataset(ROOT/'tests/fixtures/recording-flights-616.json')
        models=[];reports=[]
        for initialization_seed in [11,22,11]:
            model,report=train_dataset(dataset,TrainingOptions(epochs=2,patience=2,difficulty='hard',initialization_seed=initialization_seed))
            models.append(model);reports.append(report)
        self.assertNotEqual(models[0]['w1'],models[1]['w1'])
        for key in ['w1','b1','w2','b2']:
            self.assertEqual(models[0][key],models[2][key])
        self.assertEqual(reports[0]['history'],reports[2]['history'])
        self.assertEqual(reports[0]['training']['split'],reports[1]['training']['split'])
        for invalid in [-1,2**32,True]:
            with self.assertRaisesRegex(DataError,'Initialization seed'):
                train_dataset(dataset,TrainingOptions(difficulty='hard',initialization_seed=invalid))

    def test_new_architecture_self_training_exports_isolated_candidate(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);source=root/'starting.json';write_json(source,self.speed);before=source.read_bytes()
            report=train(source,root/'job',seconds=2,difficulty='hard')
            candidate=load_model(root/'job/candidate.json')
            self.assertEqual(candidate['architecture'],[6,16,1]);self.assertEqual(candidate['modelProfile'],'lookahead-616')
            self.assertEqual(report['difficulty'],'hard');self.assertEqual(source.read_bytes(),before)

    def test_css_export_and_parity_audit_require_all_sixteen_activations(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);source=root/'model.json';write_json(source,self.speed)
            metadata,fixture=export_bundle(source,root/'model.css',root/'inputs.json',root/'metadata.json',random_count=0)
            css=(root/'model.css').read_text()
            self.assertIn('--pipe-speed',css);self.assertIn('--h16',css)
            cases=[{'id':c['id'],'inputs':c['inputs'],'actualProbability':c['expected']['probability'],'actualLogit':c['expected']['logit'],'actualHidden':c['expected']['hidden']} for c in fixture['cases']]
            report={'version':1,'passed':True,'tolerance':1e-4,'probe':fixture['expectedProbe'],'caseCount':len(cases),'cases':cases,'cssSha256':metadata['cssSha256'],'modelSha256':metadata['modelSha256'],'browser':'unit reference','checkedAt':'test'}
            self.assertTrue(verify_report(report,fixture,metadata,(root/'model.css').read_bytes())['parityVerified'])
            cases[0]['actualHidden']=cases[0]['actualHidden'][:8]
            with self.assertRaises(DataError):verify_report(report,fixture,metadata,(root/'model.css').read_bytes())

    def test_cli_export_defaults_use_the_matching_profile_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);source=root/'model.json';write_json(source,self.speed)
            with patch('scripts.export_css.ROOT',root):
                self.assertEqual(export_main(['--model',str(source),'--random-count','0']),0)
                self.assertTrue((root/'src/models/lookahead-616/model.css').exists())
                self.assertFalse((root/'src/model.css').exists())
                self.assertEqual(export_main(['--model',str(source),'--output',str(root/'custom.css'),'--random-count','0']),0)
                self.assertTrue((root/'custom.css').exists())
