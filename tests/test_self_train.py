import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from scripts.model import load_model
from scripts.self_train import FlightBatch, pack, random_sequence, rollout, train, unpack

ROOT = Path(__file__).resolve().parents[1]


class SelfTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(ROOT / 'tests/fixtures/speed-model.json')
        script = """
        import { Game } from './src/js/game.js';
        import { createRandom } from './src/js/pipes.js';
        const results = ['normal', 'hard'].flatMap(difficulty => [0, 1, 12345, 9001, 4294967295].map(seed => {
          const random = createRandom(seed);
          const randoms = Array.from({length:50},()=>random());
          const game = new Game({seed, difficulty}); game.start();
          const trace = [];
          for(let tick=1; tick<=2400 && game.status==='running'; tick++) {
            game.step();
            const state = game.getState();
            const action = game.status==='running' && tick%6===0 && state.birdY>state.nextPipeGapCenterY+12 && state.birdVelocityY>0;
            trace.push({tick, state, action});
            if(action) game.jump();
          }
          return {seed, difficulty, randoms, trace};
        }));
        console.log(JSON.stringify(results));
        """
        cls.reference = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', script], cwd=ROOT, text=True))

    def test_pipe_randomness_matches_javascript_unsigned_overflow(self):
        for run in self.reference:
            self.assertEqual(random_sequence(run['seed'], 50), run['randoms'])

    def test_replayed_flight_matches_real_game_at_every_physics_tick(self):
        for run in self.reference:
            flight = FlightBatch([run['seed']], self.model['gameConfig'], 20, run['difficulty'])
            for point in run['trace']:
                inputs = flight.step()
                state = point['state']
                self.assertAlmostEqual(flight.y[0], state['birdY'], places=10)
                self.assertAlmostEqual(flight.velocity[0], state['birdVelocityY'], places=10)
                self.assertAlmostEqual(flight.elapsed[0], state['elapsedSeconds'], places=10)
                self.assertEqual(bool(flight.alive[0]), state['status'] == 'running')
                self.assertEqual(flight.score[0], state['score'])
                self.assertAlmostEqual(flight.speed[0], state['pipeSpeed'], places=8)
                if flight.alive[0]:
                    self.assertAlmostEqual(inputs[0, 2], min(1, state['nextPipeDistanceX'] / 600), places=10)
                    self.assertAlmostEqual(inputs[0, 3], state['nextPipeGapCenterY'] / 640, places=10)
                flight.jump(np.array([point['action']]))

    def test_batch_episode_deaths_do_not_change_other_models(self):
        parent = pack(self.model)
        waiting = np.zeros(113)
        waiting[-1] = -10
        for difficulty in ('normal', 'hard'):
            combined = rollout(np.vstack((parent, waiting)), [12345, 901], self.model['gameConfig'], .62, seconds=30, difficulty=difficulty)
            individual = rollout(parent, [12345, 901], self.model['gameConfig'], .62, seconds=30, difficulty=difficulty)
            np.testing.assert_array_equal(combined[0][0], individual[0][0])
            np.testing.assert_array_equal(combined[1][0], individual[1][0])
            self.assertTrue((combined[0][1] == 0).all())

    def test_long_hard_flight_matches_browser_speed_cap_pipe_spawning_and_scoring(self):
        script = """
        import { Game } from './src/js/game.js';
        const game=new Game({seed:42, difficulty:'hard'}); game.start();
        const trace=[];
        for(let tick=1;tick<=24000;tick++) {
          for(const pipe of game.pipes) pipe.gapCenterY=320;
          game.bird.y=320; game.bird.velocityY=0; game.step();
          if(tick%120===0) trace.push({tick, state:game.getState()});
        }
        console.log(JSON.stringify(trace));
        """
        reference = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', script], cwd=ROOT, text=True))
        flight = FlightBatch([42], self.model['gameConfig'], 200, 'hard')
        flight.gaps[:] = 320
        for point in reference:
            while flight.tick < point['tick']:
                flight.y[:] = 320
                flight.velocity[:] = 0
                inputs = flight.step()
            state = point['state']
            self.assertEqual(flight.score[0], state['score'])
            self.assertAlmostEqual(flight.speed[0], state['pipeSpeed'], places=8)
            self.assertAlmostEqual(inputs[0, 2], min(1, state['nextPipeDistanceX'] / 600), places=8)
        self.assertTrue(flight.alive[0])
        self.assertGreater(flight.score[0], 50)
        self.assertGreater(flight.speed[0], 269.99)

    def test_hard_training_exports_its_difficulty_without_changing_normal_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as directory:
            target = Path(directory)
            (target / 'stop').write_text('stop')
            source = ROOT / 'tests/fixtures/speed-model.json'
            original = source.read_bytes()
            report = train(source, target, seconds=1, difficulty='hard')
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(report['difficulty'], 'hard')
            self.assertEqual(load_model(target / 'candidate.json')['training']['difficulty'], 'hard')
            self.assertEqual(json.loads((target / 'model.meta.json').read_text())['trainingDifficulty'], 'hard')
            self.assertFalse(json.loads((target / 'model.meta.json').read_text())['parityVerified'])
        with self.assertRaises(ValueError):
            FlightBatch([1], self.model['gameConfig'], 1, 'unknown')

    def test_stop_saves_candidate_and_leaves_source_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as directory:
            target = Path(directory)
            (target / 'stop').write_text('stop')
            source = ROOT / 'tests/fixtures/speed-model.json'
            original = source.read_bytes()
            report = train(source, target, seconds=1)
            self.assertTrue(report['stoppedEarly'])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(pack(load_model(target / 'candidate.json')).tolist(), pack(self.model).tolist())
            self.assertFalse(json.loads((target / 'model.meta.json').read_text())['parityVerified'])

    def test_parameter_roundtrip_and_deadline_discard_incomplete_batch(self):
        params = pack(self.model)
        np.testing.assert_array_equal(pack(unpack(params, self.model)), params)
        self.assertIsNone(rollout(params, [1], self.model['gameConfig'], .62, seconds=30, interrupted=lambda: True))


if __name__ == '__main__':
    unittest.main()
