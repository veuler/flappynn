import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from scripts.rl_env import CONFIG, VectorFlightEnv, _random
from scripts.self_train import FlightBatch
from scripts.rl_train import Policy, _atomic_torch_save, generalized_advantage, train


class CheckpointWriteTests(unittest.TestCase):
    def test_temporary_windows_lock_is_retried_without_removing_previous_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as temp:
            target = Path(temp) / 'checkpoint.pt'
            torch.save({'updates': 10}, target)
            original_replace = os.replace
            calls = []
            def briefly_locked(source, destination):
                calls.append(source)
                self.assertEqual(torch.load(target, weights_only=False)['updates'], 10)
                if len(calls) < 3:
                    error = PermissionError('File temporarily locked')
                    error.winerror = 5 if len(calls) == 1 else 32
                    raise error
                return original_replace(source, destination)
            with patch('scripts.model.os.replace', side_effect=briefly_locked), patch('scripts.model.time.sleep'):
                _atomic_torch_save({'updates': 11}, target)
            self.assertEqual(len(calls), 3)
            self.assertEqual(torch.load(target, weights_only=False)['updates'], 11)
            self.assertEqual(list(Path(temp).iterdir()), [target])

    def test_permanent_lock_preserves_the_previous_checkpoint_and_raises(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as temp:
            target = Path(temp) / 'checkpoint.pt'
            torch.save({'updates': 10}, target)
            error = PermissionError('Persistent file lock'); error.winerror = 5
            with patch('scripts.model.os.replace', side_effect=error) as replace, patch('scripts.model.time.sleep'):
                with self.assertRaises(PermissionError):
                    _atomic_torch_save({'updates': 11}, target)
            self.assertEqual(replace.call_count, 8)
            self.assertEqual(torch.load(target, weights_only=False)['updates'], 10)


class VectorEnvironmentTests(unittest.TestCase):
    def test_episode_survives_old_limits_and_truncates_at_600_seconds(self):
        for difficulty in ('normal', 'hard'):
            with self.subTest(difficulty=difficulty):
                env = VectorFlightEnv(1, difficulty=difficulty, seed=11)
                for elapsed in (179.95, 299.95):
                    env.elapsed[0] = elapsed
                    _, _, done, info = env.step([False])
                    self.assertFalse(done[0])
                    self.assertFalse(info['truncated'][0])
                env.elapsed[0] = 599.95
                _, reward, done, info = env.step([False])
                self.assertTrue(done[0])
                self.assertTrue(info['truncated'][0])
                self.assertFalse(info['terminated'][0])
                self.assertGreater(reward[0], 0)  # Time limit is not a collision penalty.

    def test_physics_matches_flight_batch_normal_and_hard(self):
        for difficulty in ('normal', 'hard'):
            env = VectorFlightEnv(1, difficulty, seed=11)
            seed = 12345
            env.seed[0] = seed
            env.random_state[0] = seed
            for j in range(3): env.gaps[0, j] = env._new_gap(0)
            flight = FlightBatch([seed], CONFIG, 30, difficulty)
            for _ in range(6): out = flight.step()
            for _ in range(24):
                env.step([False])
                out = None
                for _ in range(6): out = flight.step()
                self.assertAlmostEqual(env.y[0], flight.y[0], places=8)
                self.assertAlmostEqual(env.vy[0], flight.velocity[0], places=8)
                self.assertAlmostEqual(env.speed[0], flight.speed[0], places=8)
                if not env.alive[0]: break
                state = env.observe()[0]
                np.testing.assert_allclose(state[:4], out[0], atol=1e-6)
                self.assertAlmostEqual(float(state[4]), float(np.clip(flight.speed[0] / 270, 0, 1)), places=6)

    def test_cooldown_and_initial_impulse_match_session_schedule(self):
        env = VectorFlightEnv(1, seed=3)
        self.assertEqual(env.vy[0], -365)
        self.assertFalse(env.can_jump()[0])
        env.step([True])
        self.assertTrue(env.can_jump()[0])
        before = env.vy[0]
        env.step([True])
        self.assertLess(env.vy[0], before)
        last = int(env.last_jump_tick[0])
        env.step([True])
        self.assertEqual(int(env.last_jump_tick[0]), last)

    def test_selective_auto_reset_does_not_advance_other_actors(self):
        env = VectorFlightEnv(2, seed=9)
        for _ in range(5): env.step([False, False])
        keys = ('seed','random_state','y','vy','score','elapsed','tick','last_jump_tick',
                'episode_reward','speed','alive','px','gaps','passed','pipe_count')
        before = {key: getattr(env, key)[1].copy() for key in keys}
        env.reset([0], seeds=[123])
        for key in keys:
            np.testing.assert_array_equal(getattr(env, key)[1], before[key], err_msg=key)

    def test_timeout_bootstraps_but_collision_does_not(self):
        reward = np.array([[1.0], [0.0]], np.float32)
        values = np.array([[2.0], [3.0]], np.float32)
        next_values = np.array([[4.0], [5.0]], np.float32)
        terminated = np.array([[False], [True]])
        truncated = np.array([[True], [False]])
        advantage, returns = generalized_advantage(reward, values, next_values, terminated, truncated, gamma=.9, gae_lambda=.8)
        self.assertAlmostEqual(float(advantage[1, 0]), -3.0)
        self.assertAlmostEqual(float(advantage[0, 0]), 2.6, places=6)
        self.assertAlmostEqual(float(returns[0, 0]), 4.6, places=6)


class PPOTests(unittest.TestCase):
    def test_cli_trains_and_exports_with_512_environments(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as temp:
            result = subprocess.run([sys.executable, 'scripts/rl_train.py', '--output', temp,
                '--difficulty', 'hard', '--mode', 'fast', '--num-envs', '512', '--seconds', '.2',
                '--initialization', 'random', '--seed', '41'], cwd=root,
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = Path(temp)
            candidate = json.loads((output/'candidate.json').read_text(encoding='utf-8'))
            status = json.loads((output/'status.json').read_text(encoding='utf-8'))
            metadata = json.loads((output/'model.meta.json').read_text(encoding='utf-8'))
            checkpoint = torch.load(output/'training-checkpoint.pt', weights_only=False)
            self.assertEqual(candidate['training']['numEnvs'], 512)
            self.assertEqual(metadata['numEnvs'], 512)
            self.assertEqual(status['state'], 'awaiting-validation')
            self.assertGreater(status['totalSteps'], 0)
            self.assertEqual(checkpoint['env']['y'].shape, (512,))

    def test_five_input_saved_policies_and_checkpoints_require_a_new_run(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as temp:
            legacy=Path(temp)/'five-input.pt'
            torch.save({'modelProfile':'speed-516'},legacy)
            with self.assertRaisesRegex(ValueError,'different input profile'):
                train(Path(temp)/'resume',mode='fast',num_envs=32,seconds=.1,
                      initialization='checkpoint',checkpoint_path=legacy)
            with self.assertRaisesRegex(ValueError,'lookahead-616'):
                train(Path(temp)/'warm',mode='fast',num_envs=32,seconds=.1,
                      initialization='saved',model_path=root/'artifacts/model-516.json')
    def test_actor_has_lookahead616_shape_and_gradients(self):
        actor = Policy()
        old = [p.detach().clone() for p in actor.parameters()]
        logits, values, _ = actor(torch.rand(8, 6))
        loss = logits.square().mean() + values.square().mean()
        loss.backward()
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        optimizer.step()
        self.assertEqual(tuple(actor.fc1.weight.shape), (16, 6))
        self.assertEqual(tuple(actor.actor.weight.shape), (1, 16))
        self.assertGreater(sum(float((a-b).abs().sum().detach()) for a,b in zip(old,actor.parameters())), 0)

    def test_short_job_exports_isolated_ppo_artifacts_and_resumes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as temp:
            out = Path(temp) / 'job'
            candidate = train(out, difficulty='normal', mode='fast', num_envs=32, seconds=4,
                              initialization='random', seed=41)
            self.assertEqual(candidate['training']['dataKind'], 'reinforcement-learning-ppo')
            self.assertEqual(candidate['architecture'], [6, 16, 1])
            self.assertEqual(candidate['training']['episodeLimitSeconds'], 600)
            expected = {'candidate.json','model.css','model.meta.json','parity-inputs.json','replay.json',
                        'status.json','training-checkpoint.pt'}
            self.assertTrue(expected.issubset({p.name for p in out.iterdir()}))
            status = json.loads((out/'status.json').read_text())
            self.assertEqual(status['state'], 'awaiting-validation')
            self.assertGreaterEqual(status['updates'], 1)
            meta = json.loads((out/'model.meta.json').read_text())
            self.assertEqual(meta['trainingMode'], 'fast')
            self.assertEqual(meta['numEnvs'], 32)
            checkpoint = out/'training-checkpoint.pt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(saved['episodeLimitSeconds'], 600)
            self.assertEqual(status['bestScore'], saved['bestScore'])
            self.assertGreaterEqual(status['bestScore'], max(saved['scores'], default=0))
            self.assertTrue(status['bestScoreComplete'])
            # A historical record must survive after it falls outside the rolling 100 scores.
            saved['bestScore'] = 10000
            saved['scores'] = [0] * 100
            saved['episodeLimitSeconds'] = 300  # Older checkpoints resume under the new horizon.
            torch.save(saved, checkpoint)
            resumed = Path(temp) / 'resumed'
            train(resumed, difficulty='normal', mode='fast', num_envs=32, seconds=4,
                  initialization='checkpoint', checkpoint_path=checkpoint, seed=999)
            resumed_status = json.loads((resumed/'status.json').read_text())
            self.assertGreaterEqual(resumed_status['updates'], status['updates'])
            self.assertGreater(resumed_status['totalSteps'], status['totalSteps'])
            self.assertEqual(resumed_status['bestScore'], 10000)
            self.assertTrue(resumed_status['bestScoreComplete'])
            self.assertEqual(torch.load(resumed/'training-checkpoint.pt', weights_only=False)['episodeLimitSeconds'], 600)
            # A decision or PPO update already in progress can finish just beyond the budget.
            self.assertLessEqual(resumed_status['elapsedSeconds'], resumed_status['durationSeconds'] + 1)
            recovered = Path(temp) / 'recovered'
            train(recovered, difficulty='normal', mode='fast', num_envs=32,
                  seconds=saved['elapsedSeconds'] + .1, initialization='checkpoint',
                  checkpoint_path=checkpoint, seed=999, resume_elapsed=True)
            recovered_state = torch.load(recovered/'training-checkpoint.pt', weights_only=False)
            self.assertGreater(recovered_state['elapsedSeconds'], saved['elapsedSeconds'])
            self.assertAlmostEqual(recovered_state['lifetimeElapsedSeconds'], recovered_state['elapsedSeconds'])
            self.assertGreater(recovered_state['steps'], saved['steps'])
            # An older checkpoint can only provide a known lower bound, not an all-time record.
            saved.pop('bestScore'); saved.pop('bestScoreComplete')
            saved['scores'] = [7000] + [0] * 99
            torch.save(saved, checkpoint)
            legacy = Path(temp) / 'legacy'
            train(legacy, difficulty='normal', mode='fast', num_envs=32, seconds=.1,
                  initialization='checkpoint', checkpoint_path=checkpoint, seed=999)
            legacy_status = json.loads((legacy/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(legacy_status['bestScore'], 7000)
            self.assertFalse(legacy_status['bestScoreComplete'])


class GameParityTests(unittest.TestCase):
    def _js_trace(self, mode, decisions=100):
        source = f"""
import {{ Game }} from './src/js/game.js';
import {{ SessionController }} from './src/js/session-controller.js';
const game = new Game({{seed:12345,difficulty:'normal'}});
const trace=[]; let session;
const ai={{ready:true,threshold:.5,reset(){{}},evaluate(s){{
  const requested = {mode!r} === 'always' ? true : s.birdY > s.nextPipeGapCenterY + 10;
  trace.push({{state:s,pipes:game.pipes.map(p=>({{...p}})),requested}});
  return {{probability:requested?1:0}};
}},recordDecision(action,requested){{
  const t=trace.at(-1);t.action=action;t.canJump=(session.tick-session.lastJumpTick)*game.config.fixedStepSeconds*1000>=100-1e-8;
}},fail(){{}}}};
const recorder={{endSession(){{}},beginSession(){{}},record(){{}}}};
session=new SessionController(game,recorder,ai);session.start(12345,'ai','normal');
for(let i=0;i<{decisions*6}&&game.status==='running';i++)session.step();
console.log(JSON.stringify({{trace,final:game.getState(),pipes:game.pipes}}));
"""
        result = subprocess.run(['node','--input-type=module','-e',source], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def _assert_env_matches(self, env, js):
        state = js['state']
        self.assertAlmostEqual(env.y[0], state['birdY'], places=8)
        self.assertAlmostEqual(env.vy[0], state['birdVelocityY'], places=8)
        self.assertEqual(int(env.score[0]), state['score'])
        self.assertAlmostEqual(env.elapsed[0], state['elapsedSeconds'], places=8)
        self.assertAlmostEqual(env.speed[0], state['pipeSpeed'], places=8)
        count = int(env.pipe_count[0])
        self.assertEqual(count, len(js['pipes']))
        np.testing.assert_allclose(env.px[0,:count], [p['x'] for p in js['pipes']], atol=1e-8)
        np.testing.assert_allclose(env.gaps[0,:count], [p['gapCenterY'] for p in js['pipes']], atol=1e-8)
        np.testing.assert_array_equal(env.passed[0,:count], [p['passed'] for p in js['pipes']])
        observation=env.observe()[0]
        self.assertEqual(len(observation),6)
        self.assertAlmostEqual(float(observation[5]),state['followingPipeGapCenterY']/CONFIG['height'],places=6)

    def test_actual_sessioncontroller_actions_score_and_collision_trace(self):
        root = Path(__file__).resolve().parents[1]
        for mode in ('always','track'):
            js = self._js_trace(mode)
            env = VectorFlightEnv(1, seed=77)
            env.reset(seeds=[12345])
            self._assert_env_matches(env, js['trace'][0])
            accepted_ticks=[]
            for i, frame in enumerate(js['trace']):
                self.assertEqual(bool(env.can_jump()[0]), frame['canJump'])
                self.assertEqual(bool(frame['action']), bool(frame['requested'] and frame['canJump']))
                if frame['action']: accepted_ticks.append(int(env.tick[0]))
                if i + 1 == len(js['trace']) and js['final']['status'] != 'gameover': break
                _, _, done, _ = env.step([frame['requested']])
                if i + 1 < len(js['trace']): self._assert_env_matches(env, js['trace'][i+1])
                else:
                    self.assertTrue(done[0])
                    self.assertAlmostEqual(env.y[0], js['final']['birdY'], places=8)
                    self.assertAlmostEqual(env.vy[0], js['final']['birdVelocityY'], places=8)
                    self.assertEqual(int(env.score[0]), js['final']['score'])
                if done[0]: break
            if mode == 'always':
                self.assertTrue(env.alive[0] is False or js['final']['status'] == 'gameover')
                self.assertGreaterEqual(len(accepted_ticks), 1)
                self.assertTrue(all(b-a >= 12 for a,b in zip(accepted_ticks,accepted_ticks[1:])))
            else:
                self.assertGreaterEqual(max((f['state']['score'] for f in js['trace']), default=0), 1)


if __name__ == '__main__':
    unittest.main()
