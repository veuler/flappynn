"""PPO training for the isolated lookahead-616 RL Lab job format."""
import argparse
from collections import deque
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli

from scripts.difficulties import HARD_RULES
from scripts.export_css import export_bundle
from scripts.model import load_model, replace_file_with_retry, write_json
from scripts.rl_env import CONFIG, EPISODE_SECONDS, VectorFlightEnv


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 16)
        self.actor = nn.Linear(16, 1)
        self.critic = nn.Linear(6, 64)
        self.critic2 = nn.Linear(64, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.actor(h).squeeze(-1), self.critic2(torch.tanh(self.critic(x))).squeeze(-1), h


def _atomic_torch_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp)
        replace_file_with_retry(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def _pack_actor(actor):
    with torch.no_grad():
        return {'w1': actor.fc1.weight.cpu().numpy().tolist(), 'b1': actor.fc1.bias.cpu().numpy().tolist(),
                'w2': actor.actor.weight.cpu().numpy().reshape(-1).tolist(),
                'b2': float(actor.actor.bias.cpu().item())}


def _load_actor(actor, model):
    with torch.no_grad():
        actor.fc1.weight.copy_(torch.tensor(model['w1'], dtype=torch.float32))
        actor.fc1.bias.copy_(torch.tensor(model['b1'], dtype=torch.float32))
        actor.actor.weight.copy_(torch.tensor(model['w2'], dtype=torch.float32).reshape(1, 16))
        actor.actor.bias.fill_(float(model['b2']))


def _model_from_actor(actor, template, difficulty, mode, num_envs, seed, seconds, updates):
    model = copy.deepcopy(template)
    model.update(_pack_actor(actor))
    model['architecture'] = [6, 16, 1]
    model['version'] = 3; model['modelProfile'] = 'lookahead-616'
    model['createdAt'] = datetime.now(timezone.utc).isoformat()
    training = {'dataKind': 'reinforcement-learning-ppo', 'method': 'clipped Proximal Policy Optimization',
        'difficulty': difficulty, 'mode': mode, 'numEnvs': int(num_envs), 'seed': int(seed),
        'trainingSeconds': float(seconds), 'updates': int(updates),
        'episodeLimitSeconds': EPISODE_SECONDS,
        'reward': 'pipe passed +1, collision -1, survival +0.02 per simulated second',
        'checkpointSelection': 'latest PPO policy; browser parity remains an explicit validation gate'}
    if difficulty == 'hard': training['difficultyConfig'] = dict(HARD_RULES)
    model['training'] = training
    return model


def _actor_snapshot(actor, observations):
    with torch.no_grad():
        x = torch.as_tensor(observations, dtype=torch.float32)
        logits, _, hidden = actor(x)
        return logits.cpu().numpy(), hidden.cpu().numpy()


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma=.99, gae_lambda=.95):
    """GAE bootstraps through time limits, but never through collisions."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    adv = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        bootstrap = (~terminated[t]).astype(np.float32)
        continuation = (~(terminated[t] | truncated[t])).astype(np.float32)
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        carry = delta + gamma * gae_lambda * continuation * carry
        adv[t] = carry
    return adv, adv + values


def _save_replay(path, replay):
    write_json(path, replay)


def train(output, difficulty='normal', mode='fast', num_envs=32, seconds=60,
          initialization='random', model_path=None, checkpoint_path=None, seed=None, resume_elapsed=False):
    output = Path(output).resolve()
    if difficulty not in ('normal', 'hard') or mode not in ('live', 'fast'):
        raise ValueError('Invalid difficulty or mode.')
    if mode == 'live': num_envs = 1
    if num_envs not in (1, 32, 64, 128, 256, 512) or seconds <= 0:
        raise ValueError('Invalid environment count or duration.')
    if initialization not in ('random', 'saved', 'checkpoint'):
        raise ValueError('Invalid initialization.')
    if initialization == 'saved' and not model_path:
        raise ValueError('Saved initialization requires --model.')
    if initialization == 'checkpoint' and not checkpoint_path:
        raise ValueError('Checkpoint initialization requires --checkpoint.')
    if resume_elapsed and initialization != 'checkpoint':
        raise ValueError('Continuing the original time budget requires a checkpoint.')
    output.mkdir(parents=True, exist_ok=True)
    seed = secrets.randbits(32) if seed is None else int(seed)
    if not 0 <= seed <= 0xffffffff: raise ValueError('seed must be an unsigned 32-bit integer.')
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    actor = Policy().to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    template_path = ROOT / ('artifacts/model-616-hard.json' if difficulty == 'hard' else 'artifacts/model-616.json')
    template = load_model(template_path)
    if initialization == 'saved':
        warm = load_model(model_path)
        if warm['architecture'] != [6, 16, 1]: raise ValueError('Saved initialization requires a lookahead-616 model.')
        _load_actor(actor, warm)
        template = warm
    env = VectorFlightEnv(num_envs, difficulty, seed=seed, horizon_seconds=EPISODE_SECONDS)
    max_replay_frames = math.ceil(env.horizon_seconds / (CONFIG['decisionIntervalMs'] / 1000)) + 1
    steps = updates = episodes = 0
    best_score = 0; best_score_complete = True
    scores = deque(maxlen=100); rewards_window = deque(maxlen=100)
    history = deque(maxlen=100)
    policy_version = 0; entropy_stat = 0.0; actor_delta = 0.0
    lifetime_prior = 0.0
    elapsed_offset = 0.0
    replay_frames = []
    replay_episode = 1
    replay_seed = int(env.seed[0])
    replay_published = 0.0
    episode_reward0 = 0.0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint.get('modelProfile') != 'lookahead-616':
            raise ValueError('Checkpoint belongs to a different input profile; start a new six-input run.')
        if checkpoint.get('difficulty') != difficulty or checkpoint.get('mode') != mode or checkpoint.get('numEnvs') != num_envs:
            raise ValueError('Checkpoint environment settings do not match this job.')
        actor.load_state_dict(checkpoint['actor']); optimizer.load_state_dict(checkpoint['optimizer'])
        seed = checkpoint.get('seed', seed)
        steps = checkpoint['steps']; updates = checkpoint['updates']; episodes = checkpoint['episodes']
        policy_version = checkpoint['policyVersion']; lifetime_prior = checkpoint.get('lifetimeElapsedSeconds', checkpoint.get('elapsedSeconds', 0.0))
        if resume_elapsed:
            elapsed_offset = float(checkpoint.get('elapsedSeconds', 0.0))
            lifetime_prior -= elapsed_offset
        scores.extend(checkpoint.get('scores', [])); rewards_window.extend(checkpoint.get('rewards', []))
        # Legacy checkpoints only retain 100 scores, so their maximum is a lower bound.
        best_score = int(checkpoint.get('bestScore', max(scores, default=0)))
        best_score_complete = checkpoint.get('bestScoreComplete', 'bestScore' in checkpoint)
        history.extend(checkpoint.get('history', []))
        torch.set_rng_state(checkpoint['torchRng']); np.random.set_state(checkpoint['numpyRng'])
        env_state = checkpoint.get('env')
        if env_state:
            for key, value in env_state.items():
                if key == 'rng': env.rng.bit_generator.state = value
                else: setattr(env, key, value.copy() if isinstance(value, np.ndarray) else value)
        replay_frames = checkpoint.get('replayFrames', [])
        replay_episode = checkpoint.get('replayEpisode', 1); replay_seed = checkpoint.get('replaySeed', int(env.seed[0]))
        episode_reward0 = checkpoint.get('episodeReward0', 0.0)
        if 'episodeReward' in checkpoint: env.episode_reward = checkpoint['episodeReward'].copy()
    started = time.monotonic(); elapsed_prior = elapsed_offset; active_elapsed = elapsed_offset
    rollout_len = 128 if mode == 'fast' else 32
    last_status = 0.0; last_checkpoint = 0.0

    def status(state):
        record = {'state': state, 'difficulty': difficulty, 'mode': mode, 'numEnvs': num_envs,
            'durationSeconds': seconds, 'elapsedSeconds': float(active_elapsed), 'totalSteps': int(steps),
            'episodes': int(episodes), 'updates': int(updates),
            'meanScore': float(np.mean(scores)) if scores else 0.0,
            'bestScore': int(best_score), 'bestScoreComplete': bool(best_score_complete),
            'meanReward': float(np.mean(rewards_window)) if rewards_window else 0.0,
            'entropy': float(entropy_stat), 'actorWeightDelta': float(actor_delta),
            'policyVersion': int(policy_version), 'history': list(history)}
        write_json(output / 'status.json', record)

    def checkpoint_save():
        env_state = {key: getattr(env, key).copy() for key in
            ('seed','random_state','y','vy','score','elapsed','tick','last_jump_tick','episode_reward','speed','alive','px','gaps','passed','pipe_count')}
        env_state['rng'] = copy.deepcopy(env.rng.bit_generator.state)
        payload = {'version': 2, 'modelProfile': 'lookahead-616', 'actor': actor.state_dict(), 'critic': {'critic': actor.critic.state_dict(), 'critic2': actor.critic2.state_dict()},
            'optimizer': optimizer.state_dict(), 'torchRng': torch.get_rng_state(), 'numpyRng': np.random.get_state(),
            'steps': steps, 'updates': updates, 'episodes': episodes, 'policyVersion': policy_version,
            'bestScore': int(best_score), 'bestScoreComplete': bool(best_score_complete),
            'elapsedSeconds': active_elapsed, 'lifetimeElapsedSeconds': lifetime_prior + active_elapsed,
            'difficulty': difficulty, 'mode': mode, 'numEnvs': num_envs,
            'durationSeconds': seconds, 'seed': seed, 'scores': list(scores), 'rewards': list(rewards_window),
            'episodeLimitSeconds': env.horizon_seconds,
            'history': list(history), 'env': env_state, 'episodeReward': env.episode_reward.copy(), 'replayFrames': replay_frames,
            'replayEpisode': replay_episode, 'replaySeed': replay_seed, 'episodeReward0': episode_reward0}
        _atomic_torch_save(payload, output / 'training-checkpoint.pt')

    status('running')
    if mode == 'live': time.sleep(.05)  # The first decision follows Game.start()'s first 50 ms interval.
    rollout = {k: [] for k in ('obs','action','old_logp','value','reward','next_value','terminated','truncated','mask')}
    try:
        while active_elapsed < seconds and not (output / 'stop').exists():
            if (output / 'pause').exists():
                active_elapsed = elapsed_prior + (time.monotonic() - started)
                status('paused'); checkpoint_save()
                while (output / 'pause').exists() and not (output / 'stop').exists():
                    time.sleep(.1)
                elapsed_prior = active_elapsed
                started = time.monotonic()
                if (output / 'stop').exists(): break
                status('running')
            decision_started = time.monotonic()
            obs = env.observe()
            mask = env.can_jump()
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32)
                logits, value, hidden = actor(x)
                dist = Bernoulli(logits=logits)
                sampled = dist.sample().cpu().numpy().astype(np.int64)
                action = sampled * mask.astype(np.int64)
                logp = dist.log_prob(torch.as_tensor(action, dtype=torch.float32)).cpu().numpy()
                entropy = dist.entropy().cpu().numpy()
            frame_elapsed = env.elapsed.copy()
            frame_score = env.score.copy()
            frame_pipes0 = [{'x': float(px), 'gapCenterY': float(g), 'passed': bool(p)}
                            for px, g, p in zip(env.px[0, :env.pipe_count[0]], env.gaps[0, :env.pipe_count[0]], env.passed[0, :env.pipe_count[0]])]
            # Match SessionController: actions are evaluated after six physics steps.
            next_obs, reward, done, info = env.step(action)
            # Only accepted policy actions affect velocity; cooldown decisions have no actor gradient.
            next_values = np.zeros(num_envs, dtype=np.float32)
            with torch.no_grad():
                continuing = ~(info['terminated'] | info['truncated'])
                if continuing.any():
                    next_values[continuing] = actor(torch.as_tensor(next_obs[continuing], dtype=torch.float32))[1].cpu().numpy()
                if info['truncated'].any():
                    ids = np.flatnonzero(info['truncated'])
                    next_values[ids] = actor(torch.as_tensor(info['terminal_observation'][ids], dtype=torch.float32))[1].cpu().numpy()
            rollout['obs'].append(obs); rollout['action'].append(action); rollout['old_logp'].append(logp)
            rollout['value'].append(value.cpu().numpy()); rollout['reward'].append(reward)
            rollout['next_value'].append(next_values); rollout['terminated'].append(info['terminated'])
            rollout['truncated'].append(info['truncated']); rollout['mask'].append(mask)
            steps += num_envs
            if mode == 'live':
                time.sleep(max(0, .05 - (time.monotonic() - decision_started)))
            active_elapsed = elapsed_prior + (time.monotonic() - started)
            # Capture actual pre-action observations and the exact policy snapshot used.
            # Reserve one frame for the terminal state of the full-length flight.
            if len(replay_frames) < max_replay_frames - 1:
                i = 0
                logit_i = float(logits[i].item()); prob_i = float(torch.sigmoid(logits[i]).item())
                replay_frames.append({'t': float(frame_elapsed[i]), 'y': float(env.y[i]),
                    'vy': float(env.vy[i]), 'pipeSpeed': float(env.speed[i]), 'score': int(frame_score[i]),
                    'status': 'running',
                    'pipes': frame_pipes0, 'inputs': obs[i].astype(float).tolist(), 'hidden': hidden[i].cpu().numpy().astype(float).tolist(),
                    'logit': logit_i, 'probability': prob_i, 'requestedAction': int(sampled[i]),
                    'action': int(action[i]), 'canJump': bool(mask[i]), 'reward': float(reward[i]),
                    'policyVersion': int(policy_version)})
                episode_reward0 += float(reward[i])
                now = time.monotonic()
                if mode == 'live' and now - replay_published >= .2:
                    _save_replay(output/'replay.json', {'version': 1, 'episodeId': replay_episode,
                        'seed': int(replay_seed), 'difficulty': difficulty, 'source': 'live-training',
                        'architecture': [6,16,1], 'generatedAt': datetime.now(timezone.utc).isoformat(),
                        'complete': False, 'frames': replay_frames})
                    replay_published = now
            if done.any():
                for i in np.flatnonzero(done):
                    episodes += 1; scores.append(int(info['score'][i]))
                    best_score = max(best_score, int(info['score'][i]))
                    rewards_window.append(float(info['episode_reward'][i]))
                    if i == 0:
                        if replay_frames:
                            terminal_logit, terminal_hidden = _actor_snapshot(actor, info['terminal_observation'][i:i+1])
                            terminal_pipes = [{'x': float(px), 'gapCenterY': float(g), 'passed': bool(p)}
                                for px, g, p in zip(env.px[i, :env.pipe_count[i]], env.gaps[i, :env.pipe_count[i]], env.passed[i, :env.pipe_count[i]])]
                            logit_value = float(terminal_logit[0])
                            if len(replay_frames) < max_replay_frames:
                                replay_frames.append({'t': float(env.elapsed[i]), 'y': float(env.y[i]), 'vy': float(env.vy[i]),
                                'pipeSpeed': float(env.speed[i]), 'score': int(info['score'][i]),
                                'status': 'gameover' if info['terminated'][i] else 'truncated', 'terminal': True,
                                'pipes': terminal_pipes, 'inputs': info['terminal_observation'][i].astype(float).tolist(),
                                'hidden': terminal_hidden[0].astype(float).tolist(), 'logit': logit_value,
                                'probability': float(1 / (1 + math.exp(-max(-80, min(80, logit_value))))),
                                'requestedAction': 0, 'action': 0, 'canJump': False,
                                'reward': 0.0, 'policyVersion': int(policy_version)})
                            replay = {'version': 1, 'episodeId': replay_episode, 'seed': int(replay_seed),
                                'difficulty': difficulty, 'source': 'live-training' if mode == 'live' else 'training-replay',
                                'architecture': [6,16,1], 'generatedAt': datetime.now(timezone.utc).isoformat(),
                                'complete': True, 'frames': replay_frames}
                            if mode == 'live' or time.monotonic() - replay_published >= 1:
                                _save_replay(output/'replay.json', replay); replay_published = time.monotonic()
                        replay_frames = []; replay_episode += 1; episode_reward0 = 0.0
                env.auto_reset(done)
                if 0 in np.flatnonzero(done): replay_seed = int(env.seed[0])
            if len(rollout['reward']) >= rollout_len:
                rewards = np.asarray(rollout['reward']); values = np.asarray(rollout['value']); nextvals = np.asarray(rollout['next_value'])
                adv, returns = generalized_advantage(rewards, values, nextvals,
                    np.asarray(rollout['terminated']), np.asarray(rollout['truncated']))
                batch_obs = torch.tensor(np.concatenate(rollout['obs']), dtype=torch.float32)
                batch_action = torch.tensor(np.concatenate(rollout['action']), dtype=torch.float32)
                batch_logp = torch.tensor(np.concatenate(rollout['old_logp']), dtype=torch.float32)
                batch_adv = torch.tensor(adv.reshape(-1), dtype=torch.float32)
                batch_ret = torch.tensor(returns.reshape(-1), dtype=torch.float32)
                batch_mask = torch.tensor(np.concatenate(rollout['mask']), dtype=torch.bool)
                if batch_mask.any(): batch_adv[batch_mask] = (batch_adv[batch_mask] - batch_adv[batch_mask].mean()) / (batch_adv[batch_mask].std(unbiased=False) + 1e-8)
                before = torch.cat([p.detach().flatten().clone() for p in (actor.fc1.weight, actor.fc1.bias, actor.actor.weight, actor.actor.bias)])
                ent_values = []
                for _ in range(4):
                    for indices in torch.randperm(len(batch_obs)).split(256):
                        logits_b, values_b, _ = actor(batch_obs[indices])
                        distribution = Bernoulli(logits=logits_b)
                        new_logp = distribution.log_prob(batch_action[indices])
                        ratio = torch.exp(new_logp - batch_logp[indices])
                        active_mask = batch_mask[indices]
                        policy_loss = torch.zeros((), dtype=torch.float32)
                        if active_mask.any():
                            s1 = ratio[active_mask] * batch_adv[indices][active_mask]
                            s2 = torch.clamp(ratio[active_mask], .8, 1.2) * batch_adv[indices][active_mask]
                            policy_loss = -torch.minimum(s1, s2).mean()
                        value_loss = .5 * (values_b - batch_ret[indices]).pow(2).mean()
                        entropy_b = distribution.entropy()
                        entropy_loss = entropy_b[active_mask].mean() if active_mask.any() else torch.zeros(())
                        loss = policy_loss + .5 * value_loss - .01 * entropy_loss
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), .5); optimizer.step()
                        if active_mask.any(): ent_values.append(float(entropy_loss.detach()))
                after = torch.cat([p.detach().flatten() for p in (actor.fc1.weight, actor.fc1.bias, actor.actor.weight, actor.actor.bias)])
                actor_delta = float(torch.mean(torch.abs(after - before)).item())
                entropy_stat = float(np.mean(ent_values)) if ent_values else 0.0
                updates += 1; policy_version += 1
                history.append({'update': updates, 'totalSteps': steps,
                    'meanScore': float(np.mean(scores)) if scores else 0.0,
                    'meanReward': float(np.mean(rewards_window)) if rewards_window else 0.0,
                    'entropy': entropy_stat, 'actorWeightDelta': actor_delta})
                rollout = {k: [] for k in rollout}
                checkpoint_save(); last_checkpoint = time.monotonic()
            if time.monotonic() - last_status >= .5:
                status('running'); last_status = time.monotonic()
        status('evaluating')
        # Publish a partial actual episode if training ended before env zero completed.
        if replay_frames:
            replay = {'version': 1, 'episodeId': replay_episode, 'seed': int(replay_seed),
                'difficulty': difficulty, 'source': 'live-training' if mode == 'live' else 'training-replay',
                'architecture': [6,16,1], 'generatedAt': datetime.now(timezone.utc).isoformat(),
                'complete': False, 'frames': replay_frames}
            _save_replay(output/'replay.json', replay)
        training_seconds = float(active_elapsed)
        candidate = _model_from_actor(actor, template, difficulty, mode, num_envs, seed, training_seconds, updates)
        write_json(output/'candidate.json', candidate)
        export_bundle(output/'candidate.json', output/'model.css', output/'parity-inputs.json', output/'model.meta.json')
        metadata_path = output/'model.meta.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(trainingMode=mode, numEnvs=int(num_envs), trainingSeconds=training_seconds)
        write_json(metadata_path, metadata)
        checkpoint_save()
        status('awaiting-validation')
        return candidate
    except Exception:
        status('failed')
        checkpoint_save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--difficulty', choices=['normal','hard'], required=True)
    parser.add_argument('--mode', choices=['live','fast'], required=True)
    parser.add_argument('--num-envs', type=int, choices=[1,32,64,128,256,512], default=32)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--initialization', choices=['random','saved','checkpoint'], default='random')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume-elapsed', action='store_true', help='Continue the original time budget when recovering an interrupted job.')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    try:
        train(args.output,args.difficulty,args.mode,args.num_envs,args.seconds,args.initialization,args.model,args.checkpoint,args.seed,args.resume_elapsed)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr); return 2
    return 0


if __name__ == '__main__': raise SystemExit(main())
