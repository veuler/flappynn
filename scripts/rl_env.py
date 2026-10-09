"""Vectorized, deterministic Flappy Bird environments for the PPO trainer."""
import math
import numpy as np

from .difficulties import HARD_RULES

EPISODE_SECONDS = 600

CONFIG = {
    'width': 480, 'height': 640, 'birdX': 124, 'birdRadius': 14,
    'initialBirdY': 300, 'gravity': 1500, 'jumpVelocity': -440,
    'pipeSpeed': 180, 'pipeWidth': 76, 'pipeGap': 190,
    'pipeSpacing': 280, 'firstPipeX': 600, 'gapMargin': 64,
    'fixedStepSeconds': 1 / 120, 'decisionIntervalMs': 50,
    'minimumJumpIntervalMs': 100,
}


def _random(state):
    """One Mulberry32 value, matching src/js/pipes.js."""
    state = (int(state) + 0x6D2B79F5) & 0xffffffff
    value = (((state ^ (state >> 15)) * (state | 1)) & 0xffffffff)
    value ^= (value + (((value ^ (value >> 7)) * (value | 61)) & 0xffffffff)) & 0xffffffff
    return state, ((value ^ (value >> 14)) & 0xffffffff) / 4294967296


class VectorFlightEnv:
    """Batched Game replica. step() advances exactly one 50 ms decision interval."""
    def __init__(self, num_envs, difficulty='normal', seed=0, horizon_seconds=EPISODE_SECONDS):
        if difficulty not in ('normal', 'hard'):
            raise ValueError('Invalid difficulty.')
        if num_envs < 1 or horizon_seconds <= 0:
            raise ValueError('num_envs and horizon_seconds must be positive.')
        self.n = int(num_envs)
        self.difficulty = difficulty
        self.horizon_seconds = float(horizon_seconds)
        self.rng = np.random.default_rng(seed)
        self.tick_per_decision = round(CONFIG['decisionIntervalMs'] / 1000 / CONFIG['fixedStepSeconds'])
        self._allocate()
        self.reset()

    def _new_gap(self, i):
        self.random_state[i], value = _random(self.random_state[i])
        low = CONFIG['gapMargin'] + CONFIG['pipeGap'] / 2
        return low + value * (CONFIG['height'] - 2 * low)

    def reset(self, indices=None, seeds=None):
        ids = np.arange(self.n) if indices is None else np.asarray(indices, dtype=int)
        seeds = self.rng.integers(0, 2**32, size=len(ids), dtype=np.uint32) if seeds is None else np.asarray(seeds, dtype=np.uint32)
        if seeds.shape != (len(ids),): raise ValueError('seeds must have one value per reset environment.')
        self.seed[ids] = seeds
        self.random_state[ids] = self.seed[ids]
        self.y[ids] = CONFIG['initialBirdY']
        self.vy[ids] = CONFIG['jumpVelocity']  # Game.start() gives the initial impulse.
        self.score[ids] = 0
        self.elapsed[ids] = 0
        self.tick[ids] = 0
        self.last_jump_tick[ids] = 0
        self.episode_reward[ids] = 0.0
        self.speed[ids] = CONFIG['pipeSpeed']
        self.alive[ids] = True
        self.pipe_count[ids] = 3
        self.px[ids] = 0; self.gaps[ids] = 0
        self.px[ids, :3] = np.array([CONFIG['firstPipeX'] + k * CONFIG['pipeSpacing'] for k in range(3)])
        self.passed[ids] = False
        for i in ids:
            self.gaps[i, :3] = [self._new_gap(i) for _ in range(3)]
        # SessionController advances six fixed ticks before its first policy decision.
        for _ in range(self.tick_per_decision):
            self._physics_tick(ids)
        return self.observe()

    def _allocate(self):
        n = self.n
        self.seed = np.zeros(n, dtype=np.uint32)
        self.random_state = np.zeros(n, dtype=np.uint32)
        self.y = np.zeros(n); self.vy = np.zeros(n); self.score = np.zeros(n, dtype=np.int64)
        self.elapsed = np.zeros(n); self.tick = np.zeros(n, dtype=np.int64)
        self.last_jump_tick = np.zeros(n, dtype=np.int64); self.episode_reward = np.zeros(n, dtype=np.float64)
        self.speed = np.zeros(n); self.alive = np.ones(n, dtype=bool)
        self.px = np.zeros((n, 4)); self.gaps = np.zeros((n, 4)); self.passed = np.zeros((n, 4), dtype=bool)
        self.pipe_count = np.full(n, 3, dtype=np.int64)

    def start(self, seed=None):
        """Initialize lazily allocated state; constructor uses this through reset()."""
        self._allocate()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.reset()

    def observe(self):
        c = CONFIG
        # Match nextPipe(): include a pipe until its right edge passes the bird's left edge.
        active = ((self.px + c['pipeWidth'] >= c['birdX'] - c['birdRadius']) &
                  (np.arange(self.px.shape[1])[None, :] < self.pipe_count[:, None]))
        idx = np.argmax(active, axis=1)
        rows = np.arange(self.n)
        distance = np.maximum(0, self.px[rows, idx] - c['birdX'])
        gap = self.gaps[rows, idx]
        # Live environments retain at least one following pipe, including offscreen pipes.
        following_gap = self.gaps[rows, np.minimum(idx + 1, self.pipe_count - 1)]
        return np.column_stack((np.clip(self.y / c['height'], 0, 1),
            np.clip(self.vy / 1200, -1, 1), np.clip(distance / 600, 0, 1),
            gap / c['height'], np.clip(self.speed / 270, 0, 1), following_gap / c['height'])).astype(np.float32)

    def can_jump(self):
        since_jump = self.tick - self.last_jump_tick
        return self.alive & (since_jump * CONFIG['fixedStepSeconds'] * 1000 >= CONFIG['minimumJumpIntervalMs'] - 1e-8)

    def _physics_tick(self, indices=None):
        c = CONFIG; dt = c['fixedStepSeconds']; active = self.alive
        if indices is not None:
            selected = np.zeros(self.n, dtype=bool); selected[np.asarray(indices, dtype=int)] = True
            active = active & selected
        if not active.any():
            return np.zeros(self.n, dtype=bool), np.zeros(self.n, dtype=bool)
        self.tick[active] += 1
        self.elapsed[active] += dt
        self.vy[active] += c['gravity'] * dt
        self.y[active] += self.vy[active] * dt
        if self.difficulty == 'hard':
            target = c['pipeSpeed'] * np.minimum(1 + self.score * HARD_RULES['speedGrowthPerPoint'], HARD_RULES['maxSpeedMultiplier'])
            self.speed[active] += (target[active] - self.speed[active]) * min(dt / HARD_RULES['speedResponseSeconds'], 1)
        self.px[active] -= self.speed[active, None] * dt
        passed_now = np.zeros(self.n, dtype=bool)
        # Match Game's remove-then-append loops; it may hold four pipes briefly.
        for i in np.flatnonzero(active):
            while self.pipe_count[i] and self.px[i, 0] + c['pipeWidth'] < 0:
                count = int(self.pipe_count[i])
                self.px[i, :count-1] = self.px[i, 1:count]
                self.gaps[i, :count-1] = self.gaps[i, 1:count]
                self.passed[i, :count-1] = self.passed[i, 1:count]
                self.pipe_count[i] -= 1
            while self.px[i, self.pipe_count[i]-1] < c['width'] + c['pipeSpacing']:
                count = int(self.pipe_count[i])
                if count >= self.px.shape[1]: raise RuntimeError('Pipe capacity exceeded.')
                self.px[i, count] = self.px[i, count-1] + c['pipeSpacing']
                self.gaps[i, count] = self._new_gap(i)
                self.passed[i, count] = False
                self.pipe_count[i] += 1
        hit = active & ((self.y - c['birdRadius'] <= 0) | (self.y + c['birdRadius'] >= c['height']))
        # Circle-to-rectangle distance is the exact JS helper, including tangency.
        for j in range(self.px.shape[1]):
            dx = np.maximum(self.px[:, j] - c['birdX'], np.maximum(0, c['birdX'] - (self.px[:, j] + c['pipeWidth'])))
            top = self.gaps[:, j] - c['pipeGap'] / 2
            bottom = self.gaps[:, j] + c['pipeGap'] / 2
            dy_top = np.maximum(self.y - top, 0)
            dy_bottom = np.maximum(bottom - self.y, 0)
            pipe_hit = ((dx ** 2 + dy_top ** 2 <= c['birdRadius'] ** 2) |
                        (dx ** 2 + dy_bottom ** 2 <= c['birdRadius'] ** 2))
            hit |= active & (self.pipe_count > j) & pipe_hit
        terminated = active & hit
        self.alive[terminated] = False
        # Game awards points only after collision checks succeed.
        for i in np.flatnonzero(active & ~terminated):
            count = int(self.pipe_count[i])
            newly = (~self.passed[i, :count]) & (self.px[i, :count] + c['pipeWidth'] < c['birdX'] - c['birdRadius'])
            if newly.any():
                self.score[i] += int(newly.sum())
                self.passed[i, :count][newly] = True
                passed_now[i] = True
        return terminated, passed_now

    def step(self, actions):
        actions = np.asarray(actions, dtype=bool)
        if actions.shape != (self.n,):
            raise ValueError(f'Expected actions of shape ({self.n},).')
        before_score = self.score.copy()
        before_alive = self.alive.copy()
        cooled = (self.tick - self.last_jump_tick) * CONFIG['fixedStepSeconds'] * 1000 >= CONFIG['minimumJumpIntervalMs'] - 1e-8
        accepted = actions & self.alive & cooled
        self.vy[accepted] = CONFIG['jumpVelocity']
        self.last_jump_tick[accepted] = self.tick[accepted]
        collision = np.zeros(self.n, dtype=bool)
        alive_ticks = np.zeros(self.n, dtype=np.int32)
        for _ in range(self.tick_per_decision):
            still = self.alive.copy()
            collision_tick, _ = self._physics_tick()
            collision |= collision_tick
            alive_ticks += still.astype(np.int32)
        reward = (self.score - before_score).astype(np.float32) + alive_ticks * (0.02 / 120)
        reward[collision] -= 1.0
        terminated = collision
        truncated = self.alive & (self.elapsed >= self.horizon_seconds - 1e-9)
        done = terminated | truncated
        terminal_obs = np.zeros((self.n, 6), dtype=np.float32)
        if done.any():
            terminal_obs[done] = self.observe()[done]
        info = {'terminated': terminated, 'truncated': truncated, 'terminal_observation': terminal_obs,
                'score': self.score.copy(), 'episode_seconds': self.elapsed.copy(),
                'seed': self.seed.copy(), 'alive_before': before_alive}
        self.episode_reward += reward
        info['episode_reward'] = self.episode_reward.copy()
        return self.observe(), reward, done, info

    def auto_reset(self, done):
        ids = np.flatnonzero(done)
        if len(ids):
            self.reset(ids)
        return self.observe()
