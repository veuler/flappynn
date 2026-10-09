const replayKey = replay => `${replay?.episodeId}:${replay?.seed}`;

// Viewing history only. This never changes the training policy or its random seed.
export class ReplayPool {
  constructor({ maxEpisodes = 8, maxFrames = 50000, random = Math.random } = {}) {
    this.maxEpisodes = maxEpisodes;
    this.maxFrames = maxFrames;
    this.random = random;
    this.clear();
  }

  clear() {
    this.episodes = [];
    this.frameCount = 0;
  }

  add(replay) {
    if (!replay?.complete || replay.source === 'live-training' || !Array.isArray(replay.frames) ||
        !replay.frames.length || replay.frames.length > this.maxFrames) return;
    const key = replayKey(replay);
    if (this.episodes.some(episode => replayKey(episode) === key)) return;
    this.episodes.push(replay);
    this.frameCount += replay.frames.length;
    while (this.episodes.length > this.maxEpisodes || this.frameCount > this.maxFrames) {
      this.frameCount -= this.episodes.shift().frames.length;
    }
  }

  pick(previous) {
    const candidates = this.episodes.filter(episode => replayKey(episode) !== replayKey(previous));
    if (!candidates.length) return null;
    return candidates[Math.min(candidates.length - 1, Math.floor(this.random() * candidates.length))];
  }

  next(previous, queued) {
    // Fast playback chooses from completed recordings before following the latest stream.
    // Live playback must continue its real environment in order.
    const pending = queued && replayKey(queued) !== replayKey(previous) ? queued : null;
    return previous?.source === 'live-training' ? pending : this.pick(previous) || pending;
  }
}
