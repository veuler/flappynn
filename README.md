# Neural Flappy

A local Flappy Bird experiment that runs neural-network inference in [model.css](src/models/lookahead-616/model.css), which stores the exported weights and computes weighted sums, ReLU, and sigmoid. The bird’s physics run in JavaScript, while training runs in Python with CPU PyTorch.




https://github.com/user-attachments/assets/cc129fd2-349e-4843-8361-89ea8381b56b




Two labs share one local server:

- **6 → 16 → 1 Lab** at [http://localhost:3030/](http://localhost:3030/) (`src/lab-616.html`). Human recording, action imitation, and PPO.
- **5 → 16 → 1 Lab** at [http://localhost:3030/lab-516.html](http://localhost:3030/lab-516.html). Human recording, action imitation, and evolutionary search.

The home page redirects to the 6 → 16 → 1 lab and keeps the query string and fragment.

## Requirements

- **Windows.** Both training services start Python from `.venv\Scripts\python.exe`. That is the Windows virtual-environment layout. A `bin/python` environment on another operating system will not be found.
- **Node.js 20 or newer.** Tested with Node.js 24.11.0. Node’s built-in test runner is enough; there is no dependency install step.
- **64-bit Python 3.11.** Tested with Python 3.11.9. Use 64-bit Python 3.11 for the pinned CPU wheels.
- **A current browser** with CSS `@property` numeric registration and `exp()` inside `calc()`. On startup the page evaluates a `--nn-probe` sigmoid. If that probe fails, CSS inference stops. There is no JavaScript neural-network fallback.
- **CPU only.** `requirements.txt` installs the PyTorch CPU build. No GPU is required.

## Run locally

```powershell
git clone https://github.com/veuler/flappynn.git
cd flappynn
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm start
```

Open [http://localhost:3030/](http://localhost:3030/). The process binds to `127.0.0.1` and prints `http://127.0.0.1:3030`. Both `localhost` and `127.0.0.1` are accepted when the page origin matches the host you opened. Leave the server running while a job is active.

`npm install` is unnecessary for the current `package.json`, because it lists no npm packages.

## Architecture, weights, and a fresh random run

These are three different things.

**Architecture** is the shape. A 6 → 16 → 1 network has 6 inputs, 16 ReLU hidden units, and 1 sigmoid output (129 actor parameters). A 5 → 16 → 1 network has 5 inputs and the same hidden and output layers (113 parameters). The shape does not say where the numbers came from.

**Weights** are the stored numbers. A warm-start file keeps newly added connections at zero so an older function is unchanged. An imitation file was fit to recorded jump/wait choices. A PPO file was produced by reinforcement learning. The bundled set contains each of those kinds. It does not contain a still-random network, and it does not mean every file was trained on every input it exposes.

**Random initialization** is a new draw you start in the lab. Fresh PPO builds a new actor and uses `artifacts/model-616.json` or `artifacts/model-616-hard.json` only as the contract template: input names, normalization, and game settings. The template’s warm-start or imitation numbers are replaced by the new actor. Fresh human imitation also trains a new network, then still scores it against the saved baseline for that difficulty. The saved file has to be present for that comparison. Choosing random does not delete the saved model; a later promotion or explicit save is what replaces it.

## How one flight is computed

JavaScript owns the world. The playfield is 480×640. Gravity is 1500 px/s² and a jump sets vertical velocity to −440 px/s. Physics steps at 120 Hz. The network is asked every 50 ms. An accepted jump then waits 100 ms before another jump can apply. Normal pipes move at 180 px/s. Hard mode raises the speed *target* by 1% per point, up to 270 px/s (1.5×), and the actual speed approaches that target over 2 seconds. Gap size, gravity, and jump velocity stay the same.

Saved play writes normalized inputs into CSS custom properties. CSS evaluates the weighted sums, ReLU, and sigmoid. The bird jumps when the sigmoid reaches the threshold, default 0.50. Training and training replay use Python’s policy outputs. The browser does not substitute a JavaScript copy of the network when CSS math is unavailable.

The five-input vector is bird height `y/640`, vertical velocity `vy/1200`, next-pipe distance `distance/600`, gap height `gapY/640`, and horizontal speed `speed/270`, each clipped to its contracted range. The sixth input is the following pipe’s gap height, also divided by 640. A recording row is those inputs plus the accepted action: `1` jump or `0` wait.

Python trains on CPU and exports weights, a CSS bundle, and metadata. PPO also keeps a separate critic, 6 → 64 tanh → 1, for value estimates during training. The critic is never part of the CSS file.

## Controls

- **Space, Up, W, or a primary click/tap** starts a flight and jumps. Key repeat is ignored.
- **P** pauses and resumes.
- **Who is flying:** free flight, record my inputs, or AI once a verified CSS model has loaded.
- **Difficulty:** Normal (steady 180 px/s) or Hard (accelerating speed target).
- **Next flight:** repeat one seed, or draw a new course. Seeds are unsigned 32-bit integers.
- **Jump threshold:** 0.01 to 0.99, default 0.50, enabled after the CSS model loads. Human imitation and evolutionary search train and compare at the threshold selected in that lab. Saved PPO playback uses 0.50 even when training sampled softer probabilities.

The 6 → 16 → 1 page can fly either the recording model or the RL policy. Those two saved families do not overwrite each other.

## Recordings stay in the page

Recorded samples live in that page’s memory, up to 100,000 samples. Refreshing, closing, or navigating away drops them. Download the JSON before you do that. Import replaces an empty buffer and must match that lab’s profile, physics, timing, and input ranges; a five-input recording will not import into the six-input lab. The import limit is 16 MiB. End the current flight before importing, and download-and-clear before importing over samples you want to keep.

Each browser tab has its own recording buffer. Training jobs do not: see [One job per service](#one-job-per-service).

## Human imitation

Both labs can imitate your actions.

1. Choose **Me · record my inputs** and the difficulty you want the model to learn.
2. Record at least **200 samples across 2 different seeds** for that difficulty. The page checks its recording buffer before enabling the button; the server checks the uploaded dataset again when training is requested.
3. Choose **Continue saved weights** or **Reset: random weights**.
4. Train. The fit uses Adam, a whole-seed train/validation split (individual frames are not shuffled across seeds), and early stopping when validation stops improving.
5. The browser then checks CSS arithmetic and plays the saved model and the candidate on the same courses. Promotion follows the rule in [Promotion and its limits](#promotion-and-its-limits).

A random reset still copies the saved model into the job and uses it as the baseline. Hard mode uses the hard slot when that metadata file exists. If the hard slot is missing entirely, the server falls back to the normal slot for the baseline. Deleting the baseline files leaves human training with nothing to compare against.

Continuing saved weights fine-tunes that baseline. Resetting starts the optimizer from a new draw. Either way, the on-disk model changes only when promotion installs a candidate.

## 5 → 16 → 1 evolutionary search

The five-input lab also runs an elitist Gaussian search over its 113 parameters. It is separate from PPO. The six-input lab’s old parameter-search endpoint is retired and tells you to use the PPO panel.

- Durations are **5, 10, or 15 minutes**. There is no 30-minute evolutionary budget.
- Search starts from the saved model and the threshold selected in the lab.
- Search flights use fresh seeds and stop at 120 simulated seconds.
- Checkpoint selection uses **8 validation seeds** and 180-second flights. Seeds 1–20 are held out of the search.
- The final paired comparison is seeds **1 through 20**, up to 180 simulated seconds, for the baseline and the candidate.
- More minutes do not promise a higher score. The search report’s own limitation is that the validation seeds which picked the checkpoint are not an untouched test set.

After that comparison, CSS parity and agreement between the Python simulator and browser CSS play are required before anything is installed.

## 6 → 16 → 1 PPO

PPO lives in the 6 → 16 → 1 lab, under Training & Settings. The panel is loaded from `src/templates/rl-workspace.html`.

- **Fast** runs 32, 64, 128, 256, or 512 environments in one batched CPU simulation. It does not open one browser per bird. A larger batch is not a promise of linear speedup.
- **Live** runs exactly one environment at game speed. Frames are polled up to five times per second.
- Budgets are **5, 10, 15, or 30 minutes**.
- **Fresh random weights** draws a new actor. It still reads `artifacts/model-616.json` (normal) or `artifacts/model-616-hard.json` (hard) as the contract template.
- **Saved RL policy** initializes the actor from `artifacts/model-rl-616-normal.json` or `artifacts/model-rl-616-hard.json`, with a new critic and optimizer. No normal PPO policy is bundled, so a new clone has only the hard RL file. Start a normal run with fresh random weights; the normal “saved” choice becomes available after you verify and save that run.
- **Latest checkpoint** restores the actor, critic, optimizer, RNG, and environment state when the difficulty, mode, and environment count match the latest job. The new job gets a new time budget. Checkpoints are created inside the job directory. The server does not accept an uploaded checkpoint path.

Pause and resume affect training. Replay pause is separate. Stop ends the run early and leaves the current policy available for verification. Training samples actions from the policy probability. Saved CSS playback uses a fixed 0.50 threshold, so the two can score differently. Decisions blocked by the jump cooldown do not receive a fake jump gradient.

The reward is **+1** for a passed pipe, **−1** for a collision, and **+0.02** per simulated second survived. Each episode is capped at 600 simulated seconds. Hitting that cap bootstraps the critic instead of treating the time limit as a death. The cap is flight time, not the 5/10/15/30-minute training budget.

Fast mode labels playback **RANDOM REPLAY**. The page keeps up to 8 completed flights from the recorded environment, and 50,000 frames, which is enough for about four full 600-second flights. It then picks another completed flight at random. Live mode keeps following its real stream. On-screen means use the latest 100 completed training episodes. They are not the 20-seed promotion benchmark.

PPO does not auto-install a policy because a score moved. Use **Verify CSS policy**, then **Save policy**. Verification checks arithmetic against Python. Save checks the candidate, CSS, reference states, report, and metadata against the server’s parity receipt, then installs that difficulty’s RL slot. A failed install rolls back from the job’s previous-slot copy. Saving an RL policy does not replace the 5 → 16 → 1 or human-imitation slots.

## Promotion and its limits

Human imitation and 5 → 16 → 1 evolutionary search share one acceptance rule. CSS parity and simulator agreement are required first. The score rule then compares the same 20 seeds, numbered 1 through 20, each run up to 180 simulated seconds:

- Average score must rise, and the new average must be at least **5%** higher.
- Median must not fall.
- P10 (the low end of those 20 scores) must keep at least **90%** of the baseline P10.
- At most **one** paired seed may regress severely: score under half the baseline on that seed, and a loss of at least 10 points.
- A baseline that scores zero still has to improve. Matching zero does not install a dead model.

If every check passes, the candidate is installed automatically. If the comparison finishes without passing, the previous model stays installed and **Use candidate permanently** can still save the candidate, regressions included. A candidate whose weights and threshold match the baseline skips that comparison and is not offered as a new save. The job keeps a copy of the previous model from the moment training started.

That 20-seed set is a fixed comparison, not a large held-out study. Repeating runs and threshold changes on seeds 1–20 does not create an independent final test. Evolutionary validation seeds were already used to pick a checkpoint. The separate benchmark page (`benchmark.html`) is another limited check: seeds 1–20, **normal** courses, **60** simulated seconds, CSS decisions, with the same warning about tuning on those seeds. RL’s rolling 100-episode means are training telemetry. None of these numbers is a leaderboard, and this README does not advertise scores.

`model-parity.html` checks CSS against a matching Python reference for a profile (`?profile=speed-516` or `?profile=lookahead-616`, plus `&difficulty=hard` when needed). The standalone page needs `parity-inputs.json` beside the selected bundle; the shipped hard `speed-516` bundle does not include that fixture. Training candidates generate their own references for validation. Bundled metadata records a passed arithmetic check at absolute tolerance `1e-4`. That check is about hidden activations, logits, and probabilities. It is not evidence of playing skill.

## One job per service

`npm start` runs three services in one process:

| Service | HTTP path | One active job means |
| --- | --- | --- |
| 5 → 16 → 1 learning | `/api/learning-516` | One human-imitation or evolutionary job |
| 6 → 16 → 1 learning | `/api/learning-616` | One human-imitation job |
| PPO | `/api/rl` | One PPO job |

A second start on the same service gets “already running” until that job finishes or is stopped. Every tab pointed at this server shares those jobs and the saved files under `artifacts/` and `src/models/`. Installing a model changes what the other tabs load from this server.

Another clone is a different directory, a different `.venv`, and a different process. It does not see this clone’s jobs or saved files. If port 3030 is already taken, stop the other server or start this one with a different `PORT`.

The server answers training routes only for `localhost` and `127.0.0.1`, and only when the page origin matches that host. The retired `/api/learning` path, and parameter search on the six-input service, respond that those entry points are gone.

## Where local training writes

Published weight files live here:

| Role | Weights | CSS and metadata |
| --- | --- | --- |
| 5 → 16 → 1 normal | `artifacts/model-516.json` | `src/models/speed-516/` |
| 5 → 16 → 1 hard | `artifacts/model-516-hard.json` | `src/models/speed-516/hard/` |
| 6 → 16 → 1 normal | `artifacts/model-616.json` | `src/models/lookahead-616/` |
| 6 → 16 → 1 hard | `artifacts/model-616-hard.json` | `src/models/lookahead-616/hard/` |
| PPO normal, after you save one | `artifacts/model-rl-616-normal.json` | `src/models/rl-616/normal/` |
| PPO hard | `artifacts/model-rl-616-hard.json` | `src/models/rl-616/hard/` |

Job output stays on the machine and is not part of the public tree:

- 5 → 16 → 1 jobs: `src/learning-516/<job-id>/`
- 6 → 16 → 1 human jobs: `src/learning-616/<job-id>/`
- PPO jobs: `src/learning-rl-616/<job-id>/`, including status, replay, candidate JSON, CSS, parity files, and `training-checkpoint.pt`

## Bundled models

Each bundled policy is a weights JSON file plus the matching CSS bundle and metadata under `src/models/`. Parity input fixtures and browser parity reports are included where they exist beside that bundle: normal `speed-516`, and both `lookahead-616` difficulties. The hard `speed-516` and hard `rl-616` directories ship CSS and metadata. Personal recordings, checkpoints, and job history are not included.

| File | Difficulty | What this file is |
| --- | --- | --- |
| `artifacts/model-516.json` | Normal | `warm-start-legacy`. A function-preserving transfer. Added parameters start at zero, including the pipe-speed input. This file is a starting point, not a trained use of pipe speed. |
| `artifacts/model-516-hard.json` | Hard | `human-gameplay-unverified`. Imitation weights whose training run began from random initialization. The recordings are not in the repository. The label is not a skill certificate. |
| `artifacts/model-616.json` | Normal | `warm-start-lookahead`. A function-preserving transfer of the normal five-input warm-start. Every sixth-column weight is 0, so the following-gap input does not change that five-input function. The pipe-speed column remains 0 as well, because that is what the parent warm-start stored. |
| `artifacts/model-616-hard.json` | Hard | `human-gameplay-unverified`. Imitation fine-tune. The recordings are not in the repository. |
| `artifacts/model-rl-616-hard.json` | Hard | PPO. Metadata records a fast run with 512 environments and a 5-minute budget. There is no bundled normal PPO policy; start a normal run from fresh random weights and save it yourself. |

Test fixtures under `tests/fixtures/` are scripted or synthetic. They keep the contracts honest. They are not a human gameplay dataset and they do not prove how any bundled model plays.

## Tests

```powershell
npm test
npm run test:ml
```

`npm test` runs Node’s built-in test runner on the JavaScript suites named in `package.json` (game timing, recordings, promotion, PPO service, replay, and related browser-side contracts).

`npm run test:ml` runs `tests/export_contract.js`, then `.venv\Scripts\python.exe -m unittest` for `tests.test_ml`, `tests.test_css`, `tests.test_self_train`, `tests.test_profiles`, and `tests.test_rl`. The Python command uses the Windows virtual environment, so create and install that environment first.

Passing tests means the contracts and the scripted fixtures behaved as written. It does not certify human recordings or published playing strength.

## File map

```
LICENSE                  PolyForm Noncommercial 1.0.0
.gitattributes           preserves exported model bytes for SHA-256 validation
README.md
package.json             npm start / npm test; no npm dependencies
requirements.txt         CPU PyTorch and NumPy
scripts/                 servers, trainers, CSS export, dataset checks
  serve.js               localhost static server plus the three training services
  learning-service.js    human imitation and 5-16-1 evolutionary jobs
  rl-service.js          PPO jobs
  human_train.py
  self_train.py
  rl_train.py
  rl_env.py
  export_css.py
  create_lookahead_model.py
  build_lab_page.py      rewrites src/lab-516.html from src/templates/lab-516.html
src/
  index.html             redirect to the 6-16-1 lab
  lab-616.html           6-16-1 lab
  lab-516.html           5-16-1 lab
  benchmark.html         20-seed, 60-second, normal-course CSS check
  model-parity.html      CSS versus Python arithmetic check
  learning-check.html    browser side of promotion validation
  styles.css
  rl.css
  js/                    physics, input, CSS inference, labs, promotion rule
  templates/             5-16-1 page template and PPO panel template
  models/speed-516/      5-16-1 CSS bundles
  models/lookahead-616/  6-16-1 imitation / warm-start CSS bundles
  models/rl-616/hard/    bundled PPO hard CSS bundle
artifacts/
  model-516.json
  model-516-hard.json
  model-616.json
  model-616-hard.json
  model-rl-616-hard.json
tests/                   JavaScript suites, Python suites, scripted fixtures
```

A clone does not include personal notes, the docs directory, recordings, checkpoints, or training-job history. New jobs you run are written under the `src/learning*` directories and are meant to stay on that machine. Retired bundles (`src/models/rl-516/`, `src/models/hard/`, and the old root CSS/metadata names `src/model.css` and `src/model.meta.json`) are also left out of the public tree.

Saving into a bundled model slot modifies tracked weights and CSS files. Those changes appear in `git diff`; share them only when you deliberately want to publish a new policy. Newly created Normal PPO weights and its CSS directory are ignored by default. The root allowlist also keeps new root-level notes out of Git unless you explicitly opt them in. `.gitattributes` prevents automatic line-ending conversion of exported model files, since their metadata validates exact bytes.

## Troubleshooting

- **Port 3030 is in use.** Stop the other Neural Flappy server, or start this clone with another `PORT`. Two clones do not share jobs or model files.
- **Python fails immediately.** Create the virtual environment with `python -m venv .venv` on 64-bit Windows Python 3.11, then install `requirements.txt` with `.venv\Scripts\python.exe`. The servers do not search `PATH` for another interpreter.
- **`npm run test:ml` cannot find Python.** The same Windows virtual environment has to exist and contain the CPU requirements.
- **The page says the browser cannot do the CSS calculations.** Use a current browser that passes the `--nn-probe` check. There is no JavaScript inference fallback.
- **Training says a session is already running.** That service has one shared job. Use the tab that started it, or stop it, before starting another of the same kind. The other two services can still be idle.
- **Recordings disappeared.** They were only in that page’s memory. Import a JSON you downloaded earlier.
- **Train from my recordings stays disabled.** That difficulty needs at least 200 samples and two different seeds in the current page buffer.
- **Fresh human training fails before it starts.** Random imitation still needs the saved baseline files for the comparison. Keep the bundled weights in place, or save a model into that slot first.
- **Saved normal RL policy is unavailable.** That is expected on a fresh clone. Start from fresh random weights, verify CSS, and save. Fresh random PPO still needs `artifacts/model-616.json` or `artifacts/model-616-hard.json` present as the contract template.
- **Opening HTML files directly does not work.** Start the local server with `npm start`, then open `http://localhost:3030/`; the pages use ES modules and fetch their templates and model files over HTTP.

This repository is **source-available**, not OSI open source. Personal and noncommercial experimentation is permitted. Commercial use is not granted. The license terms control. Read [LICENSE](https://github.com/veuler/flappynn/blob/main/LICENSE) and the [PolyForm Noncommercial License 1.0.0](https://github.com/polyformproject/polyform-licenses/blob/1.0.0/PolyForm-Noncommercial-1.0.0.md). PyTorch and NumPy keep their own licenses. `package.json` currently declares no npm dependencies.
