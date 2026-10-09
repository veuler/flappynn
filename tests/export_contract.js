import { mkdir, writeFile } from 'node:fs/promises';
import { TrainingRecorder } from '../src/js/training-recorder.js';
import { Game } from '../src/js/game.js';
import { SessionController } from '../src/js/session-controller.js';

const data = new TrainingRecorder().exportDataset();
delete data.exportedAt;
delete data.counts;
delete data.sessions;
delete data.samples;
await mkdir(new URL('./fixtures/', import.meta.url), { recursive: true });
await writeFile(new URL('./fixtures/training-contract.json', import.meta.url), `${JSON.stringify(data, null, 2)}\n`);

// Exercise the complete browser recorder -> Python dataset contract for both levels.
const game = new Game();
const recorder = new TrainingRecorder();
const session = new SessionController(game, recorder);
for (const [seed, difficulty] of [[11, 'normal'], [22, 'hard'], [33, 'normal'], [44, 'hard']]) {
  session.start(seed, 'training', difficulty);
  for (let tick = 0; tick < 720; tick++) {
    for (const pipe of game.pipes) pipe.gapCenterY = 320;
    game.bird.y = 320; game.bird.velocityY = 0;
    if (tick % 12 === 11) session.requestJump();
    session.step();
  }
  session.finish();
}
await writeFile(new URL('./fixtures/recording-flights.json', import.meta.url), `${JSON.stringify(recorder.exportDataset(), null, 2)}\n`);
for (const [profileId, suffix] of [['speed-516', '516'], ['lookahead-616', '616']]) {
const speedRecorder = new TrainingRecorder({profileId});
const speedSession = new SessionController(game,speedRecorder);
for(const [seed,difficulty] of [[11,'normal'],[22,'hard'],[33,'normal'],[44,'hard']]) {
  speedSession.start(seed,'training',difficulty);
  for(let tick=0;tick<720;tick++) {
    for(const pipe of game.pipes) pipe.gapCenterY=320;
    game.bird.y=320;game.bird.velocityY=0;
    if(tick%12===11) speedSession.requestJump();
    speedSession.step();
  }
  speedSession.finish();
}
await writeFile(new URL(`./fixtures/recording-flights-${suffix}.json`,import.meta.url),`${JSON.stringify(speedRecorder.exportDataset(),null,2)}\n`);

}
