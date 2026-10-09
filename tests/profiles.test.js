import test from 'node:test';
import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { modelProfile } from '../src/js/model-profiles.js';
import { normalizeState } from '../src/js/normalization.js';
import { TrainingRecorder } from '../src/js/training-recorder.js';
import { SessionController } from '../src/js/session-controller.js';
import { Game } from '../src/js/game.js';
import { AiController, validateMetadata } from '../src/js/ai-controller.js';
import { modelPaths, createLearningService } from '../scripts/learning-service.js';
import { validateRecordingDataset } from '../src/js/recording-dataset.js';

const speed = modelProfile('lookahead-616');
test('the six-input lab rejects removed parameter-search training without launching a worker',async()=>{
  let launches=0;
  const service=createLearningService(process.cwd(),speed.id,{spawnWorker:()=>{launches++;}});
  const request=Readable.from([JSON.stringify({seconds:300,threshold:.5,difficulty:'normal'})]);
  request.method='POST';request.headers={host:'127.0.0.1:3030',origin:'http://127.0.0.1:3030','content-type':'application/json'};
  const response={writeHead(status){this.status=status;return this;},end(text){this.body=JSON.parse(text);}};
  assert.equal(await service(request,response,`${speed.api}/start`),true);
  assert.equal(response.status,410);assert.equal(launches,0);assert.match(response.body.error,/PPO/);
});
test('saved RL model selection stays in its own slots and does not fall back to recording weights',async context=>{
  const ai=new AiController({},undefined,speed.id);ai.setSavedFamily('rl');
  const sources=[];const requests=[];
  context.mock.method(globalThis,'fetch',async url=>{requests.push(url);return new Response('{}',{status:404});});
  context.mock.method(ai,'initialize',async function(base){sources.push(base);this.ready=true;this.modelBase=base;return true;});
  assert.equal(await ai.loadSaved('hard'),true);
  assert.deepEqual(requests,[`${speed.rlHardBase}model.meta.json`]);
  assert.deepEqual(sources,[speed.rlBase]);
  ai.setSavedFamily('recordings');
  assert.equal(await ai.loadSaved('normal'),true);assert.equal(sources.at(-1),speed.base);
  assert.throws(()=>ai.setSavedFamily('unknown'));
});
test('lookahead observes offscreen gaps, retains both targets during overlap and shifts after full clearance',()=>{
  const game=new Game();
  const [first,second,third]=game.pipes;
  first.gapCenterY=200;second.gapCenterY=400;third.gapCenterY=300;
  assert.ok(second.x > game.config.width);
  assert.equal(normalizeState(game.getState(),speed.normalization)[5],400/640);
  first.x=game.config.birdX-game.config.pipeWidth;
  assert.equal(game.getState().nextPipeGapCenterY,200);
  assert.equal(game.getState().followingPipeGapCenterY,400);
  first.x=game.config.birdX-game.config.birdRadius-game.config.pipeWidth-0.01;
  assert.equal(game.getState().nextPipeGapCenterY,400);
  assert.equal(game.getState().followingPipeGapCenterY,300);
  assert.throws(()=>normalizeState({...game.getState(),followingPipeGapCenterY:null},speed.normalization),/following/);
  const recorder=new TrainingRecorder({profileId:speed.id});
  const session=new SessionController(game,recorder);session.start(42,'training');session.finish();
  const data=recorder.exportDataset();
  assert.equal(data.samples[0][5],game.getState().followingPipeGapCenterY/640);
  const broken=structuredClone(data);broken.samples[0][5]=1.01;
  assert.throws(()=>validateRecordingDataset(broken,speed.id),/range/);
});
test('importing valid recordings preserves data, rejects wrong profiles and leaves existing data intact on failure',()=>{
  const source=new TrainingRecorder({profileId:speed.id});const game=new Game();const session=new SessionController(game,source);
  session.start(42,'training','hard');session.finish();const data=source.exportDataset();
  const imported=new TrainingRecorder({profileId:speed.id});imported.importDataset(data);
  assert.equal(imported.samples.length,1);assert.equal(imported.sessions[0].difficulty,'hard');
  const broken=structuredClone(data);broken.samples[0][4]=2;
  assert.throws(()=>imported.importDataset(broken));assert.equal(imported.samples.length,1);
  assert.throws(()=>validateRecordingDataset(data,'speed-516'));
  data.sessions[0].id=4;imported.importDataset(data);new SessionController(game,imported).start(1,'training','hard');
  assert.equal(imported.sessions[1].id,5);
});

test('human training rejects incompatible, insufficient or corrupt recordings before starting a worker',async()=>{
  const service=createLearningService(process.cwd(),speed.id);
  const game=new Game();const recorder=new TrainingRecorder({profileId:speed.id});const session=new SessionController(game,recorder);
  session.start(42,'training','hard');session.finish();
  for(const dataset of [{},recorder.exportDataset(),{...recorder.exportDataset(),version:2}]) {
    const request=Readable.from([JSON.stringify({dataset,difficulty:'hard',threshold:.5})]);request.method='POST';
    request.headers={host:'127.0.0.1:3030',origin:'http://127.0.0.1:3030','content-type':'application/json'};
    const response={writeHead(status){this.status=status;return this;},end(text){this.body=JSON.parse(text);}};
    await service(request,response,`${speed.api}/human`);assert.equal(response.status,400);assert.ok(response.body.error);
  }
});
test('speed input records actual accelerating physics while lookahead records the second pipe gap', () => {
  const game = new Game();
  const recorder = new TrainingRecorder({profileId:speed.id});
  const session = new SessionController(game,recorder);
  session.start(42,'training','hard');
  for(let tick=0; tick<3600; tick++) {
    game.pipes.forEach(pipe=>pipe.gapCenterY=320); game.bird.y=320; game.bird.velocityY=0;
    session.step();
  }
  session.finish();
  const dataset = recorder.exportDataset();
  assert.equal(dataset.version,4);
  assert.equal(dataset.modelProfile,speed.id);
  assert.equal(dataset.samples[0][4],2/3);
  assert.ok(dataset.samples.at(-1)[4]>2/3);
  assert.ok(dataset.samples.every(row=>row.length===7));
  assert.equal(normalizeState(game.getState()).length,4);
  assert.equal(normalizeState(game.getState(),speed.normalization).length,6);
  assert.throws(()=>normalizeState({...game.getState(),pipeSpeed:undefined},speed.normalization));
});

test('experimental AI rejects baselineOther metadata and never falls back to the baselineOther weight directory',async context=>{
  const baselineMetaSource=JSON.parse(await readFile('tests/fixtures/speed-model.meta.json'));
  assert.throws(()=>validateMetadata(baselineMetaSource,speed.id),/architecture/);
  const meta={...baselineMetaSource,version:3,modelProfile:speed.id,architecture:speed.architecture,inputNames:speed.inputNames,normalization:speed.normalization};
  assert.doesNotThrow(()=>validateMetadata(meta,speed.id));
  assert.throws(()=>validateMetadata(meta),/architecture/);
  const ai=new AiController({},undefined,speed.id);
  const sources=[];
  context.mock.method(globalThis,'fetch',async()=>new Response('{}',{status:404}));
  context.mock.method(ai,'initialize',async function(base){sources.push(base);this.modelBase=base;this.ready=true;return true;});
  assert.equal(await ai.loadSaved('hard'),true);
  assert.deepEqual(sources,[speed.base]);
  assert.equal(ai.modelBase,speed.base);
});

test('four model slots are distinct and services ignore each other’s endpoint and job history', async()=>{
  const root=process.cwd();
  for(const field of ['model','css','metadata']) assert.equal(new Set(['speed-516','lookahead-616'].flatMap(profile=>['normal','hard'].map(difficulty=>modelPaths(root,difficulty,profile)[field]))).size,4);
  const request={method:'GET',headers:{host:'127.0.0.1:3030'}};
  const response={writeHead(){return this;},end(text){this.body=JSON.parse(text);}};
  assert.equal(await createLearningService(root,'speed-516')(request,response,speed.api),false);
  assert.equal(await createLearningService(root,speed.id)(request,response,'/api/learning'),false);
  const isolated=await mkdtemp(path.resolve('artifacts','namespace-test-'));
  try {
    await mkdir(path.join(isolated,'src','learning'),{recursive:true});
    await writeFile(path.join(isolated,'src','learning','latest.json'),JSON.stringify({id:'11111111-1111-4111-8111-111111111111',state:'running'}));
    await createLearningService(isolated,speed.id)(request,response,speed.api);
    assert.equal(response.body.job,null);
  } finally {assert.ok(path.resolve(isolated).startsWith(path.resolve('artifacts')+path.sep));await rm(isolated,{recursive:true,force:true});}
});

test('experimental promotion writes only its own hard slot and rejects a baselineOther candidate',async()=>{
  const root=await mkdtemp(path.resolve('artifacts','profile-test-'));
  const id='55555555-5555-4555-8555-555555555555';
  const folder=path.join(root,'src',speed.learning,id);
  const original=await readFile('tests/fixtures/speed-model.json');
  const baselineMetaSource=JSON.parse(await readFile('tests/fixtures/speed-model.meta.json'));
  const experimental=Buffer.from(JSON.stringify({profile:'test baseline'}));
  const css=Buffer.from('test CSS');
  const digest=data=>createHash('sha256').update(data).digest('hex');
  const baselineMeta={...baselineMetaSource,version:3,architecture:speed.architecture,modelProfile:speed.id,modelSha256:digest(experimental),cssSha256:digest(css)};
  const candidate=Buffer.from(JSON.stringify({profile:'test candidate'}));
  const meta={...baselineMeta,trainingDifficulty:'hard',difficultyConfig:{speedGrowthPerPoint:.01,maxSpeedMultiplier:1.5,speedResponseSeconds:2},modelSha256:digest(candidate)};
  const call=async(service,operation,data)=>{
    const request=Readable.from([JSON.stringify(data)]);request.method='POST';request.headers={host:'127.0.0.1:3030',origin:'http://127.0.0.1:3030','content-type':'application/json'};
    const response={writeHead(status){this.status=status;return this;},end(body){this.body=JSON.parse(body);}};
    await service(request,response,`${speed.api}/${operation}`);return response;
  };
  try {
    await mkdir(path.join(folder,'baseline'),{recursive:true});await mkdir(path.join(root,'artifacts'));
    const baselineOther=modelPaths(root,'normal','speed-516');const baselineSlot=modelPaths(root,'normal',speed.id);const target=modelPaths(root,'hard',speed.id);
    await mkdir(path.dirname(baselineSlot.metadata),{recursive:true});await mkdir(path.dirname(baselineOther.metadata),{recursive:true});
    for(const [file,bytes] of [[baselineOther.model,original],[baselineOther.css,css],[baselineOther.metadata,JSON.stringify(baselineMetaSource)],
      [baselineSlot.model,experimental],[baselineSlot.metadata,JSON.stringify(baselineMeta)],
      [path.join(folder,'candidate.json'),candidate],[path.join(folder,'model.css'),css],
      [path.join(folder,'model.meta.json'),JSON.stringify({...meta,architecture:[5,16,1]})],
      [path.join(folder,'baseline','model.meta.json'),JSON.stringify(baselineMeta)],
      [path.join(root,'src',speed.learning,'latest.json'),JSON.stringify({id,state:'validating',difficulty:'hard',threshold:.5})]]) await writeFile(file,bytes);
    const baseline={modelSha256:baselineMeta.modelSha256,threshold:.5,difficulty:'hard',maximumSimulationSeconds:180,runs:Array.from({length:20},(_,i)=>({seed:i+1,score:10,survivalSeconds:20}))};
    const result={...baseline,modelSha256:meta.modelSha256,runs:baseline.runs.map(run=>({...run,score:11}))};
    await writeFile(path.join(folder,'report.json'),JSON.stringify({difficulty:'hard',baselineTest:baseline,candidateTest:result}));
    const service=createLearningService(root,speed.id);const claim=await call(service,'claim',{id});
    assert.equal((await call(service,'result',{id,token:claim.body.token,baseline,candidate:result})).status,400);
    await writeFile(path.join(folder,'model.meta.json'),JSON.stringify(meta));
    assert.equal((await call(service,'result',{id,token:claim.body.token,baseline,candidate:result})).body.job.state,'promoted');
    assert.deepEqual(await readFile(target.model),candidate);
    assert.deepEqual(await readFile(baselineOther.model),original);
    assert.deepEqual(JSON.parse(await readFile(baselineOther.metadata)),baselineMetaSource);
    assert.deepEqual(await readFile(baselineSlot.model),experimental);
  } finally { assert.ok(path.resolve(root).startsWith(path.resolve('artifacts')+path.sep));await rm(root,{recursive:true,force:true}); }
});
