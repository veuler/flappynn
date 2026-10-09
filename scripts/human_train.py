"""Train human recordings from saved or random weights, then prepare CSS/gameplay checks."""
import argparse
from pathlib import Path
import sys
import time
import secrets
ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''):sys.path.insert(0,str(ROOT))
from scripts.dataset import load_dataset,select_difficulty
from scripts.model import load_model,write_json
from scripts.train import train_dataset,TrainingOptions
from scripts.self_train import pack,rollout,summarize
from scripts.export_css import export_bundle

def prepare(model_path,data_path,output,difficulty,threshold=.5,initialization='saved',initialization_seed=None):
    if initialization not in ('saved','random'):
        raise ValueError('Unknown human training initialization.')
    if initialization_seed is not None and (type(initialization_seed) is not int or not 0 <= initialization_seed <= 4294967295):
        raise ValueError('Initialization seed must be an unsigned 32-bit integer.')
    # A reset starts a fresh network; keep the data split independent and repeatable.
    if initialization == 'random' and initialization_seed is None:
        initialization_seed = secrets.randbits(32)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    dataset=load_dataset(data_path);selected,_=select_difficulty(dataset,difficulty)
    maximum_epochs=500
    def status(state,epoch=0):
        write_json(output/'status.json',{'state':state,'kind':'human','initialization':initialization,'difficulty':difficulty,'epoch':epoch,'maximumEpochs':maximum_epochs,
                   'sampleCount':len(selected.samples),'flightCount':len(selected.sessions),'elapsedSeconds':time.monotonic()-started,'durationSeconds':0,
                   **({'initializationSeed':initialization_seed} if initialization_seed is not None else {})})
    status('running')
    baseline=load_model(model_path)
    options=TrainingOptions(epochs=maximum_epochs,patience=60,learning_rate=.001 if initialization=='saved' else .01,difficulty=difficulty,threshold=threshold,
                            initialization_seed=initialization_seed)
    candidate,training_report=train_dataset(dataset,options,initial_model=model_path if initialization=='saved' else None,should_stop=lambda:(output/'stop').exists(),
                                           progress=lambda epoch,*_:status('running',epoch),progress_interval=1)
    write_json(output/'candidate.json',candidate);write_json(output/'human-training-report.json',training_report)
    status('evaluating',training_report['training']['epochsRun'])
    seeds=list(range(1,21))
    report={'version':1,'kind':'human','initialization':initialization,'difficulty':difficulty,'maximumSimulationSeconds':180,'stoppedEarly':(output/'stop').exists(),
            'training':candidate['training'],'humanMetrics':training_report['validation'],
            'baselineTest':summarize(rollout(pack(baseline),seeds,baseline['gameConfig'],threshold,seconds=180,difficulty=difficulty),seeds),
            'candidateTest':summarize(rollout(pack(candidate),seeds,candidate['gameConfig'],threshold,seconds=180,difficulty=difficulty),seeds)}
    write_json(output/'report.json',report)
    export_bundle(output/'candidate.json',output/'model.css',output/'parity-inputs.json',output/'model.meta.json')
    status('awaiting-validation',training_report['training']['epochsRun'])
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--difficulty',choices=['normal','hard'],required=True);parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--initialization',choices=['saved','random'],default='saved')
    parser.add_argument('--initialization-seed',type=int,help='Repeat a specific random reset; omitted resets use a fresh seed.')
    args=parser.parse_args()
    prepare(args.model,args.data,args.output,args.difficulty,args.threshold,args.initialization,args.initialization_seed)
