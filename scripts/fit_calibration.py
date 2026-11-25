#!/usr/bin/env python3
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.calibration import fit,validate
p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--heldout',required=True);p.add_argument('--out',required=True);p.add_argument('--min-heldout-coverage',type=float,default=0.8);p.add_argument('--max-heldout-mae-ms',type=float,default=40);p.add_argument('--max-ranking-error-rate',type=float,default=.5);a=p.parse_args()
def read(path): return [json.loads(s) for s in pathlib.Path(path).read_text().splitlines() if s.strip()]
train,held=read(a.train),read(a.heldout)
if not (0<=a.min_heldout_coverage<=1 and a.max_heldout_mae_ms>=0 and 0<=a.max_ranking_error_rate<=1): p.error('invalid validation threshold')
trainids={x.get('prompt_id') for x in train if x.get('prompt_id')}; heldids={x.get('prompt_id') for x in held if x.get('prompt_id')}
if trainids or heldids:
    if not trainids or not heldids or trainids & heldids: raise SystemExit('real-prompt train and held-out IDs must be nonempty and disjoint')
else:
    trainkeys={(x['length'],x['cached']) for x in train}; testkeys={(x['length'],x['cached']) for x in held}
    if trainkeys & testkeys: raise SystemExit('train and held-out length pairs overlap')
model=fit(train); out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
(out/'predictor.json').write_text(json.dumps(model.to_dict(),indent=2)+'\n')
report=validate(model,held)
rate=(report['ranking_mistakes']/report['ranking_comparisons']) if report['ranking_comparisons'] else None
accepted=(report['covered']/report['count']>=a.min_heldout_coverage and report['mae_ms'] is not None and report['mae_ms']<=a.max_heldout_mae_ms and rate is not None and rate<=a.max_ranking_error_rate)
report['acceptance']={'accepted':accepted,'min_heldout_coverage':a.min_heldout_coverage,'max_heldout_mae_ms':a.max_heldout_mae_ms,'max_ranking_error_rate':a.max_ranking_error_rate,'observed_ranking_error_rate':rate}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
if not accepted: raise SystemExit('held-out calibration acceptance criteria not met; see '+str(out/'validation.json'))
