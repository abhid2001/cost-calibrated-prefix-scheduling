#!/usr/bin/env python3
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from prefix_scheduler.calibration import fit,validate
p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--heldout',required=True);p.add_argument('--out',required=True);a=p.parse_args()
def read(path): return [json.loads(s) for s in pathlib.Path(path).read_text().splitlines() if s.strip()]
train,held=read(a.train),read(a.heldout)
trainkeys={(x['length'],x['cached']) for x in train}; testkeys={(x['length'],x['cached']) for x in held}
if trainkeys & testkeys: raise SystemExit('train and held-out length pairs overlap')
model=fit(train); out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
(out/'predictor.json').write_text(json.dumps(model.to_dict(),indent=2)+'\n')
(out/'validation.json').write_text(json.dumps(validate(model,held),indent=2)+'\n')
