import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import train
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='phobert_span_all369_v1');p.add_argument('--epochs',type=int,default=12);p.add_argument('--seed',type=int,default=20261117);p.add_argument('--experiment',default='configs/experiment_all.json');a=p.parse_args();train(ROOT,a.run,a.epochs,a.seed,experiment=a.experiment)
