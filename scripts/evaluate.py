import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training import load_checkpoint,evaluate_model
from src.dataset import read_documents,write_json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/phobert_span_all369_v1'));p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--device',default='cpu');a=p.parse_args()
    model,extra=load_checkpoint(a.checkpoint,a.device);gold=read_documents(ROOT/'data/dataset'/(a.split+'.json'))
    metric,pred=evaluate_model(model,gold,extra['thresholds'],extra['decoder_topk']);write_json(ROOT/'reports'/'reevaluation'/(a.split+'.json'),metric)
    print(json.dumps({'documents':len(gold),'micro':metric['entity_exact_span']['micro'],'macro':metric['entity_exact_span']['macro_f1_supported_labels']},ensure_ascii=False,indent=2))
