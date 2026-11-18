import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.pipeline import predict
from src.dataset import write_json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('docx');p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/phobert_span_all369_v1'));p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    result=predict(a.docx,a.checkpoint,device=a.device);write_json(a.output,result);print('EXTRACTED_ENTITIES',len(result['entities']),'OUTPUT',a.output)
