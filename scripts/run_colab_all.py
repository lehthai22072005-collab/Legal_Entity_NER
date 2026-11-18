"""Persistent GPU worker: verify public initialization, train, evaluate and package."""
import hashlib,json,os,subprocess,sys,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='phobert_span_all369_v1';REPORT=ROOT/'reports'/RUN
REPORT.mkdir(parents=True,exist_ok=True);started=time.time()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def status(phase,**extra):
    p=REPORT/'worker_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'phase':phase,'run_name':RUN,'pid':os.getpid(),'elapsed_seconds':time.time()-started,**extra}));t.replace(p)
def main():
    if (ROOT/'checkpoints'/RUN/'model.pt').exists():raise FileExistsError('Never overwrite an existing run')
    try:
        import torch
        assert torch.cuda.is_available(),'No allocated GPU; refusing CPU training fallback'
        status('PREFLIGHT',gpu=torch.cuda.get_device_name(0))
        records=[]
        for command in [[sys.executable,'scripts/check_all_data.py']]:
            r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True);records.append({'command':command,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
            (REPORT/'PREFLIGHT.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print(r.stdout[-2000:],r.stderr[-1000:],flush=True);r.check_returncode()
        status('DOWNLOADING_OFFICIAL_PRETRAINED',gpu=torch.cuda.get_device_name(0))
        config=json.loads((ROOT/'configs/experiment_all.json').read_text())
        from huggingface_hub import hf_hub_download
        path=Path(hf_hub_download(config['official_repo'],'pytorch_model.bin',revision=config['official_revision'],local_dir=ROOT/'resources/phobert-base'))
        assert sha(path)==config['official_pretrained_sha256']
        (REPORT/'PRETRAINING_ORIGIN.json').write_text(json.dumps({'official_repository':config['official_repo'],'revision':config['official_revision'],'sha256':sha(path),'fitted_old_weights_used':False,'matches_pinned_official_file':True},indent=2))
        status('TRAINING',gpu=torch.cuda.get_device_name(0))
        command=[sys.executable,'-u','scripts/train.py','--experiment','configs/experiment_all.json','--run',RUN,'--epochs',str(config['max_epochs']),'--seed',str(config['seed'])]
        with (REPORT/'train.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('PACKAGING',gpu=torch.cuda.get_device_name(0))
        results=json.loads((REPORT/'RESULTS.json').read_text());assert results['official_pretrained_only'] and results['relation_parameters']==0
        output=ROOT.parent/'Legal_Entity_NER_All369_Trained.zip'
        with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for base in ['src','scripts','configs','tests','data','resources/phobert-base','checkpoints/'+RUN,'reports/'+RUN]:
                for p in (ROOT/base).rglob('*'):
                    if p.is_file() and not any(x in p.parts for x in ['__pycache__','.cache']) and p.suffix not in ['.bin','.safetensors','.lock']:
                        z.write(p,ROOT.name+'/'+str(p.relative_to(ROOT)))
            for name in ['README.md','app.py','requirements.txt']:z.write(ROOT/name,ROOT.name+'/'+name)
        with zipfile.ZipFile(output) as z:assert z.testzip() is None
        artifact={'path':str(output),'bytes':output.stat().st_size,'sha256':sha(output)}
        metrics={s:dict(m['entity_exact_span']['micro'],macro_f1=m['entity_exact_span']['macro_f1_supported_labels']) for s,m in results['results'].items()}
        status('COMPLETE',artifact=artifact,metrics=metrics);print('EXPERIMENT_COMPLETE',json.dumps(metrics),flush=True)
    except Exception as error:
        status('FAILED',error=str(error),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
