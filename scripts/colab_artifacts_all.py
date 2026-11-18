"""Status/download through this notebook's callback; verify/recover without writes to raw data."""
import argparse,base64,hashlib,json,sys,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN='phobert_span_all369_v1'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def connection():
    from marionette_driver.marionette import Marionette
    from urllib.parse import urlsplit
    handoff=json.loads((ROOT/'colab/COLAB_HANDOFF.json').read_text())
    expected_path=urlsplit(handoff['url']).path
    assert expected_path.startswith('/drive/'),'Invalid recorded notebook URL'
    m=Marionette(port=2828);m.start_session(timeout=20)
    for h in m.window_handles:
        m.switch_to_window(h)
        try:
            if urlsplit(m.get_url()).path==expected_path:break
        except Exception:continue
    else:m.delete_session();raise RuntimeError('Open the actual NER_ALL369 notebook first')
    for f in m.find_elements('css selector','#cell-ner_all369_3 iframe[title="Cell output"]'):
        m.switch_to_frame(f)
        if m.execute_script("return typeof window.nckhPoll==='function'",sandbox=None):break
        m.switch_to_frame()
    else:m.delete_session();raise RuntimeError('Execute NER_ALL369 worker launcher before monitoring')
    m.timeout.script=55;return m
def callback(m,expression):
    r=m.execute_async_script('const done=arguments[arguments.length-1];'+expression+'.then(done,e=>done({error:String(e)}));',sandbox=None)
    if not isinstance(r,dict):
        raise RuntimeError('Colab returned no callback response; check runtime connection before treating cached progress as live')
    if 'error' in r:raise RuntimeError(r['error'])
    data=r['data']['application/json'];return json.loads(data) if isinstance(data,str) else data
def browser(command,budget):
    m=connection()
    try:
        s=callback(m,'window.nckhPoll()');folder=ROOT/'colab/received';folder.mkdir(parents=True,exist_ok=True)
        (folder/'worker_status.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
        history=s.get('history.json',[])
        print(json.dumps({'phase':s['phase'],'gpu':s.get('gpu'),'progress':s.get('progress.json'),'latest_epoch':history[-1] if history else None,'metrics':s.get('metrics'),'artifact':s.get('artifact'),'error':s.get('error'),'log_tail':s.get('log_tail','')[-1700:]},ensure_ascii=False,indent=2),flush=True)
        if command=='status':return
        assert s['phase']=='COMPLETE' and s['run_name']==RUN,'Do not download an unfinished or different run'
        (folder/'transfer_manifest.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
        for key,name in [('results','RESULTS.json'),('history','history.json')]:
            data=callback(m,f"window.nckhTransfer('{key}',0,4194304)");blob=base64.b64decode(data['base64'],validate=True)
            assert len(blob)==data['bytes']==data['total'];p=folder/name
            if p.exists():assert p.read_bytes()==blob
            else:p.write_bytes(blob)
        expected=s['artifact'];target=ROOT/'colab/Legal_Entity_NER_All369_Trained.zip';pending=target.with_suffix('.zip.partial')
        if target.exists():assert target.stat().st_size==expected['bytes'] and sha(target)==expected['sha256'];return
        offset=pending.stat().st_size if pending.exists() else 0;assert offset<=expected['bytes']
        with pending.open('ab') as f:
            for _ in range(budget):
                if offset==expected['bytes']:break
                d=callback(m,f"window.nckhTransfer('trained_zip',{offset},4194304)");blob=base64.b64decode(d['base64'],validate=True)
                assert d['offset']==offset and d['total']==expected['bytes'] and 0<len(blob)==d['bytes']
                f.write(blob);f.flush();offset+=len(blob)
        print('DOWNLOAD_BYTES',offset,'/',expected['bytes'],flush=True)
        if offset==expected['bytes']:
            assert sha(pending)==expected['sha256'];pending.replace(target);print('ARCHIVE_SHA256_PASS',expected['sha256'],flush=True)
    finally:m.delete_session()
def verify_archive_dataset(z,prefix,root,result):
    """Compare every record, accepting frozen JSONL archives and current JSON arrays."""
    from src.dataset import read_documents,dataset_sha256
    names=set(z.namelist());local={split:read_documents(root/'data/dataset'/(split+'.json')) for split in ['train','val']}
    for split in ['train','val']:
        candidates=[prefix+base+'/'+split+extension for base in ['data/dataset','data/toan_bo/dataset'] for extension in ['.json','.jsonl'] if prefix+base+'/'+split+extension in names]
        assert len(candidates)==1,('Missing or ambiguous archived dataset',split)
        text=z.read(candidates[0]).decode()
        archived=json.loads(text) if candidates[0].endswith('.json') else [json.loads(line) for line in text.splitlines() if line.strip()]
        assert archived==local[split],('Dataset documents/labels differ from training archive',split)
    if 'dataset_sha256' in result:
        assert dataset_sha256(local['train'],local['val'])==result['dataset_sha256']
    else:
        legacy=next((prefix+base+'/manifest.json' for base in ['data/dataset','data/toan_bo/dataset'] if prefix+base+'/manifest.json' in names),None)
        assert legacy and hashlib.sha256(z.read(legacy)).hexdigest()==result['dataset_manifest_sha256']
    return local

def recover():
    from src.dataset import write_json,validate
    from src.entity_task import evaluate_entities
    archive=ROOT/'colab/Legal_Entity_NER_All369_Trained.zip';manifest=json.loads((ROOT/'colab/received/transfer_manifest.json').read_text())
    assert sha(archive)==manifest['artifact']['sha256']
    prefix=ROOT.name+'/'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;names=z.namelist();assert len(names)==len(set(names))
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        result=json.loads(z.read(prefix+f'reports/{RUN}/RESULTS.json'))
        assert result['status']=='TRAINED_ENTITY_ONLY_ON_GPU' and result['official_pretrained_only'] and result['fitted_parent_checkpoint_used'] is False
        assert result['relation_parameters']==0 and (result['train_documents'],result['validation_documents'],result['test_documents'])==(314,55,0)
        assert result['hardware']['device'].startswith('cuda')
        for name,digest in result['source_hashes'].items():assert hashlib.sha256(z.read(prefix+name)).hexdigest()==digest
        gold_splits=verify_archive_dataset(z,prefix,ROOT,result)
        ontology=json.loads(z.read(prefix+'configs/ontology.json'));assert len(ontology['entities'])==40 and 'relations' not in ontology
        views=0
        for split in ['train','val']:
            gold=gold_splits[split];pred=json.loads(z.read(prefix+f'reports/{RUN}/predictions/{split}.json'));assert len(gold)==len(pred)
            assert all(g['full_text']==p['full_text'] and validate(p) for g,p in zip(gold,pred))
            assert evaluate_entities(gold,pred,ontology)['entity_exact_span']==result['results'][split]['entity_exact_span'];views+=len(gold)
        with z.open(prefix+f'checkpoints/{RUN}/model.pt') as f:model_sha=hashlib.file_digest(f,'sha256').hexdigest()
        profile=json.loads(z.read(prefix+f'checkpoints/{RUN}/inference_settings.json'));assert model_sha==profile['base_checkpoint_sha256']
        allowed=(prefix+'checkpoints/'+RUN+'/',prefix+'reports/'+RUN+'/')
        for name in names:
            if not name.startswith(allowed) or name.endswith('/'):continue
            target=ROOT/name.removeprefix(prefix)
            if target.exists():
                with z.open(name) as f:assert sha(target)==hashlib.file_digest(f,'sha256').hexdigest(),name
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(name) as source,target.open('wb') as dest:
                while b:=source.read(1048576):dest.write(b)
    verification={'status':'GPU_ARTIFACT_METRICS_VERIFIED_CPU_CHECK_PENDING','document_views':views,'checkpoint_sha256':model_sha,'train_documents':314,'validation_documents':55,'test_documents':0,'relation_parameters':0,'official_pretrained_only':True,'metrics':{s:m['entity_exact_span']['micro'] for s,m in result['results'].items()}}
    write_json(ROOT/f'reports/{RUN}/RECOVERY_VERIFICATION.json',verification);print(json.dumps(verification,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','download','recover']);p.add_argument('--chunk-budget',type=int,default=4);a=p.parse_args()
    if a.command=='recover':recover()
    else:browser(a.command,a.chunk_budget)
