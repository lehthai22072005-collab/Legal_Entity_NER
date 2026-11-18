"""Materialize authenticated Drive file references without retaining download tokens."""
import argparse,concurrent.futures,hashlib,json,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def download(job):
    document_id=job['document_id'];assert Path(document_id).name==document_id
    metadata=job['metadata'];is_docx=metadata['mime_type']=='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    if is_docx:target=ROOT/'data/source_docx'/(document_id+'.docx')
    else:
        assert metadata['mime_type']=='application/msword'
        sources=list((ROOT/'data/data_tho').rglob(document_id+'_ket_qua.json'))
        assert len(sources)==1,'Ambiguous original annotation folder'
        target=sources[0].parent/Path(metadata['title']).name
    result={k:v for k,v in job.items() if k not in ['download_url']}
    result['path']=str(target.relative_to(ROOT));result['format']='docx' if is_docx else 'doc'
    try:
        with urllib.request.urlopen(job['download_url'],timeout=60) as response:raw=response.read()
        if metadata.get('size'):assert len(raw)==int(metadata['size']),'Incomplete source download'
        digest=hashlib.sha256(raw).hexdigest();pending=target.with_suffix(target.suffix+'.partial')
        if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==digest,'Never overwrite a different original source'
        else:
            pending.write_bytes(raw)
            try:
                if is_docx:
                    with zipfile.ZipFile(pending) as z:assert 'word/document.xml' in z.namelist() and z.testzip() is None
                pending.replace(target)
            finally:
                if pending.exists():pending.unlink()
        result.update(status='DOWNLOADED_ORIGINAL',sha256=digest,bytes=len(raw),magic_hex=raw[:8].hex())
    except Exception as error:
        result.update(status='DOWNLOAD_FAILED',error_type=type(error).__name__,error_code=getattr(error,'code',None))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('batch_json');a=p.parse_args();jobs=json.loads(a.batch_json)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(download,jobs))
    report=ROOT/'reports/DOCX_SOURCE_RECOVERY.json';state=json.loads(report.read_text()) if report.exists() else {'status':'DOWNLOADING_FROM_DRIVE','recovered':[]}
    index={x['document_id']:x for x in state.get('recovered',[])}
    for item in results:index[item['document_id']]=item
    state.update(status='DOWNLOADING_FROM_DRIVE',recovered=list(index.values()))
    report.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'batch':len(jobs),'downloaded':sum(x['status']=='DOWNLOADED_ORIGINAL' for x in results),'total_downloaded':sum(x['status']=='DOWNLOADED_ORIGINAL' for x in index.values()),'errors':[{'document_id':x['document_id'],'error_type':x.get('error_type'),'error_code':x.get('error_code')} for x in results if x['status']!='DOWNLOADED_ORIGINAL']}))
if __name__=='__main__':main()
