"""Validate dataset, offsets and sources using hashes already stored per document."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset import read_documents,dataset_sha256,validate

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    folder=ROOT/'data';dataset=folder/'dataset'
    experiment=json.loads((ROOT/'configs/experiment_all.json').read_text())
    train=read_documents(dataset/'train.json');val=read_documents(dataset/'val.json')
    assert (len(train),len(val))==(experiment['train_documents'],experiment['validation_documents'])==(314,55)
    docs={d['document_id']:d for d in train+val}
    assert len(docs)==len(train)+len(val)==experiment['unique_documents']==369
    for documents,split in [(train,'train'),(val,'validation')]:
        assert all(d['split']==split for d in documents)
    for field in ['document_id','group_id','docx_sha256','text_sha256','family']:
        a={d[field] for d in train if d.get(field)};b={d[field] for d in val if d.get(field)}
        assert not a&b,('Leakage',field)
    ontology=json.loads((ROOT/'configs/ontology.json').read_text())
    labels={e['label'] for e in ontology['entities']};assert len(labels)==40
    bundle_path=ROOT/'reports/COLAB_INPUT_MANIFEST.json'
    bundle=json.loads(bundle_path.read_text()) if bundle_path.exists() else None
    if bundle:
        assert bundle['mode']=='normalized_training_inputs_originals_kept_locally'
        assert bundle['dataset_files']=={n:sha(dataset/n) for n in ['train.json','val.json']}
        assert bundle['local_source_preflight']['source_hashes_verified'] is True
    raw_files=list((folder/'data_tho').rglob('*.json'))
    normalized_only=not raw_files and bundle is not None
    assert raw_files or normalized_only,'Missing source JSONs and no explicit normalized-input bundle'
    raw_hashes={sha(p) for p in raw_files}
    source_docx_count=export_only_count=recovered_count=recovered_whitespace_count=0
    for key,d in docs.items():
        assert hashlib.sha256(d['full_text'].encode()).hexdigest()==d['text_sha256']
        if not normalized_only:assert d['export_sha256'] in raw_hashes,('Missing/changed source export',key)
        if d['docx_sha256']:
            assert d['source_text_origin']=='original_docx_reparsed'
            if not normalized_only:
                p=folder/'source_docx'/(key+'.docx')
                assert p.is_file() and sha(p)==d['docx_sha256'],('Missing/changed DOCX',key)
            source_docx_count+=1
        else:
            assert d['source_text_origin']=='reviewed_annotation_export_body_original_docx_unavailable'
            export_only_count+=1
            recovered=folder/'source_docx'/(key+'.docx')
            if not normalized_only and recovered.exists():
                from src.docx_parser import parse_docx
                recovered_text=parse_docx(recovered)['canonical_text']
                assert ''.join(recovered_text.split())==''.join(d['full_text'].split()),('Recovered Word differs by source words',key)
                recovered_count+=1
                recovered_whitespace_count+=recovered_text!=d['full_text']
        assert d['input_view']=='docx' and len(d['full_text'])==d['docx_length'];validate(d)
        assert {e['label'] for e in d['entities']}<=labels
    result={'status':'OFFLINE_DATA_CHECK_PASS','documents':369,'train':314,'validation':55,
            'docx_sources_verified_here':0 if normalized_only else source_docx_count+recovered_count,
            'original_docx_available_at_dataset_preparation':source_docx_count,
            'exported_word_body_only_at_dataset_preparation':export_only_count,
            'recovered_docx_sources_verified_here':recovered_count,
            'recovered_whitespace_only_differences':recovered_whitespace_count,
            'independent_gold_claimed':False,'source_hashes_verified':not normalized_only,'offsets_verified':True,
            'source_manifest_required':False,
            'dataset_sha256':dataset_sha256(train,val),
            'original_source_checks':'verified_before_packaging_locally' if normalized_only else 'verified_from_original_bytes_here'}
    print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':main()
