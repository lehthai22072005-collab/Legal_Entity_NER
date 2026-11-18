"""Small dataset I/O and exact source-span validation."""
import hashlib,json,re
from pathlib import Path

def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def read_documents(path):
    path=Path(path)
    if path.suffix=='.jsonl':return read_jsonl(path)
    documents=json.loads(path.read_text())
    if not isinstance(documents,list):raise ValueError('Dataset JSON must contain an array of documents')
    return documents
def _hash_documents(h,documents):
    h.update(b'[')
    for i,d in enumerate(documents):
        if i:h.update(b',')
        h.update(json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    h.update(b']')
def documents_sha256(documents):
    h=hashlib.sha256();_hash_documents(h,documents);return h.hexdigest()
def dataset_sha256(train,val):
    h=hashlib.sha256();h.update(b'{"train":');_hash_documents(h,train);h.update(b',"val":');_hash_documents(h,val);h.update(b'}');return h.hexdigest()
def write_documents(path,documents):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        f.write('[\n')
        for i,d in enumerate(documents):
            if i:f.write(',\n')
            json.dump(d,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        f.write('\n]\n')
def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def validate(doc):
    text=doc['full_text'];entities=doc['entities'];assert len({e['id'] for e in entities})==len(entities),'Duplicate entity IDs'
    for e in entities:
        assert 0<=e['start']<e['end']<=len(text),(doc.get('document_id'),e['id'])
        assert text[e['start']:e['end']]==e['text'],(doc.get('document_id'),e['id'])
    assert not doc.get('relations',[]),'This project does not train or predict relations'
    return True

def map_body_offsets(annotation,word):
    if annotation==word:return lambda a,b:(a,b)
    # Teacher-tool exports sometimes omit a paragraph separator. Only this
    # mechanical difference can be projected; no missing legal words are fixed.
    if re.sub(r'\s+','',annotation)!=re.sub(r'\s+','',word):
        raise ValueError('Source and annotation contain different non-whitespace characters')
    amap=[i for i,c in enumerate(annotation) if not c.isspace()]
    wmap=[i for i,c in enumerate(word) if not c.isspace()]
    correspondence=dict(zip(amap,wmap))
    def project(a,b):
        source=[i for i in range(a,b) if i in correspondence]
        if not source:raise ValueError('Whitespace-only entity is not extractable')
        s,e=correspondence[source[0]],correspondence[source[-1]]+1
        # Preserve existing surrounding whitespace when the characters exist.
        left=annotation[a:source[0]];right=annotation[source[-1]+1:b]
        if left and s>=len(left) and word[s-len(left):s]==left:s-=len(left)
        if right and word[e:e+len(right)]==right:e+=len(right)
        assert re.sub(r'\s+','',annotation[a:b])==re.sub(r'\s+','',word[s:e])
        return s,e
    return project
