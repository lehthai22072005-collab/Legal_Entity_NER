"""Prepare every unique usable annotation export from the authorized Drive root."""
import collections,hashlib,json,random,re,sys,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.docx_parser import parse_docx
from src.dataset import map_body_offsets,validate,write_documents,write_json
BASE=ROOT/'data';REPORT=ROOT/'reports'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def family(title):
    s=unicodedata.normalize('NFC',title).casefold()
    s=re.sub(r'\s+số\s+(?:không số|\S*\d\S*).*$', '',s)
    s=re.sub(r'^(?:lệnh\s+)?(?:về\s+việc\s+)?công\s+bố\s+','',s)
    s=re.sub(r'^(?:bộ luật|luật|pháp lệnh|lệnh|nghị định|nghị quyết|thông tư)\s*','',s)
    s=re.sub(r'sửa đổi\s*,?\s*bổ sung\s*(?:một số\s*)?(?:điều\s*\d*\s*)?(?:của\s*)?','',s)
    return re.sub(r'\W+',' ',s).strip()

def split_groups(documents,seed=20261117):
    parent=list(range(len(documents)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for i,d in enumerate(documents):
        for k in [('source',d['docx_sha256']),('text',d['text_sha256']),('family',d['family'])]:
            if not k[1]:continue
            if k in seen:parent[find(i)]=find(seen[k])
            else:seen[k]=i
    groups=collections.defaultdict(list)
    for i,d in enumerate(documents):groups[find(i)].append(d)
    blocks=sorted(groups.values(),key=lambda g:tuple(d['document_id'] for d in g));target=round(len(documents)*.15)
    forms=collections.Counter(d['category'] for d in documents);desired={k:v*.15 for k,v in forms.items()}
    rng=random.Random(seed);best=None
    # Deterministic search over whole-group assignments, based only on counts,
    # document identity, family and source hash. No predictions or label scores.
    for _ in range(256):
        order=list(range(len(blocks)));rng.shuffle(order);states={0:[]}
        for i in order:
            n=len(blocks[i])
            for count,selected in list(states.items()):
                total=count+n
                if total<=target+3 and total not in states:states[total]=selected+[i]
        count=min(states,key=lambda n:abs(n-target));chosen=states[count]
        actual=collections.Counter(d['category'] for i in chosen for d in blocks[i])
        cost=(abs(count-target),sum(abs(actual[k]-desired[k])/max(1,desired[k]) for k in forms))
        if best is None or cost<best[0]:best=(cost,set(chosen))
    for i,g in enumerate(blocks):
        for d in g:d.update(group_id='group_'+str(i),split='validation' if i in best[1] else 'train')
    return documents

def main():
    output=BASE/'dataset'
    if output.exists():raise FileExistsError('Keep frozen datasets; use another corpus revision')
    ontology=json.loads((ROOT/'configs/ontology.json').read_text());known={e['label'] for e in ontology['entities']}
    files=[]
    for path in sorted((BASE/'data_tho').rglob('*.json')):
        rel=path.relative_to(BASE/'data_tho')
        files.append({'category':rel.parts[0],'id':str(rel),'name':path.name,
                      'path':str(path),'sha256':sha(path)})
    docx_index=collections.defaultdict(list)
    for path in sorted((BASE/'source_docx').glob('*.docx')):
        docx_index[path.stem.rsplit('_',1)[-1]].append(str(path))
    candidates=[];duplicates=[];rejected=[];projection=[];label_issues=[];sources=[];counters=collections.Counter();seen_export={};seen_doc={}
    # The preserved local fallback is already an ordinary source export.
    export_versions=collections.defaultdict(set)
    for f in files:export_versions[f['name']].add(f['sha256'])
    jsons=sorted([f for f in files if f['name'].lower().endswith('.json')],key=lambda f:(f['category'],f['name'],f['id']))
    for f in jsons:
        path=Path(f['path']);assert sha(path)==f['sha256'];a=json.loads(path.read_text(encoding='utf-8-sig'))
        name=f['name'].removesuffix('_ket_qua.json')
        if len(export_versions[f['name']])>1:
            rejected.append({'document_id':name,'source_file':f['id'],'reason':'Conflicting source exports; requires human review, not automatic path-order selection'});continue
        if not isinstance(a,dict) or not isinstance(a.get('full_text'),str) or not isinstance(a.get('entities'),list):
            rejected.append({'file_id':f['id'],'file_name':f['name'],'reason':'Not an annotation export with full_text/entities'});continue
        if f['sha256'] in seen_export:
            duplicates.append({'document_id':name,'file_id':f['id'],'same_as':seen_export[f['sha256']],'reason':'Identical raw export SHA256'});continue
        seen_export[f['sha256']]=f['id']
        if name in seen_doc:
            rejected.append({'document_id':name,'file_id':f['id'],'reason':'Conflicting nonidentical exports of same document ID; requires review'});continue
        matches=[Path(p) for p in docx_index.get(name.rsplit('_',1)[-1],[])]
        exact=[p for p in matches if p.stem==name or p.parent.name==name];matches=exact or matches
        text=a['full_text'];delimiter='--- THÔNG TIN THUỘC TÍNH & LƯỢC ĐỒ TỪ FILE JSON ---';cut=text.find(delimiter)
        if len(matches)==1:
            source=matches[0];parsed=parse_docx(source);word=parsed['canonical_text']
            body=text[:len(word)] if text.startswith(word) else (text[:cut].rstrip('\r\n') if cut>=0 else text)
            source_origin='original_docx_reparsed'
        else:
            # Training consumes reviewed annotation exports. When the physical
            # Word is unavailable, use only the exported Word body; never
            # fabricate a DOCX or fill from the external metadata appendix.
            if cut<0:
                rejected.append({'document_id':name,'category':f['category'],'file_id':f['id'],'reason':'No original DOCX and no explicit exported-body boundary; scope needs review'});continue
            source=None;word=body=text[:cut].rstrip('\r\n');source_origin='reviewed_annotation_export_body_original_docx_unavailable'
            parsed={'canonical_text':word,'docx_sha256':None,'text_sha256':hashlib.sha256(word.encode()).hexdigest(),'source_issues':[{'code':'ORIGINAL_DOCX_NOT_AVAILABLE_FOR_REPARSE','message':'Train text is the Word-body prefix of the user-provided annotation export. Physical DOCX equivalence not independently verified.'}]}
            counters['documents_using_exported_word_body_without_physical_docx']+=1
        try:mapper=map_body_offsets(body,word)
        except ValueError:
            rejected.append({'document_id':name,'category':f['category'],'file_id':f['id'],'reason':'Annotation DOCX body differs from source by non-whitespace characters','first_difference':next((i for i,(x,y) in enumerate(zip(body,word)) if x!=y),min(len(body),len(word)))});continue
        entities=[];seen=set();bad=[]
        for e in a['entities']:
            s,z=e['start'],e['end']
            if s>=len(body) or z>len(body):counters['supplemental_entities_excluded']+=1;continue
            if not(0<=s<z<=len(text) and text[s:z]==e['text']):bad.append(e['id']);continue
            if e['label'] not in known:
                label_issues.append({'document_id':name,'entity_id':e['id'],'label':e['label'],'status':'NEEDS_HUMAN_REVIEW','used_as_target':False});continue
            start,end=mapper(s,z);key=(e['label'],start,end)
            if key in seen:counters['duplicate_typed_spans_merged']+=1;continue
            seen.add(key);entities.append({**e,'start':start,'end':end,'text':word[start:end],'origin':'docx','status':'user_supplied_audited_not_independent_gold'})
            if (s,z,e['text'])!=(start,end,word[start:end]):projection.append({'document_id':name,'entity_id':e['id'],'old':[s,z],'new':[start,end],'reason':'Whitespace-only alignment; source words unchanged'})
        if bad:
            rejected.append({'document_id':name,'category':f['category'],'file_id':f['id'],'reason':'Invalid offsets inside DOCX; not automatically relabeled','entity_ids':bad});continue
        if not entities:
            rejected.append({'document_id':name,'category':f['category'],'file_id':f['id'],'reason':'No Word-side known40 entity targets; cannot assume an independently verified all-negative document'});continue
        title=a.get('document_metadata',{}).get('ten_van_ban') or next((e['text'] for e in entities if e['label']=='Tên văn bản'),name)
        d={'document_id':name,'category':f['category'],'family':family(title),'full_text':word,'docx_length':len(word),'docx_sha256':parsed['docx_sha256'],'text_sha256':parsed['text_sha256'],'export_sha256':f['sha256'],'entities':entities,'relations':[],'input_view':'docx','source_text_origin':source_origin,'source_issues':parsed['source_issues'],'review_status':'User supplied; audited mechanically; no independent Gold claim'}
        validate(d);candidates.append(d);seen_doc[name]=f['id'];sources.append({'document_id':name,'category':f['category'],'drive_export_id':f['id'],'json_path':str(path.relative_to(ROOT)),'json_sha256':f['sha256'],'source_docx_original':str(source) if source else None,'docx_sha256':parsed['docx_sha256'],'source_text_origin':source_origin})
    # Duplicates with a different identity but identical physical Word source
    # stay in the same group; do not double-count completely identical targets.
    unique=[];content_seen={}
    for d in candidates:
        signature=(d['text_sha256'],tuple(sorted((e['label'],e['start'],e['end']) for e in d['entities'])))
        if signature in content_seen:duplicates.append({'document_id':d['document_id'],'same_as':content_seen[signature],'reason':'Same Word text and exact entity targets'});continue
        content_seen[signature]=d['document_id'];unique.append(d)
    docs=split_groups(unique);output.mkdir(parents=True)
    for split,name in [('train','train'),('validation','val')]:
        write_documents(output/(name+'.json'),[d for d in docs if d['split']==split])
    support={}
    for split in ['train','validation']:
        subset=[d for d in docs if d['split']==split];c=collections.Counter(e['label'] for d in subset for e in d['entities'])
        support[split]={'documents':len(subset),'characters':sum(len(d['full_text']) for d in subset),'entities':sum(c.values()),'forms':dict(collections.Counter(d['category'] for d in subset)),'per_label':{e['label']:c[e['label']] for e in ontology['entities']}}
    write_json(REPORT/'EXCLUDED_DOCUMENTS.json',rejected);write_json(REPORT/'DUPLICATES.json',duplicates);write_json(REPORT/'OFFSET_PROJECTIONS.json',projection);write_json(REPORT/'LABEL_REVIEW.json',label_issues)
    audit={'status':'PREPARED_NOT_TRAINED','raw_json_exports_including_local_fallback':len(jsons),'usable_unique_documents':len(docs),'excluded_documents':len(rejected),'duplicates':len(duplicates),'counts':support,'excluded_entity_counts':dict(counters),'unknown_label_cases':len(label_issues),'offset_projections':len(projection),'raw_unchanged':True,'independent_test':False,'longest_documents':sorted([(len(d['full_text']),d['document_id']) for d in docs],reverse=True)[:10]}
    write_json(REPORT/'DATA_AUDIT.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ['counts']},ensure_ascii=False,indent=2));print('SPLITS',{s:v['documents'] for s,v in support.items()})
if __name__=='__main__':main()
