"""PhoBERT entity boundaries, nested spans, character refinement and marker adapter."""
import bisect,math,random,unicodedata,re
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def balanced_binary_loss(logits,target,max_hard_negatives=128,positive_weights=None):
 """Give each observed positive class equal weight to hard negatives.

 The loss cannot shrink merely by generating thousands more easy negative
 spans. Only Train targets/predictions enter this objective.
 """
 logits=logits.float();target=target.float();losses=[]
 for k in range(logits.shape[1]):
  pos=logits[:,k][target[:,k]>0];neg=logits[:,k][target[:,k]==0];terms=[]
  if len(pos):terms.append(F.softplus(-pos).mean()*(positive_weights[k] if positive_weights is not None else 1.))
  if len(neg):
   nl=F.softplus(neg);terms.append(nl.topk(min(max_hard_negatives,len(nl))).values.mean()+.25*nl.mean())
  if terms:losses.append(sum(terms)/((1. if len(pos) else 0)+(1.25 if len(neg) else 0)))
 return torch.stack(losses).mean() if losses else logits.sum()*0

def endpoint_competition_loss(logits,target,bounds):
 """Contrast Train spans sharing an endpoint without merging nested labels."""
 terms=[]
 positive_rows=torch.nonzero(target>0).detach().cpu().tolist()
 by_class={}
 for row,k in positive_rows:by_class.setdefault(k,[]).append(row)
 for k,positives in by_class.items():
  for endpoint in [0,1]:
   for key in {bounds[row][endpoint] for row in positives}:
    rows=[i for i,b in enumerate(bounds) if b[endpoint]==key]
    pos=[i for i in positives if bounds[i][endpoint]==key];positive_set=set(pos)
    neg=[i for i in rows if i not in positive_set and target[i,k]==0]
    if pos and neg:
     negative=logits[neg,k].float().topk(min(16,len(neg))).values
     positive=logits[pos,k].float()
     terms.append(torch.logsumexp(torch.cat([positive,negative]),0)-torch.logsumexp(positive,0))
 return torch.stack(terms).mean() if terms else logits.sum()*0

def attribute_singleton_ranking_loss(start_logits,end_logits,span_logits,target,bounds,label_indices):
 """Rank a present Word attribute above other spans across the document.

 Only unambiguous single-positive Train fields participate. Missing fields,
 multiple aliases and non-attribute entity types are never invented or merged.
 """
 terms=[]
 a=torch.tensor([x[0] for x in bounds],device=span_logits.device)
 b=torch.tensor([x[1]-1 for x in bounds],device=span_logits.device)
 for column,k in enumerate(label_indices):
  if k>=12:continue
  positive=torch.where(target[:,column]>0)[0];negative=torch.where(target[:,column]==0)[0]
  if len(positive)!=1 or not len(negative):continue
  joint=F.logsigmoid(start_logits[a,k].float())+F.logsigmoid(end_logits[b,k].float())+F.logsigmoid(span_logits[:,column].float())
  chosen=joint[negative].topk(min(16,len(negative))).values
  good=joint[positive]
  terms.append(torch.logsumexp(torch.cat([good,chosen]),0)-good[0])
 return torch.stack(terms).mean() if terms else span_logits.sum()*0

def type_competition_loss(logits,target):
 """Rank every reviewed positive above competing types, preserving multilabel rows."""
 terms=[]
 for row in torch.where((target>0).any(-1))[0].tolist():
  positive=logits[row][target[row]>0].float();negative=logits[row][target[row]==0].float()
  if len(negative):terms.append(F.softplus(.5+negative.topk(min(3,len(negative))).values[:,None]-positive[None,:]).mean())
 return torch.stack(terms).mean() if terms else logits.sum()*0

def marker_rows(text,width=12,marker_only=False):
 """Source-only lexical features. No label/regex assigns a legal structure."""
 rows=[];starts=[];ends=[];offset=0
 for line in text.splitlines(keepends=True):
  body=line.rstrip('\r\n')
  if body and not body[0].isspace():
   prefix=body[:width]
   if marker_only:prefix=prefix.split()[0]
   normalized=['0' if c.isdigit() else c.lower() if len(c.lower())==1 else c for c in prefix]
   rows.append([(ord(c)%4095)+1 for c in normalized]+[0]*(width-len(prefix)))
   starts.append(offset);ends.append(offset+len(body)-1)
  offset+=len(line)
 return rows,starts,ends

def line_quote_context(text):
 """Soft source features only; never bans entities inside quotations."""
 curly=0;straight=False;offset=0;previous_blank=True;features=[]
 for line in text.splitlines(keepends=True):
  body=line.rstrip('\r\n')
  if body and not body[0].isspace():features.append([min(curly,3)/3,float(straight),float(previous_blank),offset/max(len(text),1)])
  for char in body:
   if char=='“':curly+=1
   elif char=='”':curly=max(0,curly-1)
   elif char=='"':straight=not straight
  previous_blank=not body.strip();offset+=len(line)
 return features

def mine_typed_endpoint_negatives(bounds,scores,gold,types,limit=12,minimum=.5):
 """Select model mistakes on Train; reviewed typed spans are never negatives."""
 values=scores.detach().float().cpu();chosen=[];seen=set()
 for index in values.flatten().argsort(descending=True).tolist():
  row,col=divmod(index,len(types));value=float(values[row,col])
  if value<minimum:break
  bound=bounds[row];kind=types[col];key=(bound,kind)
  if kind in gold.get(bound,set()) or key in seen:continue
  chosen.append((bound,kind));seen.add(key)
  if len(chosen)>=limit:break
 return chosen

class StructureMarkerAdapter(nn.Module):
 """Learn short line prefixes instead of relying only on clause content means."""
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(4096,8,padding_idx=0);self.out=nn.Linear(12*8,7)
  nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def enable_context(self):
  if not hasattr(self,'context'):
   self.context=nn.Linear(4,7).to(self.out.weight.device);nn.init.zeros_(self.context.weight);nn.init.zeros_(self.context.bias)
 def forward(self,text,end_types,context_v7=False,context_rows=None):
  rows,starts,ends=marker_rows(text,marker_only=context_v7);device=self.out.weight.device;n=len(text)
  zero=self.out.weight.new_zeros((n,40))
  if not rows:return zero,zero.clone()
  logits=self.out(self.emb(torch.tensor(rows,device=device)).flatten(1))
  if context_v7:logits=logits+self.context(self.out.weight.new_tensor(context_rows if context_rows is not None else line_quote_context(text)))
  full=F.pad(logits,(33,0)).to(zero.dtype)
  s=zero.index_add(0,torch.tensor(starts,device=device),full)
  mask=full.new_tensor([label in end_types for label in range(40)])
  e=zero.index_add(0,torch.tensor(ends,device=device),full*mask)
  return s,e

def label_code(label):
 s=unicodedata.normalize('NFD',label).replace('đ','d').replace('Đ','D')
 return re.sub(r'[^A-Z0-9]+','_', ''.join(c for c in s if not unicodedata.combining(c)).upper()).strip('_')

class CharCNN(nn.Module):
 def __init__(self,vocab,hidden=96):
  super().__init__();self.vocab=vocab;self.emb=nn.Embedding(len(vocab)+1,48);self.pos=nn.Embedding(64,16)
  self.proj=nn.Linear(64,hidden)
  self.layers=nn.ModuleList([nn.Sequential(nn.Conv1d(hidden,hidden,5,padding=2*d,dilation=d,groups=hidden),nn.GELU(),nn.Conv1d(hidden,hidden,1),nn.GELU()) for d in [1,2,4]])
 def forward(self,text):
  device=self.emb.weight.device;n=len(text)
  ids=torch.tensor([self.vocab.get(c,0) for c in text],device=device)
  pos=(torch.arange(n,device=device)*64//max(n,1)).clamp(max=63)
  x=self.proj(torch.cat([self.emb(ids),self.pos(pos)],dim=-1)).T.unsqueeze(0)
  for layer in self.layers:x=x+layer(x)
  return x.squeeze(0).T,[(i,i+1) for i in range(n)],[]

class PhoBERTDocument(nn.Module):
 def __init__(self,path,hidden=96,window=254,overlap=64,checkpointing=True,hf_config=None,initialize_pretrained=True):
  super().__init__()
  from transformers import AutoModel,AutoConfig
  from .tokenization import load_tokenizer
  self.tokenizer=load_tokenizer(path)
  if initialize_pretrained:self.encoder=AutoModel.from_pretrained(path,local_files_only=Path(path).exists())
  else:
   if not hf_config:raise ValueError('Checkpoint lacks its encoder architecture config')
   settings=dict(hf_config);model_type=settings.pop('model_type');self.encoder=AutoModel.from_config(AutoConfig.for_model(model_type,**settings))
  self.proj=nn.Linear(self.encoder.config.hidden_size,hidden);self.window=window;self.overlap=overlap
  self.char=nn.Embedding(4096,16);self.char_fusion=nn.Linear(hidden+16,hidden)
  self.refine=nn.Conv1d(hidden,hidden,5,padding=2,groups=hidden)
  if checkpointing:self.encoder.gradient_checkpointing_enable()
 def forward(self,text):
  from .tokenization import tokenize_source
  enc=tokenize_source(text,self.tokenizer);ids=enc['input_ids'];offsets=enc['offsets'];device=self.proj.weight.device
  if not ids:raise ValueError('Empty tokenized input')
  stride=self.window-self.overlap;parts=[];covered=0
  for start in range(0,len(ids),stride):
   stop=min(start+self.window,len(ids));seq=[self.tokenizer.bos_token_id]+ids[start:stop]+[self.tokenizer.eos_token_id]
   inp=torch.tensor([seq],device=device)
   h=self.encoder(input_ids=inp,attention_mask=torch.ones_like(inp)).last_hidden_state[0,1:-1]
   # Central ownership retains every source token exactly once, including long
   # spans whose endpoints are in entirely different encoder windows.
   left=max(covered,start);right=stop if stop==len(ids) else stop-self.overlap//2
   if right>left:parts.append(h[left-start:right-start]);covered=right
   if stop==len(ids):break
  assert covered==len(ids)
  token_h=self.proj(torch.cat(parts))
  # Source-aligned character refinement supports boundaries INSIDE a BPE token
  # and arbitrary long spans. Mapping is computed from source/token offsets,
  # never from gold boundaries. Whitespace inherits the previous token context.
  starts=[a for a,b in offsets]
  lookup=torch.tensor([max(0,bisect.bisect_right(starts,i)-1) for i in range(len(text))],device=device)
  chars=torch.tensor([ord(c)%4096 for c in text],device=device)
  char_h=self.char_fusion(torch.cat([token_h[lookup],self.char(chars)],dim=-1))
  char_h=char_h+self.refine(char_h.T.unsqueeze(0)).squeeze(0).T
  return char_h,[(i,i+1) for i in range(len(text))],enc['warnings']

class EntityExtractor(nn.Module):
 def __init__(self,config,ontology):
  super().__init__();assert config.get('entity_only') is True;assert len(ontology['entities'])==40;self.config=config;self.ontology=ontology;h=config.get('hidden',96)
  self.e_labels=[x['label'] for x in ontology['entities']];self.r_labels=[]
  self.e_index={v:i for i,v in enumerate(self.e_labels)};self.r_index={v:i for i,v in enumerate(self.r_labels)}
  self.encoder=CharCNN(config['vocab'],h) if config['encoder']=='char_cnn' else PhoBERTDocument(config['pretrained_path'],h,checkpointing=config.get('checkpointing',True),hf_config=config.get('hf_config'),initialize_pretrained=config.get('initialize_pretrained',True))
  if config['encoder']=='phobert':self.config['hf_config']=self.encoder.encoder.config.to_dict()
  boundary_extra=12 if config.get('quality_v2') else 0
  self.start=nn.Linear(h+boundary_extra,len(self.e_labels));self.end=nn.Linear(h+boundary_extra,len(self.e_labels))
  extra=8 if config.get('span_layout_features') else 0
  self.span_proj=nn.Sequential(nn.Linear(3*h+4+extra,128),nn.GELU(),nn.Dropout(.1))
  self.span_type=nn.Linear(128,len(self.e_labels))
  self.register_buffer('active_entities',torch.tensor(config.get('active_entities',[True]*len(self.e_labels)),dtype=torch.bool))
  if config.get('marker_v6'):self.enable_marker_adapter()
  if config.get('context_v7'):self.enable_context_v7()

 def enable_marker_adapter(self):
  if len(self.e_labels)!=40:raise ValueError('Marker adapter requires the fixed 40-label NEW30 ontology')
  if not hasattr(self,'marker'):self.marker=StructureMarkerAdapter().to(self.start.weight.device)
  self.config.setdefault('marker_end_labels',list(self.config.get('decoder_policy',{}).get('line_end_labels',[])))
  self.config['marker_v6']=True

 def marker_features(self,text):
  if self.config.get('marker_v6'):
   end_labels=self.config['marker_end_labels']
   end_types={self.e_index[label] for label in end_labels}
   self._marker_start,self._marker_end=self.marker(text,end_types,self.config.get('context_v7',False),getattr(self,'_source_marker_context',None))

 def enable_context_v7(self):
  self.enable_marker_adapter();self.marker.enable_context();self.config['context_v7']=True

 def encode(self,text):
  cache=getattr(self,'_inference_cache',None)
  if cache is not None and not self.training and text in cache:
   h,offsets,warnings,nl,layout=cache[text];self._text=text;self._offsets=offsets;self._newline_prefix=nl;self._boundary_layout=layout
   self.marker_features(text)
   return h,offsets,warnings
  h,offsets,warnings=self.encoder(text);self._text=text;self._offsets=offsets
  count=0;prefix=[0]
  for char in text:count+=char=='\n';prefix.append(count)
  self._newline_prefix=prefix
  if self.config.get('quality_v2'):
   layout=[];line=0;n=max(len(text),1)
   for i,c in enumerate(text):
    left=text[i-1] if i else '\n';right=text[i+1] if i+1<len(text) else '\n'
    if left=='\n':line=i
    origin,length=getattr(self,'_document_context',(0,n))
    layout.append([float(left=='\n'),float(right=='\n'),float(not left.isalnum()),float(not right.isalnum()),float(c.isdigit()),float(c.isalpha()),float(c.isupper()),float(c=='.'),float(c==')'),float(c=='\n'),min(i-line,80)/80,(i+origin)/max(length,1)])
   self._boundary_layout=h.new_tensor(layout)
  if cache is not None and not self.training:cache[text]=(h,offsets,warnings,self._newline_prefix,getattr(self,'_boundary_layout',None))
  self.marker_features(text)
  return h,offsets,warnings

 def boundary_logits(self,h):
  x=torch.cat([h,self._boundary_layout],dim=-1) if self.config.get('quality_v2') else h
  sl,el=self.start(x),self.end(x)
  if self.config.get('marker_v6'):sl=sl+self._marker_start;el=el+self._marker_end
  return sl,el

 def span_logits(self,h,bounds,pool_prefix=None,vectors=None):
  logits=self.span_type(vectors if vectors is not None else self.spans(h,bounds,pool_prefix))
  if self.config.get('marker_v6') and bounds:logits=logits+self._marker_start[[a for a,b in bounds]]
  return logits

 def spans(self,h,bounds,pool_prefix=None):
  if not bounds:return torch.empty((0,128),device=h.device)
  s=torch.tensor([b[0] for b in bounds],device=h.device);e=torch.tensor([b[1] for b in bounds],device=h.device)
  pooled=h.float() if h.dtype in (torch.float16,torch.bfloat16) else h
  prefix=pool_prefix if pool_prefix is not None else torch.cat([pooled.new_zeros((1,h.shape[1])),pooled.cumsum(0)],dim=0)
  mean=(prefix[e]-prefix[s])/(e-s).float().unsqueeze(-1);n=max(len(h),1)
  origin,length=getattr(self,'_document_context',(0,n));length=max(length,1)
  loc=torch.stack([(s+origin)/length,(e+origin)/length,(e-s).float().log1p()/10,(((s+origin)>length*.9)|((e+origin)<length*.1)).float()],dim=-1)
  if self.config.get('span_layout_features'):
   layout=[];text=self._text;off=self._offsets;nl=self._newline_prefix
   for a,b in bounds:
    left,right=off[a][0],off[b-1][1];line_start=text.rfind('\n',0,left)+1;prefix_text=text[line_start:left]
    layout.append([float(left==0 or text[left-1]=='\n'),float(right==len(text) or text[right]=='\n'),math.log1p(nl[right]-nl[left])/5,float(text[left:left+1].isdigit()),float(text[right-1:right] in '.;:)”\"'),math.log1p(right-left)/10,float(':' in prefix_text),float(prefix_text.startswith('['))])
   loc=torch.cat([loc,h.new_tensor(layout)],dim=-1)
  return self.span_proj(torch.cat([h[s],h[e-1],mean,loc],dim=-1))

 def align(self,entities,offsets):
  starts=[x[0] for x in offsets];ends=[x[1] for x in offsets];result=[];missing=[]
  for e in entities:
   a=bisect.bisect_left(starts,e['start']);b=bisect.bisect_left(ends,e['end'])
   if a<len(offsets) and b<len(offsets) and a<=b and starts[a]==e['start'] and ends[b]==e['end']:
    result.append((e,(a,b+1)))
   else:missing.append({'id':e['id'],'label':e['label'],'start':e['start'],'end':e['end']})
  return result,missing

 def loss(self,doc,rng=None):
  self._document_context=(doc.get('source_start',0),doc.get('source_length',len(doc['full_text'])))
  self._source_marker_context=doc.get('source_marker_context')
  rng=rng or random.Random(0);h,offsets,warnings=self.encode(doc['full_text']);aligned,missing=self.align(doc['entities'],offsets)
  if missing:raise ValueError('Unrepresentable gold boundaries; refusing silent label loss: '+str(missing[:4]))
  if self.config.get('quality_v2'):aligned.sort(key=lambda x:(x[1][0],x[1][1],self.e_index[x[0]['label']]))
  n=len(h);c=len(self.e_labels);device=h.device;ys=h.new_zeros((n,c));ye=ys.clone();gold={}
  ignored=doc.get('ignored_partial_entities',[])
  for e in ignored:
   k=self.e_index[e['label']];ys[e['start']:e['end'],k]=-1;ye[e['start']:e['end'],k]=-1
  for e,(a,b) in aligned:
   k=self.e_index[e['label']];ys[a,k]=1;ye[b-1,k]=1;gold.setdefault((a,b),set()).add(k)
  start_logits,end_logits=self.boundary_logits(h)
  chosen=set(torch.nonzero(((ys>0)|(ye>0)).any(-1)).flatten().tolist())
  if self.config.get('quality_v2'):
   for logits in [start_logits,end_logits]:
    target=ys if logits is start_logits else ye
    hardness=(logits.detach().sigmoid()*(target==0)).amax(-1)
    chosen.update(hardness.topk(min(n,1000)).indices.tolist())
  chosen.update(rng.sample(range(n),min(n,2500)));chosen=torch.tensor(sorted(chosen),device=device)
  active=self.active_entities
  def boundary_loss(logits,target):
   t=target[chosen][:,active];l=logits[chosen][:,active]
   if self.config.get('balanced_entity_loss'):
    weights=l.new_tensor(self.config.get('document_recall_weights',[1.]*c))[active]
    return balanced_binary_loss(l,t,positive_weights=weights)
   weights=((len(chosen)-t.sum(0))/t.sum(0).clamp(min=1)).clamp(1,100 if self.config.get('quality_v2') else 5000)
   loss=F.binary_cross_entropy_with_logits(l,t,pos_weight=weights,reduction='none')
   if self.config.get('quality_v2'):
    prob=l.sigmoid();loss=loss*torch.where(t>0,(1-prob).pow(1.5),prob.pow(1.5))
   return loss.mean()
  boundary=(boundary_loss(start_logits,ys)+boundary_loss(end_logits,ye))/2
  bounds=list(gold)
  if self.config.get('hard_negatives'):
   by_type={}
   for bound,types in gold.items():
    for k in types:by_type.setdefault(k,[]).append(bound)
   # Chimeric spans with plausible starts/ends are much harder than random
   # text. Select from Train gold only; no validation-derived correction rule.
   for k,items in by_type.items():
    if len(items)>1:
     for _ in range(min(512,len(items)*6)):
      a,_=rng.choice(items);_,b=rng.choice(items)
      if a<b and offsets[b-1][1]-offsets[a][0]<=self.config.get('max_span_chars',4096):bounds.append((a,b))
  if self.config.get('quality_v2'):
   # Mine the model's own high-score mistakes on THIS Train document. No
   # Validation predictions, labels or correction examples enter this loss.
   with torch.no_grad():
    ps=start_logits.detach().sigmoid().cpu();pe=end_logits.detach().sigmoid().cpu();mined={}
    for k in torch.where(self.active_entities)[0].tolist():
     starts=ps[:,k].topk(min(n,24)).indices.tolist();ends=pe[:,k].topk(min(n,24)).indices.tolist()
     for a in starts:
      for z in ends:
       if a<=z and z-a<self.config.get('max_span_chars',4096) and k not in gold.get((a,z+1),[]):
        bound=(a,z+1);value=float(ps[a,k]*pe[z,k]);mined[bound]=max(mined.get(bound,0.),value)
    bounds.extend(sorted(mined,key=mined.get,reverse=True)[:512])
  if self.config.get('quality_v4'):
   # Train-only endpoint negatives: adjacent line ends and character truncations.
   line_ends=[i for i in range(1,n+1) if i==n or doc['full_text'][i]=='\n'];nearby=[]
   for a,b in gold:
    z=bisect.bisect_left(line_ends,b)
    nearby.extend((a,end) for end in line_ends[max(0,z-1):z+3] if a<end and end-a<=self.config.get('max_span_chars',4096))
    nearby.extend((a,end) for end in [b-1,b+1] if a<end<=n)
   bounds.extend(nearby[:1024])
  for _ in range(max(16,len(bounds)*3)):
   if gold and rng.random()<.7:
    s,e=rng.choice(list(gold));s=max(0,min(n-1,s+rng.choice([-2,-1,0,1,2])));e=max(s+1,min(n,e+rng.choice([-2,-1,0,1,2])))
   else:s=rng.randrange(n);e=min(n,s+rng.randint(1,min(n-s,250)))
   bounds.append((s,e))
  bounds=list(dict.fromkeys(bounds));vectors=self.spans(h,bounds);target=h.new_zeros((len(bounds),c))
  for row,(a,b) in enumerate(bounds):
   for e in ignored:
    if a<e['end'] and b>e['start']:target[row,self.e_index[e['label']]]=-1
  for row,bound in enumerate(bounds):
   for k in gold.get(bound,[]):target[row,k]=1
  st=target[:,active];sw=((len(bounds)-st.sum(0))/st.sum(0).clamp(min=1)).clamp(1,8 if self.config.get('quality_v2') else 50)
  span_logits=self.span_logits(h,bounds,vectors=vectors)[:,active]
  recall_weights=span_logits.new_tensor(self.config.get('document_recall_weights',[1.]*c))[active]
  span_loss=balanced_binary_loss(span_logits,st,positive_weights=recall_weights) if self.config.get('balanced_entity_loss') else F.binary_cross_entropy_with_logits(span_logits,st,pos_weight=sw)
  attribute_ranking=span_logits.float().sum()*0
  if self.config.get('quality_v10') or self.config.get('entity_only'):
   attribute_ranking=attribute_singleton_ranking_loss(start_logits,end_logits,span_logits,st,bounds,torch.where(active)[0].tolist())
   span_loss=span_loss+.5*attribute_ranking
  if self.config.get('quality_v2'):
   ranking=[]
   for k in range(st.shape[1]):
    pos=span_logits[:,k][st[:,k]>0];neg=span_logits[:,k][st[:,k]==0]
    if len(pos) and len(neg):ranking.append(F.softplus(.5+neg.topk(min(8,len(neg))).values.mean()-pos).mean())
   if ranking:span_loss=span_loss+.25*torch.stack(ranking).mean()
  if self.config.get('quality_v4'):span_loss=span_loss+.5*endpoint_competition_loss(span_logits,st,bounds)
  type_loss=type_competition_loss(span_logits,st) if self.config.get('marker_v6') else span_logits.float().sum()*0
  if self.config.get('marker_v6'):span_loss=span_loss+.25*type_loss
  loss=boundary+span_loss
  return loss,{'boundary_loss':float(boundary.detach()),'span_loss':float(span_loss.detach()),'relation_loss':0.,'parent_loss':0.,'type_competition_loss':float(type_loss.detach()),'attribute_ranking_loss':float(attribute_ranking.detach()),'aligned_entities':len(aligned),'negative_relation_endpoints':0,'warnings':warnings}

 @torch.no_grad()
 def infer(self,text,boundary_threshold=.5,span_threshold=.5,relation_threshold=.5,max_boundaries=64,max_entities=None,oracle_entities=None,structure_boundary_threshold=None,structure_span_threshold=None,attribute_boundary_threshold=None,attribute_span_threshold=None):
  self._document_context=(0,len(text))
  self._source_marker_context=None
  if self.config.get('entity_only') and oracle_entities is not None:raise ValueError('Entity-only inference cannot accept gold/oracle entities')
  if max_entities is None:max_entities=int(self.config.get('decoder_max_entities',300))
  if not text:return {'full_text':'','entities':[],'relations':[],'model_warnings':[{'code':'EMPTY_DOCUMENT'}],'decoder_diagnostics':{'boundary_candidates_truncated':0,'entities_truncated':0,'span_width_cap_characters':self.config.get('max_span_chars',4096)},'confidence_note':'No source text to extract.'}
  self.eval();h,offsets,warnings=self.encode(text);n=len(h)
  if oracle_entities is None:
   sl,el=self.boundary_logits(h);ps=sl.sigmoid().cpu();pe=el.sigmoid().cpu();candidates={};truncated=0;policy=self.config.get('decoder_policy')
   from .decoding import candidate_boundary_masks
   for k in range(len(self.e_labels)):
    if not self.active_entities[k]:continue
    valid_s,valid_e=candidate_boundary_masks(text,offsets,self.e_labels[k],policy)
    bound_cut=structure_boundary_threshold if k>=33 and structure_boundary_threshold is not None else boundary_threshold
    if k<12 and attribute_boundary_threshold is not None:bound_cut=attribute_boundary_threshold
    starts=torch.where((ps[:,k]>=bound_cut)&torch.tensor(valid_s))[0];ends=torch.where((pe[:,k]>=bound_cut)&torch.tensor(valid_e))[0]
    if len(starts)>max_boundaries:truncated+=len(starts)-max_boundaries;starts=starts[ps[starts,k].topk(max_boundaries).indices]
    if len(ends)>max_boundaries:truncated+=len(ends)-max_boundaries;ends=ends[pe[ends,k].topk(max_boundaries).indices]
    from .document_windows import candidate_pairs
    width_limit=policy['label_max_chars'].get(self.e_labels[k],self.config.get('max_span_chars',4096)) if policy else self.config.get('max_span_chars',4096)
    nl=self._newline_prefix if policy and self.e_labels[k] in policy.get('single_line_labels',[]) else None
    for pair in candidate_pairs(starts.tolist(),ends.tolist(),offsets,width_limit,nl):
     candidates.setdefault(pair,set()).add(k)
   found=[];bounds=list(candidates);prefix=torch.cat([h.new_zeros((1,h.shape[1])),h.cumsum(0)],dim=0)
   for start in range(0,len(bounds),256):
    part=bounds[start:start+256];scores=self.span_logits(h,part,prefix).sigmoid().cpu()
    for j,bound in enumerate(part):
     a,b=bound
     for k in sorted(candidates[bound]):
      span_cut=structure_span_threshold if k>=33 and structure_span_threshold is not None else span_threshold
      if k<12 and attribute_span_threshold is not None:span_cut=attribute_span_threshold
      if scores[j,k]>=span_cut:
       score=float((ps[a,k]*pe[b-1,k]*scores[j,k]).pow(1/3));found.append((score,k,bound))
   if policy:
    from .decoding import select_candidates
    found=select_candidates(found,self.e_labels,policy)
   found=sorted(found,key=lambda x:(-x[0],x[2],x[1]));entity_truncated=max(0,len(found)-max_entities);found=found[:max_entities]
   found.sort(key=lambda x:(x[2][0],x[2][1],x[1]));counter=Counter();entities=[];eb=[];types=[]
   for score,k,(a,b) in found:
    label=self.e_labels[k];counter[label]+=1;num=k+1;eid=f'E{num}_{counter[label]}'
    entities.append({'id':eid,'code':label_code(label),'label':label,'scope':'metadata' if num<=12 else ('diagram' if num<=33 else 'structure'),'start':offsets[a][0],'end':offsets[b-1][1],'text':text[offsets[a][0]:offsets[b-1][1]],'confidence':score,'status':'auto'})
    eb.append((a,b));types.append(k)
  else:
   aligned,missing=self.align(oracle_entities,offsets)
   if missing:raise ValueError('Oracle entity alignment failed')
   entities=[dict(e) for e,b in aligned];eb=[b for e,b in aligned];types=[self.e_index[e['label']] for e,b in aligned];truncated=entity_truncated=0
  return {'full_text':text,'entities':entities,'relations':[],'model_warnings':warnings,'decoder_diagnostics':{'boundary_candidates_truncated':truncated,'entities_truncated':entity_truncated,'span_width_cap_characters':self.config.get('max_span_chars',4096)},'confidence_note':'Uncalibrated entity scores. No relation head or relation-based filtering.'}
