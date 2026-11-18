"""Exact-span / directed end-to-end relation metrics, including unsupported labels."""
from collections import Counter

def entities(doc):return {(e['label'],e['start'],e['end']) for e in doc['entities']}
def relations(doc):
 ids={e['id']:(e['label'],e['start'],e['end']) for e in doc['entities']}
 return {(ids[r['from_id']],r['label'],ids[r['to_id']]) for r in doc['relations']}

def summarize(gold,pred,labels,kind='entity'):
 counts={label:Counter(tp=0,fp=0,fn=0,support=0) for label in labels}
 for g,p in zip(gold,pred):
  gs=entities(g) if kind=='entity' else relations(g);ps=entities(p) if kind=='entity' else relations(p)
  label_of=(lambda item:item[0]) if kind=='entity' else (lambda item:item[1])
  for x in gs:counts[label_of(x)]['support']+=1
  for x in gs&ps:counts[label_of(x)]['tp']+=1
  for x in ps-gs:counts[label_of(x)]['fp']+=1
  for x in gs-ps:counts[label_of(x)]['fn']+=1
 def rate(c):
  tp,fp,fn=c['tp'],c['fp'],c['fn'];p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
  return {**dict(c),'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0.}
 per={k:rate(v) for k,v in counts.items()};total=Counter()
 for value in counts.values():total.update(value)
 supported=[v['f1'] for v in per.values() if v['support']>0]
 return {'micro':rate(total),'macro_f1_supported_labels':sum(supported)/len(supported) if supported else None,'per_label':{k:{**v,'f1':v['f1'] if v['support'] else None} for k,v in per.items()},'unsupported_gold_labels':[k for k,v in per.items() if not v['support']]}

def evaluate(gold,pred,ontology):
 assert len(gold)==len(pred)
 return {'documents':len(gold),'entity_exact_span':summarize(gold,pred,[x['label'] for x in ontology['entities']]),'relation_end_to_end':summarize(gold,pred,[x['label'] for x in ontology['relations']],'relation'),'metric_policy':'Exact entity type+span. Relations require exact source and target type/spans, directed label. Null is not an entity TP.'}
