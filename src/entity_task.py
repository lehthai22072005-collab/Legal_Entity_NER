"""Entity-only metrics and the Train-fitted entity decoder policy."""
from .metrics import summarize

def entity_policy(policy):
    p=dict(policy)
    for k in ['allowed_relation_triples','functional_source_relation','preceding_parent_triples']:p.pop(k,None)
    return p
def evaluate_entities(gold,pred,ontology):
    assert len(gold)==len(pred)
    return {'documents':len(gold),'entity_exact_span':summarize(gold,pred,[e['label'] for e in ontology['entities']]),
            'metric_policy':'Exact type and Python Unicode [start,end) span. Null is not a true-positive entity. Labels without positive support have F1 null.'}
