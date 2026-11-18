"""Optional decoding constraints estimated exclusively from the Train split.

These filters never create entities, change source text or add relations. They
only remove contradictory neural candidates. Keep the unconstrained run as an
ablation and measure the impact on Validation before selecting this policy.
"""
from bisect import bisect_right
from collections import defaultdict, Counter
import math


def fit_policy(documents, ontology, structured=False, strict_boundaries=False):
    labels = [e['label'] for e in ontology['entities']]
    lengths = defaultdict(list)
    overlaps = defaultdict(bool)
    triples = set()
    line_start = defaultdict(list)
    outgoing = Counter()
    support = Counter()
    preceding = defaultdict(list)
    line_ends=defaultdict(list);single_lines=defaultdict(list);word_ends=defaultdict(list)
    for doc in documents:
        by_label = defaultdict(list)
        by_id = {e['id']: e for e in doc['entities']}
        for e in doc['entities']:
            if strict_boundaries:
                text=doc['full_text'];a,b=e['start'],e['end']
                line_ends[e['label']].append(b==len(text) or text[b]=='\n')
                single_lines[e['label']].append('\n' not in text[a:b])
                word_ends[e['label']].append(b==len(text) or not (text[b-1].isalnum() and text[b].isalnum()))
            lengths[e['label']].append(e['end'] - e['start'])
            by_label[e['label']].append((e['start'], e['end']))
            if structured:line_start[e['label']].append(e['start']==0 or doc['full_text'][e['start']-1]=='\n')
        for label, spans in by_label.items():
            spans.sort()
            overlaps[label] |= any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
        counts=Counter()
        for r in doc['relations']:
            source,target=by_id[r['from_id']],by_id[r['to_id']]
            triple=(source['label'],target['label'],r['label']);triples.add(triple)
            counts[(source['id'],r['label'])]+=1
            support[(source['label'],r['label'])]+=1
            preceding[triple].append(target['start']<source['start'])
        for (source_id,rel),count in counts.items():
            key=(by_id[source_id]['label'],rel);outgoing[key]=max(outgoing[key],count)
    policy = {'fitted_on': 'Train'+str(len(documents))+' annotated full input only',
            'label_max_chars': {k: max(8, 2*max(v)) for k, v in lengths.items()},
            'nonoverlap_labels': [k for k in labels if lengths[k] and not overlaps[k]],
            # Metadata slots in the schema are document-level, single valued.
            'singleton_labels': labels[:12],
            'allowed_relation_triples': sorted(triples),
            'creates_predictions': False}
    if structured:
        structural=set(labels[33:])
        policy.update(line_start_labels=[k for k in labels if k in structural and line_start[k] and all(line_start[k])],
                      functional_source_relation=[list(k) for k,v in sorted(outgoing.items()) if v==1 and support[k]>=3],
                      preceding_parent_triples=[list(k) for k,v in sorted(preceding.items()) if k[0] in structural and len(v)>=3 and all(v)],
                      interval_objective='log_odds')
    if strict_boundaries:
        structural=set(labels[33:])
        policy.update(line_end_labels=[k for k,v in line_ends.items() if k in structural and len(v)>=3 and all(v)],
                      single_line_labels=[k for k,v in single_lines.items() if k in structural and len(v)>=3 and all(v)],
                      word_end_labels=[k for k,v in word_ends.items() if len(v)>=5 and all(v)],
                      boundary_policy='Train evidence only; reject candidates, never snap or create spans')
    return policy


def candidate_boundary_masks(text,offsets,label,policy):
    """Apply learned exclusions before top-k, never move an offset."""
    policy=policy or {}
    starts=[not text[a].isspace() for a,b in offsets]
    ends=[not text[b-1].isspace() for a,b in offsets]
    if label in policy.get('line_start_labels',[]):
        starts=[ok and (a==0 or text[a-1]=='\n') for ok,(a,b) in zip(starts,offsets)]
    if label in policy.get('line_end_labels',[]):
        ends=[ok and (b==len(text) or text[b]=='\n') for ok,(a,b) in zip(ends,offsets)]
    if label in policy.get('word_end_labels',[]):
        ends=[ok and (b==len(text) or not (text[b-1].isalnum() and text[b].isalnum())) for ok,(a,b) in zip(ends,offsets)]
    return starts,ends


def select_candidates(found, labels, policy):
    """Maximum-confidence singleton/non-overlapping span selection by type.

    Different types remain free to nest and to share exactly the same span.
    Confidence must exceed the neural thresholds before a candidate enters here.
    """
    groups = defaultdict(list)
    for candidate in found:
        groups[candidate[1]].append(candidate)
    selected = []
    for k, group in groups.items():
        label = labels[k]
        if label in policy['singleton_labels']:
            selected.append(max(group, key=lambda x: (x[0], -x[2][0], -x[2][1])))
        elif label in policy['nonoverlap_labels']:
            group = sorted(group, key=lambda x: (x[2][1], x[2][0], -x[0]))
            ends = [x[2][1] for x in group]
            score = [0.]
            back = []
            for i, (confidence, _, (start, end)) in enumerate(group):
                previous = bisect_right(ends, start, hi=i)
                weight=math.log(max(confidence,1e-6)/max(1-confidence,1e-6)) if policy.get('interval_objective')=='log_odds' else confidence
                take = score[previous] + max(0.,weight)
                if take > score[-1]:
                    score.append(take)
                    back.append((True, previous))
                else:
                    score.append(score[-1])
                    back.append((False, i))
            i = len(group)
            while i:
                take, previous = back[i-1]
                if take:
                    selected.append(group[i-1])
                i = previous
        else:
            selected.extend(group)
    return selected


def select_relations(relations, entities, policy, null_scores=None):
    """Remove wrong-direction/multiple-parent candidates, never add an edge."""
    ids={e['id']:e for e in entities};functional={tuple(x) for x in policy.get('functional_source_relation',[])}
    preceding={tuple(x) for x in policy.get('preceding_parent_triples',[])}
    result=[];best={}
    for r in relations:
        s,t=ids[r['from_id']],ids[r['to_id']];label=r['label']
        if (s['label'],t['label'],label) in preceding and t['start']>=s['start']:continue
        if (s['label'],label) in functional:
            if null_scores and r['confidence']<=null_scores[label]:continue
            key=(s['id'],label)
            if key not in best or r['confidence']>best[key]['confidence']:best[key]=r
        else:result.append(r)
    return result+list(best.values())
