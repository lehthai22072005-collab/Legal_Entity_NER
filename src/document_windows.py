"""Bounded Train views; document-level split and source offsets stay immutable."""
import copy


def training_views(documents, window_chars=8192, overlap_chars=2048, context_chars=256):
    if not 0 <= overlap_chars < window_chars:
        raise ValueError('Overlap must be smaller than window size')
    result = []
    for doc in documents:
        text = doc['full_text']; n = len(text)
        from .model import line_quote_context
        context = line_quote_context(text)
        positions=[]; pos=0
        for line in text.splitlines(keepends=True):
            body=line.rstrip('\r\n')
            if body and not body[0].isspace(): positions.append(pos)
            pos+=len(line)
        context_by_start=dict(zip(positions,context))
        ranges = set()
        def line_range(a,b):
            left=text.rfind('\n',max(0,a-256),a)
            right=text.find('\n',b,min(n,b+256))
            return (left+1 if left>=0 else a, right+1 if right>=0 else b)
        for start in range(0, n, window_chars-overlap_chars):
            ranges.add(line_range(start, min(n, start+window_chars)))
            if start+window_chars >= n:
                break
        # Extra views preserve every positive, including nested or long spans.
        for e in doc['entities']:
            if not any(a <= e['start'] < e['end'] <= b for a,b in ranges):
                width = max(window_chars, e['end']-e['start']+2*context_chars)
                a = max(0, min(n-width, e['start']-context_chars))
                ranges.add(line_range(a, min(n, a+width)))
        covered = set()
        for a,b in sorted(ranges):
            view = {k:v for k,v in doc.items() if k not in ['full_text','entities','relations']}
            view.update(full_text=text[a:b], entities=[], relations=[], docx_length=b-a,
                        source_document_id=doc['document_id'], source_start=a, source_end=b,
                        source_length=n, ignored_partial_entities=[])
            rows=[];pos=a
            for line in view['full_text'].splitlines(keepends=True):
                body=line.rstrip('\r\n')
                if body and not body[0].isspace(): rows.append(context_by_start.get(pos,[0.,0.,0.,pos/max(n,1)]))
                pos+=len(line)
            view['source_marker_context']=rows
            for i,e in enumerate(doc['entities']):
                if a <= e['start'] < e['end'] <= b:
                    entity = copy.copy(e)
                    entity.update(start=e['start']-a, end=e['end']-a)
                    assert view['full_text'][entity['start']:entity['end']] == entity['text']
                    view['entities'].append(entity); covered.add(i)
                elif e['start'] < b and e['end'] > a:
                    view['ignored_partial_entities'].append({
                        'label':e['label'], 'start':max(a,e['start'])-a,
                        'end':min(b,e['end'])-a})
            result.append(view)
        assert len(covered) == len(doc['entities']), ('Uncovered positive',doc['document_id'])
    return result


def candidate_pairs(starts, ends, offsets, width_limit, newline_prefix=None):
    """Same valid pairs as Cartesian enumeration, without scanning impossible ends."""
    import bisect
    ends = sorted(ends)
    end_chars = [offsets[b][1] for b in ends]
    for a in starts:
        limit = offsets[a][0]+width_limit
        lo = bisect.bisect_left(ends,a)
        hi = bisect.bisect_right(end_chars,limit)
        for b in ends[lo:hi]:
            if newline_prefix is not None and newline_prefix[offsets[b][1]] != newline_prefix[offsets[a][0]]:
                continue
            yield a,b+1
