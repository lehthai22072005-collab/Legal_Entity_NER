"""Frozen parser functions from the teacher annotation tool; no server/network code.
Upstream JonnyDangr/tool_labeling commit 7665edb141c7dcd06e94a7c1d5e2b6c038012067.
"""
import io,re,zipfile,unicodedata
import xml.etree.ElementTree as ET
W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

def clean_glued_footer_text(txt):
    """Fix glued runs in legacy legal .docx files without losing any characters."""
    txt = re.sub(r'(\./\.)\s*(KT\.\s*BỘ TRƯỞNG|BỘ TRƯỞNG|THỨ TRƯỞNG|CHỦ TỊCH)', r'\1\n\2', txt)
    txt = re.sub(r'(KT\.\s*BỘ TRƯỞNG)\s+(THỨ TRƯỞNG)', r'\1\n\2', txt)
    txt = re.sub(r'(THỨ TRƯỞNG|BỘ TRƯỞNG|CHỦ TỊCH)\s*(\(Đã ký\))', r'\1\n\2', txt)
    txt = re.sub(r'(\(Đã ký\))\s*([^\n\s])', r'\1\n\2', txt)
    return txt

def parse_paragraph_xml(p_elem):
    """
    Extract 100% of text and formatting (align, bold, italic, style, indent) from a <w:p> element.
    Captures all <w:t>, <w:br/>, <w:cr/>, <w:tab/> inside runs, hyperlinks, smart tags, etc.
    """
    pPr = p_elem.find(f'{W_NS}pPr')
    jc = None
    p_style = None
    indent_left = 0
    if pPr is not None:
        jc_el = pPr.find(f'{W_NS}jc')
        if jc_el is not None:
            jc = jc_el.attrib.get(f'{W_NS}val')
        st_el = pPr.find(f'{W_NS}pStyle')
        if st_el is not None:
            p_style = st_el.attrib.get(f'{W_NS}val')
        ind_el = pPr.find(f'{W_NS}ind')
        if ind_el is not None:
            try:
                indent_left = int(ind_el.attrib.get(f'{W_NS}left', '0'))
            except ValueError:
                indent_left = 0

    parts = []
    bold_chars = 0
    italic_chars = 0
    total_chars = 0

    for elem in p_elem.iter():
        tag = elem.tag.replace(W_NS, '')
        if tag == 'r':
            rPr = elem.find(f'{W_NS}rPr')
            is_b = rPr is not None and (
                rPr.find(f'{W_NS}b') is not None and
                rPr.find(f'{W_NS}b').attrib.get(f'{W_NS}val', 'true') not in ('0', 'false')
            )
            is_i = rPr is not None and (
                rPr.find(f'{W_NS}i') is not None and
                rPr.find(f'{W_NS}i').attrib.get(f'{W_NS}val', 'true') not in ('0', 'false')
            )
            for child in elem:
                ctag = child.tag.replace(W_NS, '')
                if ctag == 't' and child.text:
                    parts.append(child.text)
                    c_len = len(child.text.strip())
                    total_chars += c_len
                    if is_b:
                        bold_chars += c_len
                    if is_i:
                        italic_chars += c_len
                elif ctag in ('br', 'cr'):
                    parts.append('\n')
                elif ctag == 'tab':
                    parts.append('\t')

    raw_text = ''.join(parts)
    cleaned_text = unicodedata.normalize('NFC', clean_glued_footer_text(raw_text).strip('\r'))
    if not cleaned_text.strip():
        return []

    is_bold = (bold_chars >= total_chars * 0.55) if total_chars > 0 else False
    is_italic = (italic_chars >= total_chars * 0.55) if total_chars > 0 else False

    align = 'left'
    if jc in ('center',):
        align = 'center'
    elif jc in ('right', 'end'):
        align = 'right'
    elif jc in ('both', 'distribute', 'justify'):
        align = 'justify'

    # Split if internal line breaks exist so each line gets proper alignment & offset
    sub_lines = [ln.strip() for ln in cleaned_text.split('\n') if ln.strip()]
    result_paras = []
    for ln in sub_lines:
        result_paras.append({
            'text': ln,
            'align': align,
            'bold': is_bold,
            'italic': is_italic,
            'style': p_style,
            'indent': indent_left
        })
    return result_paras

def extract_docx_structured(file_bytes):
    """
    Extract 100% of text and structural layout (paragraphs, alignments, bold/italic, tables, columns)
    from .docx bytes without losing a single character.
    Returns (full_text, blocks) where every paragraph in blocks has exact [start, end] offsets in full_text.
    """
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        xml_bytes = z.read('word/document.xml')
    root = ET.fromstring(xml_bytes)
    body = root.find(f'{W_NS}body')
    if body is None:
        return '', []

    raw_blocks = []
    for child in body:
        tag = child.tag.replace(W_NS, '')
        if tag == 'p':
            paras = parse_paragraph_xml(child)
            for p in paras:
                raw_blocks.append({'type': 'p', **p})
        elif tag == 'tbl':
            tblPr = child.find(f'{W_NS}tblPr')
            has_borders = False
            if tblPr is not None and tblPr.find(f'{W_NS}tblBorders') is not None:
                has_borders = True

            rows_data = []
            for tr in child.findall(f'{W_NS}tr'):
                row_cells = []
                for tc in tr.findall(f'{W_NS}tc'):
                    colspan = 1
                    tcPr = tc.find(f'{W_NS}tcPr')
                    if tcPr is not None:
                        gs = tcPr.find(f'{W_NS}gridSpan')
                        if gs is not None:
                            try:
                                colspan = int(gs.attrib.get(f'{W_NS}val', '1'))
                            except ValueError:
                                colspan = 1
                    cell_paras = []
                    for p_el in tc.findall(f'.//{W_NS}p'):
                        cell_paras.extend(parse_paragraph_xml(p_el))
                    row_cells.append({
                        'colspan': colspan,
                        'paras': cell_paras
                    })
                if any(len(c['paras']) > 0 for c in row_cells):
                    rows_data.append(row_cells)

            if len(rows_data) >= 3 or (len(rows_data) >= 1 and len(rows_data[0]) >= 3):
                has_borders = True

            if rows_data:
                raw_blocks.append({
                    'type': 'tbl',
                    'bordered': has_borders,
                    'rows': rows_data
                })

    # Apply intelligent Vietnamese Legal Document alignment heuristics where Word left jc=None
    in_title_zone = False
    for idx, blk in enumerate(raw_blocks):
        if blk['type'] == 'p':
            txt = blk['text'].strip()
            upper_txt = txt.upper()

            if re.match(r'^(QUYẾT ĐỊNH|QUYẾT ĐỊNH|NGHỊ ĐỊNH|THÔNG TƯ|CHỈ THỊ|NGHỊ QUYẾT|BỘ LUẬT|LUẬT|PHÁP LỆNH)$', upper_txt):
                blk['align'] = 'center'
                blk['bold'] = True
                in_title_zone = True
                continue

            if in_title_zone:
                if re.match(r'^[_—\-]{4,}$', txt):
                    blk['align'] = 'center'
                    in_title_zone = False
                    continue
                if txt == upper_txt and re.match(r'^(BỘ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|CHÍNH PHỦ|ỦY BAN|UỶ BAN|QUỐC HỘI)', upper_txt):
                    blk['align'] = 'center'
                    blk['bold'] = True
                    in_title_zone = False
                    continue
                if re.match(r'^(Căn\s+cứ|Theo\s+đề\s+nghị)', txt):
                    in_title_zone = False
                elif blk['bold'] or len(txt) < 120:
                    blk['align'] = 'center'
                    blk['bold'] = True
                    continue
                else:
                    in_title_zone = False

            if re.match(r'^(Điều\s+\d+|Khoản\s+\d+|\d+\.\s|[a-eđgh]\)\s|Căn\s+cứ\s|Theo\s+đề\s+nghị)', txt):
                blk['align'] = 'justify' if len(txt) > 85 else 'left'
            elif re.match(r'^[_—\-]{4,}$', txt):
                blk['align'] = 'center'
            elif txt == upper_txt and re.match(r'^(BỘ TRƯỞNG|THỦ TƯỚNG CHÍNH PHỦ|CHỦ TỊCH|HỘI ĐỒNG)', upper_txt) and len(txt) < 70:
                blk['align'] = 'center'
                blk['bold'] = True
            elif upper_txt in ('QUYẾT ĐỊNH:', 'NGHỊ ĐỊNH:', 'QUYẾT ĐỊNH:'):
                blk['align'] = 'center'
                blk['bold'] = True
            elif re.match(r'^(KT\.\s*BỘ TRƯỞNG|KT\.\s*THỦ TƯỚNG|KT\.\s*CHỦ TỊCH|THỨ TRƯỞNG|BỘ TRƯỞNG|CHỦ TỊCH|\(ĐÃ KÝ\))$', upper_txt):
                blk['align'] = 'right'
                blk['bold'] = True
            elif idx > 0 and raw_blocks[idx - 1].get('type') == 'p' and raw_blocks[idx - 1].get('text', '').strip() == '(Đã ký)':
                blk['align'] = 'right'
                blk['bold'] = True
            elif blk['align'] == 'left' and len(txt) > 95 and blk.get('style') != 'Heading1':
                blk['align'] = 'justify'

        elif blk['type'] == 'tbl' and not blk['bordered']:
            # Format 2-column header / footer tables cleanly like official legal documents
            for r in blk['rows']:
                if len(r) == 2:
                    for p in r[0]['paras']:
                        if not p['text'].startswith('-') and not p['text'].startswith('Nơi nhận'):
                            p['align'] = 'center'
                    for p in r[1]['paras']:
                        p['align'] = 'center'

    # Assign exact character offsets [start, end] in full_text for every paragraph
    all_lines = []
    current_offset = 0

    for blk in raw_blocks:
        if blk['type'] == 'p':
            txt = blk['text']
            blk['start'] = current_offset
            blk['end'] = current_offset + len(txt)
            all_lines.append(txt)
            current_offset = blk['end'] + 1
        elif blk['type'] == 'tbl':
            for r in blk['rows']:
                for cell in r:
                    for p in cell['paras']:
                        txt = p['text']
                        p['start'] = current_offset
                        p['end'] = current_offset + len(txt)
                        all_lines.append(txt)
                        current_offset = p['end'] + 1

    full_text = '\n'.join(all_lines)
    return full_text, raw_blocks
