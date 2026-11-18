"""Canonicalization compatible with the teacher's reviewed annotation tool."""
import hashlib
from pathlib import Path
from zipfile import ZipFile
from .teacher_tool_parser import extract_docx_structured

def parse_docx(path):
 path=Path(path)
 if path.suffix.lower()!='.docx':raise ValueError('Đầu vào phải là DOCX')
 raw=path.read_bytes();text,blocks=extract_docx_structured(raw);issues=[]
 with ZipFile(path) as z:
  for name in z.namelist():
   if name.startswith(('word/header','word/footer','word/footnote','word/endnote','word/embeddings/')):
    issues.append({'code':'PART_OUTSIDE_REVIEWED_BODY_CANONICAL','part':name,'message':'Reviewed tool annotations cover body flow. This part is retained in source DOCX, not silently treated as annotated body.'})
 return {'canonical_text':text,'blocks':blocks,'source_issues':issues,'docx_sha256':hashlib.sha256(raw).hexdigest(),'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'offset_convention':'Python Unicode [start,end)','canonicalization':'teacher-tool body parser, version frozen with this project'}
