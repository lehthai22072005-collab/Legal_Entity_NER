from pathlib import Path
from .docx_parser import parse_docx
from .training import load_checkpoint
from .dataset import validate

FIELDS={'Tên văn bản':'ten_van_ban','Số hiệu':'so_hieu','Loại văn bản':'loai_van_ban','Ngành':'nganh','Ngày ban hành':'ngay_ban_hanh','Lĩnh vực':'linh_vuc','Ngày có hiệu lực':'ngay_co_hieu_luc','Ngày hết hiệu lực':'ngay_het_hieu_luc','Tình trạng hiệu lực':'tinh_trang_hieu_luc','Cơ quan ban hành':'co_quan_ban_hanh','Chức danh':'chuc_danh','Người ký':'nguoi_ky'}
def predict(path,checkpoint,context_text=None,device='cpu'):
 model,extra=load_checkpoint(checkpoint,device)
 if model.config.get('entity_only') and context_text:raise ValueError('Entity-only extraction accepts DOCX content only')
 parsed=parse_docx(path);text=parsed['canonical_text'];docx_length=len(text)
 if context_text:text+='\n'+context_text
 result=model.infer(text,max_boundaries=extra['decoder_topk'],**extra['thresholds'])
 metadata={key:None for key in FIELDS.values()}
 for label,key in FIELDS.items():
  items=[e for e in result['entities'] if e['label']==label]
  if items:metadata[key]=max(items,key=lambda e:e['confidence'])['text']
 for e in result['entities']:e['origin']='docx' if e['end']<=docx_length else 'provided_supplemental_context'
 result.update(schema_version='new_joint_1.0',file_name=Path(path).name,text_encoding='utf-8',offset_convention='Python Unicode [start,end)',document_metadata=metadata,docx_length=docx_length,source_issues=parsed['source_issues'],docx_sha256=parsed['docx_sha256'],model={'encoder':model.config['encoder'],'checkpoint':str(checkpoint),'decoder':'Train-fitted span selection','legacy_checkpoint_used':False,'new_reviewed30_only':bool(model.config.get('new_reviewed30_only',False)),'dataset_dir':model.config.get('dataset_dir','data/processed'),'dataset_manifest_sha256':model.config.get('dataset_manifest_sha256'),'dataset_sha256':model.config.get('dataset_sha256')},external_metadata_retrieved=False)
 validate(result)
 if model.config.get('entity_only'):
  result['schema_version']='entity_only_1.0'
  result['model']['task']='entity_extraction';result['model']['relation_head_present']=False
  result.pop('relations',None)
 return result
