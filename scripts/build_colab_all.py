"""One concise notebook, with private source/data embedded and no old weights."""
import argparse,base64,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'colab'
def main(output_dir=None):
    global OUT
    if output_dir is not None:OUT=Path(output_dir).resolve()
    OUT.mkdir(parents=True,exist_ok=True);archive=OUT/'Inputs_All369.zip'
    if any((OUT/n).exists() for n in ['Inputs_All369.zip','Train_PhoBERT_NER_All369.ipynb','INPUT_MANIFEST.json']):raise FileExistsError('Preserve frozen inputs; select another --output-dir for a new revision')
    files=[]
    for name in ['src','scripts','configs','data/dataset','resources/phobert-base']:
        files.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.cache']) and p.suffix not in ['.bin','.safetensors'])
    files.extend(ROOT/n for n in ['README.md','requirements.txt','app.py'])
    import subprocess,sys
    check=subprocess.run([sys.executable,'scripts/check_all_data.py'],cwd=ROOT,capture_output=True,text=True,check=True)
    preflight=json.loads(check.stdout.splitlines()[-1]);assert preflight['source_hashes_verified']
    digest_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    bundle={'mode':'normalized_training_inputs_originals_kept_locally','dataset_files':{n:digest_file(ROOT/'data/dataset'/n) for n in ['train.json','val.json']},'local_source_preflight':preflight}
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,ROOT.name+'/'+str(p.relative_to(ROOT)))
        z.writestr(ROOT.name+'/reports/COLAB_INPUT_MANIFEST.json',json.dumps(bundle,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    raw=archive.read_bytes();digest=hashlib.sha256(raw).hexdigest();encoded=base64.b64encode(raw).decode();payload='\n'.join(repr(encoded[i:i+1000]) for i in range(0,len(encoded),1000))
    md=lambda s:{'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)}
    code=lambda s:{'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'outputs':[],'execution_count':None}
    cells=[md('''# PhoBERT + Span Head — 369 văn bản / chỉ thực thể
**314 Train / 55 Validation / chưa có Test độc lập.** Chia theo nhóm văn bản trước khi train; cửa sổ của cùng văn bản luôn ở cùng tập.

Chọn **Runtime → Change runtime type → T4 GPU → Save**, sau đó **Runtime → Run all**. Không cần upload thêm dữ liệu hoặc đăng nhập Hugging Face. Notebook chứa dữ liệu chuẩn hóa; các DOCX/JSON/CSV gốc vẫn lưu trên máy. Hash nguồn đã được kiểm tra trước khi đóng gói; Colab kiểm tra lại hash dữ liệu Train/Validation, offset và split, không giả định đã đọc lại nguồn thô trên Colab. Sau khi huấn luyện, notebook tự in Precision/Recall/Micro-F1/Macro-F1 và tải ZIP chứa checkpoint/kết quả. Giữ runtime kết nối đến khi có trạng thái COMPLETE.

Khởi tạo mới từ vinai/phobert-base chính thức đã ghim revision/SHA, không dùng checkpoint cũ. 40 nhãn thực thể; không huấn luyện quan hệ. Huấn luyện tối đa 12 epoch, dừng sớm theo Validation; không hứa trước mức F1. Cửa sổ Train khoảng 8.192 ký tự có overlap, thêm cửa sổ cho nhãn dài, che nhãn bị cắt ở biên; đánh giá trên toàn bộ văn bản với offset gốc.

Nguồn: 385 bản xuất JSON, bỏ 14 bản trùng và 2 trường hợp chưa an toàn để chuẩn hóa → 369 văn bản. Hiện có đủ nguồn Word cho369văn bản:367DOCXgốc và2DOCXchuyển từDOCgốc. Khi chuẩn bị snapshot,152bản dùngWordbody từJSON; đã đối chiếu nguồn bổ sung,146khớp nguyênchuỗi và6chỉkhácwhitespace. Dataset hiện gồm train.json/val.json chuẩn, giữ nguyên mọi bản ghi của snapshot. Phần thuộc tính/lược đồ bổ sung ngoài Word bị loại khỏi đầu vào và mục tiêu. Nhãn có sẵn do người dùng kiểm tra, chưa phải một cuộc đánh giá Gold độc lập. Những nhãn không có mẫu dương không được tuyên bố đạt điểm cao.
'''),code('''import os,sys,subprocess,torch
os.environ['USE_TF']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
assert torch.cuda.is_available(), 'Chưa được cấp GPU; không tự train CPU.'
print('ALLOCATED_GPU',torch.cuda.get_device_name(0),torch.__version__)
subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.51.3','tokenizers>=0.21,<0.22','pyvi==0.1.1','huggingface_hub'],check=True)
'''),code('''#@title Khôi phục mã nguồn và dữ liệu đã kiểm tra
from pathlib import Path
import base64,hashlib,io,json,zipfile
workspace=Path('/content/legal_ner_all369');workspace.mkdir(parents=True,exist_ok=True)
raw=base64.b64decode(\n'''+payload+'''\n)
assert hashlib.sha256(raw).hexdigest()=='''+repr(digest)+'''
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 assert z.testzip() is None
 assert all((workspace/n).resolve().is_relative_to(workspace.resolve()) for n in z.namelist())
 z.extractall(workspace)
root=workspace/'Legal_Entity_NER'
print('FROZEN_INPUTS_RESTORED',root)
'''),code('''# Chạy worker; kết nối trình duyệt mất không tự làm mất log/checkpoint.
run='phobert_span_all369_v1';report=root/'reports'/run;report.mkdir(parents=True,exist_ok=True)
assert not (report/'worker_status.json').exists(), 'Đã có worker, đọc trạng thái thay vì chạy trùng.'
with (report/'worker.log').open('w') as log:
 process=subprocess.Popen([sys.executable,'-u','scripts/run_colab_all.py'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('WORKER_STARTED',process.pid)
from google.colab import output
from IPython.display import JSON,Javascript,display
allowed={'trained_zip':workspace/'Legal_Entity_NER_All369_Trained.zip','results':report/'RESULTS.json','history':report/'history.json'}
def transfer(key,offset=0,limit=4*1024*1024):
 path=allowed[key];offset=int(offset);limit=min(int(limit),4*1024*1024)
 assert offset>=0 and limit>0
 with path.open('rb') as stream:stream.seek(offset);block=stream.read(limit)
 return JSON({'offset':offset,'total':path.stat().st_size,'bytes':len(block),'base64':base64.b64encode(block).decode()})
def poll():
 p=report/'worker_status.json';s=json.loads(p.read_text()) if p.exists() else {'phase':'STARTING'}
 for name in ['progress.json','history.json']:
  try:
   if (report/name).exists():s[name]=json.loads((report/name).read_text())
  except ValueError:pass
 p=report/'train.log' if (report/'train.log').exists() else report/'worker.log'
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4000));s['log_tail']=f.read().decode(errors='replace')
 return JSON(s)
output.register_callback('ner_all369.status',poll);output.register_callback('ner_all369.transfer',transfer)
display(Javascript("window.nckhPoll=()=>google.colab.kernel.invokeFunction('ner_all369.status',[],{});window.nckhTransfer=(key,offset,limit)=>google.colab.kernel.invokeFunction('ner_all369.transfer',[key,offset,limit],{});document.body.appendChild(Object.assign(document.createElement('pre'),{textContent:'NER_ALL369_WORKER_MONITOR_READY'}));"))
'''),code('''import time
from IPython.display import clear_output
while True:
 state=poll().data
 clear_output(wait=True)
 print(json.dumps(state,ensure_ascii=False,indent=2))
 if state.get('phase')=='FAILED':raise RuntimeError(state.get('error','Worker failed'))
 if state.get('phase')=='COMPLETE':break
 time.sleep(30)
'''),code('''state=json.loads((report/'worker_status.json').read_text());assert state['phase']=='COMPLETE',state
from google.colab import files
files.download(state['artifact']['path'])
''')]
    for i,c in enumerate(cells):
        c['id']='ner_all369_'+str(i)
        if c['cell_type']=='code':compile(''.join(c['source']),'<ner_all369_'+str(i)+'>','exec')
    cells[2]['metadata']['cellView']='form'
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':cells}
    path=OUT/'Train_PhoBERT_NER_All369.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=1))
    record={'status':'READY_NOT_TRAINED','input_bytes':len(raw),'input_sha256':digest,'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_documents':369,'train_documents':314,'validation_documents':55,'official_pretrained_only':True}
    (OUT/'INPUT_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=None);args=parser.parse_args();main(args.output_dir)
