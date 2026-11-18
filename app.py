"""Upload DOCX, extract evidence-backed entity spans and download JSON."""
import json,tempfile
from pathlib import Path
import streamlit as st
from src.pipeline import predict
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Nhận dạng thực thể văn bản pháp luật',layout='wide')
st.title('Nhận dạng thực thể văn bản pháp luật')
st.caption('Upload DOCX để trích xuất đoạn chữ và nhãn. Thông tin không tìm thấy được để null; kết quả cần được kiểm tra.')
models=[p for p in (ROOT/'checkpoints').glob('*') if (p/'model.pt').exists()]
if not models:st.info('Chưa có checkpoint của lượt huấn luyện mới.');st.stop()
checkpoint=st.selectbox('Mô hình',sorted(models),format_func=lambda p:p.name)
file=st.file_uploader('Văn bản Word',type=['docx'])
if file and st.button('Trích xuất',type='primary'):
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/Path(file.name).name;p.write_bytes(file.getvalue())
        with st.spinner('Đang trích xuất…'):result=predict(p,checkpoint)
    st.subheader('Thuộc tính');st.json(result['document_metadata'])
    st.subheader('Thực thể và đoạn chữ chứng minh');st.dataframe(result['entities'])
    if result['source_issues']:st.warning('Có thành phần nguồn cần kiểm tra.');st.json(result['source_issues'])
    st.download_button('Tải JSON',json.dumps(result,ensure_ascii=False,indent=2),file_name=Path(file.name).stem+'_entities.json',mime='application/json')
