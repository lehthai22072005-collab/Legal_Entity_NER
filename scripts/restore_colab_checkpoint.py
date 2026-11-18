"""Upload the verified epoch-11 backup, then launch evaluation without training."""
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

from marionette_driver.marionette import Marionette
from colab_artifacts_all import ROOT


def invoke(m, expression, *args):
    result = m.execute_async_script(
        'const done=arguments[arguments.length-1];' + expression +
        '.then(done,e=>done({error:String(e)}));', list(args), sandbox=None)
    if not isinstance(result, dict) or 'error' in result:
        raise RuntimeError(str(result))
    data = result['data']['application/json']
    return json.loads(data) if isinstance(data, str) else data


def main():
    source = ROOT / 'colab/NER_All369_Checkpoint_Backup.zip'
    with source.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == (
            'e3b2e57899700e34e60343c1a6d6a81100864ceeb2de6c00ca2f302beb893828')
    m = Marionette(port=2828)
    m.start_session(timeout=30)
    m.timeout.script = 60
    receipt = ROOT / 'colab/received/checkpoint_upload.json'
    try:
        for handle in m.window_handles:
            m.switch_to_window(handle)
            if '/drive/1SmVaZgWPyfwJh44RFiX96ZkwCXipOgmA' in m.get_url():
                break
        else:
            raise RuntimeError('Open the authorized finalization notebook')
        for frame in m.find_elements('css selector', '#cell-ner_all369_2 iframe[title="Cell output"]'):
            m.switch_to_frame(frame)
            if m.execute_script("return typeof window.nerReceive==='function'", sandbox=None):
                break
            m.switch_to_frame()
        else:
            raise RuntimeError('Execute checkpoint receiver cell first')
        state = invoke(m, 'window.nerUploadStatus()')
        offset = state['bytes']
        assert 0 <= offset <= source.stat().st_size
        with source.open('rb') as stream:
            stream.seek(offset)
            while block := stream.read(4 * 1024 * 1024):
                ack = invoke(m, 'window.nerReceive(arguments[0],arguments[1])',
                             offset, base64.b64encode(block).decode())
                assert ack == {'offset': offset, 'received': len(block),
                               'total_received': offset + len(block)}
                offset += len(block)
                receipt.write_text(json.dumps({'phase': 'UPLOADING', 'bytes': offset,
                                               'total': source.stat().st_size}, indent=2))
                if offset % (32 * 1024 * 1024) == 0:
                    print('CHECKPOINT_UPLOAD', offset, '/', source.stat().st_size, flush=True)
        result = invoke(m, 'window.nerFinishUpload()')
        assert result['status'] == 'EPOCH_11_CHECKPOINT_RESTORED'
        receipt.write_text(json.dumps(dict(result, bytes=offset), indent=2))
        print('GPU_CHECKPOINT_RESTORED', json.dumps(result), flush=True)
        m.switch_to_frame()
        m.execute_script("colab.global.notebook.cells.find(c=>c.cellId==='ner_all369_3').manualExecute()", sandbox=None)
        for _ in range(20):
            time.sleep(3)
            info = m.execute_script("return colab.global.notebook.cells.find(c=>c.cellId==='ner_all369_3').model.toJson().metadata.executionInfo||{}", sandbox=None)
            if info.get('status') == 'ok':
                break
        assert info.get('status') == 'ok', info
        m.execute_script("colab.global.notebook.cells.find(c=>c.cellId==='ner_all369_4').manualExecute()", sandbox=None)
        print('GPU_FINALIZATION_WORKER_LAUNCHED', flush=True)
    finally:
        m.delete_session()
    subprocess.run(['systemctl', '--user', 'start', 'legal-ner-colab-watch.service'], check=True)


if __name__ == '__main__':
    main()
