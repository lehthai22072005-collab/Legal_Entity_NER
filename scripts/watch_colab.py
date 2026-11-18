"""Monitor the authorized Colab run and keep one verified intermediate backup."""
import argparse
import base64
import fcntl
import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from colab_artifacts_all import ROOT, browser, callback, connection


def record(**values):
    folder = ROOT / 'colab/received'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'watch_status.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(values, observed_at=datetime.now(timezone.utc).isoformat())
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(current, ensure_ascii=False, indent=2))
    pending.replace(path)


def download_backup(m, snapshot):
    target = ROOT / 'colab/NER_All369_Checkpoint_Backup.zip'
    pending = target.with_suffix('.zip.partial')
    receipt = target.with_suffix('.json')
    info = json.loads(receipt.read_text()) if receipt.exists() else {}
    if target.exists() and info.get('sha256') == snapshot['sha256']:
        return
    if info.get('pending_sha256') != snapshot['sha256']:
        pending.unlink(missing_ok=True)
    receipt.write_text(json.dumps(dict(info, pending_sha256=snapshot['sha256'])))
    offset = pending.stat().st_size if pending.exists() else 0
    assert 0 <= offset <= snapshot['bytes']
    with pending.open('ab') as stream:
        while offset < snapshot['bytes']:
            data = callback(m, f"window.nckhTransfer('checkpoint_backup',{offset},4194304)")
            blob = base64.b64decode(data['base64'], validate=True)
            assert data['offset'] == offset and data['total'] == snapshot['bytes']
            assert 0 < len(blob) == data['bytes'] <= 4194304
            stream.write(blob)
            stream.flush()
            offset += len(blob)
            record(backup_downloaded_bytes=offset, backup_total_bytes=snapshot['bytes'])
    with pending.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == snapshot['sha256']
    with zipfile.ZipFile(pending) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('BACKUP_MANIFEST.json'))
        assert manifest['kind'] == 'INTERMEDIATE_CHECKPOINT_NOT_FINAL_RESULTS'
        assert manifest['run_name'] == 'phobert_span_all369_v1'
        assert manifest['epoch'] == snapshot['epoch']
        assert (manifest['train_documents'], manifest['validation_documents']) == (314, 55)
        assert manifest['input_sha256'] == '13629a93823051971ce193a140a67209ab6deb233e46c1ea978d17fa72bd4247'
        with archive.open('checkpoints/phobert_span_all369_v1/model.pt') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['model_sha256']
    pending.replace(target)
    receipt.write_text(json.dumps(snapshot, indent=2))
    record(backup_epoch=snapshot['epoch'], backup_sha256=snapshot['sha256'], backup_verified=True)


def observe():
    lock_path = ROOT / 'colab/.browser.lock'
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        m = connection()
        try:
            state = callback(m, 'window.nckhPoll()')
            folder = ROOT / 'colab/received'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'worker_status.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
            record(phase=state['phase'], progress=state.get('progress.json'), connection_error=None,
                   worker_observed_at=datetime.now(timezone.utc).isoformat())
            if state['phase'] == 'TRAINING' and state.get('history.json'):
                snapshot = callback(m, 'window.nckhSnapshot()')
                if snapshot.get('available'):
                    download_backup(m, snapshot)
            return state['phase']
        finally:
            m.delete_session()


def main(interval=60, once=False):
    while True:
        try:
            phase = observe()
            if phase == 'COMPLETE':
                browser('download', 8)
                if (ROOT / 'colab/Legal_Entity_NER_All369_Trained.zip').exists():
                    record(phase='COMPLETE_ARTIFACT_DOWNLOADED_PENDING_VERIFICATION')
                    return
                if not once:
                    continue
            if phase == 'FAILED':
                record(phase='FAILED_REQUIRES_INSPECTION')
                return
        except Exception as error:
            record(phase='MONITOR_ERROR_LIVE_WORKER_STATE_UNKNOWN',
                   connection_error=str(error)[:1200])
            print(type(error).__name__, str(error)[:300], flush=True)
        if once:
            return
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    main(args.interval, args.once)
