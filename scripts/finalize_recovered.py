"""Evaluate an unchanged, verified checkpoint after a terminated Colab run."""
import hashlib
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = 'phobert_span_all369_v1'
REPORT = ROOT / 'reports' / RUN
FOLDER = ROOT / 'checkpoints' / RUN
EXPECTED_MODEL = 'e99e2fd7cde7d386869e5444e7ff02460f9660090377d00dca7215adcb3ba01f'
STARTED = time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def status(phase, **extra):
    from src.dataset import write_json
    write_json(REPORT / 'worker_status.json', {
        'phase': phase, 'run_name': RUN, 'pid': os.getpid(),
        'elapsed_seconds': time.monotonic() - STARTED, **extra,
    })


def main():
    import torch
    from src.dataset import read_jsonl, write_json
    from src.training import evaluate_model, load_checkpoint, objective

    assert torch.cuda.is_available(), 'GPU required for final evaluation'
    assert sha(FOLDER / 'model.pt') == EXPECTED_MODEL
    assert not (REPORT / 'RESULTS.json').exists(), 'Preserve an existing final result'
    experiment = json.loads((REPORT / 'EXPERIMENT.json').read_text())
    history = json.loads((REPORT / 'history.json').read_text())
    assert [row['epoch'] for row in history] == list(range(1, 12))
    for name, expected in experiment['source_hashes'].items():
        assert sha(ROOT / name) == expected, name
    data_dir = ROOT / experiment['config']['dataset_dir']
    train = read_jsonl(data_dir / 'train.jsonl')
    val = read_jsonl(data_dir / 'val.jsonl')
    assert (len(train), len(val)) == (314, 55)
    assert not {d['group_id'] for d in train} & {d['group_id'] for d in val}
    assert sha(data_dir / 'manifest.json') == experiment['config']['dataset_manifest_sha256']
    hardware = {'device': 'cuda', 'gpu': torch.cuda.get_device_name(0),
                'torch': torch.__version__, 'cuda': torch.version.cuda}
    torch.set_num_threads(2)
    status('RESTORING_CHECKPOINT', gpu=hardware['gpu'], recovered_epoch=11)
    model, extra = load_checkpoint(FOLDER, 'cuda')
    assert extra['epoch'] == 11
    assert 'relations' not in model.ontology
    topk = extra['decoder_topk']
    model._inference_cache = {}
    status('VERIFYING_RESTORED_CHECKPOINT', gpu=hardware['gpu'], recovered_epoch=11)
    baseline, _ = evaluate_model(model, val, extra['thresholds'], topk)
    actual = baseline['entity_exact_span']['micro']
    recorded = history[-1]['validation']
    assert all(actual[key] == recorded[key] for key in ['tp', 'fp', 'fn', 'support']), (
        'Restored checkpoint does not reproduce recorded validation counts', actual, recorded)
    write_json(REPORT / 'RESTORED_CHECKPOINT_VERIFICATION.json', {
        'status': 'RECORDED_EPOCH_11_COUNTS_REPRODUCED',
        'checkpoint_sha256': EXPECTED_MODEL, 'recorded': recorded, 'reproduced': actual,
        'weights_updated': False, 'hardware': hardware,
    })
    trials = []
    best_score = -1.
    best_thresholds = dict(extra['thresholds'])
    for boundary in [.35, .5, .65]:
        for span in [.35, .5, .75]:
            thresholds = {'boundary_threshold': boundary, 'span_threshold': span}
            status('CALIBRATING', gpu=hardware['gpu'], trial=len(trials) + 1,
                   trials=9, recovered_epoch=11)
            metric, _ = evaluate_model(model, val, thresholds, topk)
            score = objective(metric)
            trials.append({'thresholds': thresholds, 'selection_score': score,
                           'micro': metric['entity_exact_span']['micro']})
            write_json(REPORT / 'THRESHOLD_TRIALS_PROGRESS.json', trials)
            if score > best_score:
                best_score, best_thresholds = score, thresholds
    assert sha(FOLDER / 'model.pt') == EXPECTED_MODEL
    write_json(FOLDER / 'inference_settings.json', {
        'weights_updated': False, 'base_checkpoint_sha256': EXPECTED_MODEL,
        'selection_split': 'Validation hyperparameters only', 'thresholds': best_thresholds,
    })
    result = {
        'status': 'TRAINED_ENTITY_ONLY_ON_GPU', 'run_name': RUN,
        'train_documents': 314, 'validation_documents': 55, 'test_documents': 0,
        'epochs_completed': 11, 'selected_epoch': 11, 'seed': extra['seed'],
        'hardware': experiment['hardware'], 'finalization_hardware': hardware,
        'source_hashes': experiment['source_hashes'],
        'dataset_manifest_sha256': experiment['config']['dataset_manifest_sha256'],
        'official_pretrained_only': True, 'fitted_parent_checkpoint_used': False,
        'relation_parameters': 0, 'independent_test_claimed': False,
        'threshold_trials': trials, 'thresholds': best_thresholds, 'results': {},
        'completion_reason': 'RECOVERED_UNCHANGED_EPOCH_11_AFTER_RUNTIME_TERMINATED_DURING_EPOCH_12',
        'interrupted_epoch': 12, 'last_observed_interrupted_step': 981,
        'weights_updated_during_finalization': False,
        'recovered_model_sha256': EXPECTED_MODEL,
        'training_completed_epoch_seconds': sum(row['seconds'] for row in history),
        'finalization_seconds': None,
    }
    model._inference_cache = None
    for split, documents in [('train', train), ('val', val)]:
        status('EVALUATING', gpu=hardware['gpu'], split=split, documents=len(documents),
               recovered_epoch=11)
        metric, predictions = evaluate_model(model, documents, best_thresholds, topk)
        result['results'][split] = metric
        write_json(REPORT / 'predictions' / (split + '.json'), predictions)
    result['finalization_seconds'] = time.monotonic() - STARTED
    write_json(REPORT / 'RESULTS.json', result)
    status('PACKAGING', gpu=hardware['gpu'], recovered_epoch=11)
    output = ROOT.parent / 'Legal_Entity_NER_All369_Trained.zip'
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for base in ['src', 'scripts', 'configs', 'data', 'resources/phobert-base',
                     'checkpoints/' + RUN, 'reports/' + RUN]:
            for path in (ROOT / base).rglob('*'):
                if (path.is_file() and not any(p in path.parts for p in ['__pycache__', '.cache'])
                        and path.suffix not in ['.bin', '.safetensors', '.lock']):
                    archive.write(path, ROOT.name + '/' + str(path.relative_to(ROOT)))
        for name in ['README.md', 'app.py', 'requirements.txt']:
            archive.write(ROOT / name, ROOT.name + '/' + name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    metrics = {split: dict(metric['entity_exact_span']['micro'],
                          macro_f1=metric['entity_exact_span']['macro_f1_supported_labels'])
               for split, metric in result['results'].items()}
    status('COMPLETE', gpu=hardware['gpu'], metrics=metrics, recovered_epoch=11,
           artifact={'path': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output)})
    print('RECOVERED_FINAL_EVALUATION_COMPLETE', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('FAILED', error=str(error), traceback=traceback.format_exc())
        raise
