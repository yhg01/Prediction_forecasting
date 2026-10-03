"""Validate the separately authorized draw extension against the frozen baseline."""
import json
from pathlib import Path
from train_insecure_code import digest, stable


def verify_bundle(root):
    bundle = json.loads((root / 'BUNDLE.json').read_text())
    for name, expected in bundle['files_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError('Extension bundle changed: ' + name)
    return bundle


def bind_extension(root, config, binding, batch, worker):
    verify_bundle(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['version'] != 'baseline90-extension-v1' or batch not in (1, 2):
        raise ValueError('Unapproved extension or batch')
    key = config['model_key']
    entry = protocol['models'][key]
    if digest(root / 'configs' / (key + '.json')) != entry['config_sha256']:
        raise ValueError('Original config changed')
    if worker.name != entry['worker'] or digest(worker) != entry['extension_worker_sha256']:
        raise ValueError('Unreviewed extension worker')
    original = root / 'originals' / key
    for name, expected in entry['original_files_sha256'].items():
        if digest(original / name) != expected:
            raise ValueError('Original baseline provenance changed')
    manifest = json.loads((original / 'manifest.json').read_text())
    complete = json.loads((original / 'complete.json').read_text())
    old_binding = manifest['inference_binding']
    if manifest['arm'] != 'base' or manifest['training_seed'] is not None or manifest['adapter_receipt_sha256'] is not None:
        raise ValueError('Source baseline is not untuned')
    if stable(old_binding) != manifest['inference_binding_sha256'] or stable(old_binding) != entry['original_binding_sha256']:
        raise ValueError('Original inference binding hash changed')
    if complete['status'] != 'completed' or complete['forecast_count'] != 30 or complete['manifest_sha256'] != digest(original / 'manifest.json') or complete['results_sha256'] != digest(original / 'results.jsonl'):
        raise ValueError('Invalid original completion receipt')
    # Only evaluator code differs: reviewed edits add disjoint draw IDs and provenance.
    # All actual model/runtime/prompt/generation/precision/GPU fields must be identical.
    comparable = dict(binding)
    comparable['evaluator_sha256'] = entry['original_worker_sha256']
    if comparable != old_binding:
        changed = sorted(k for k in set(comparable) | set(old_binding) if comparable.get(k) != old_binding.get(k))
        raise ValueError('Extension changed frozen scientific/runtime settings: ' + ', '.join(changed))
    for name, expected in protocol['unchanged_helpers_sha256'].items():
        if digest(root / 'scripts' / name) != expected:
            raise ValueError('Frozen helper changed: ' + name)
    binding['baseline_extension'] = {
        'version': protocol['version'], 'batch': batch,
        'replicate_start': batch * 10, 'replicate_stop': (batch + 1) * 10,
        'protocol_sha256': digest(root / 'protocol.json'),
        'helper_sha256': digest(__file__),
        'original_inference_binding_sha256': stable(old_binding),
        'original_manifest_sha256': digest(original / 'manifest.json')}
    return old_binding
