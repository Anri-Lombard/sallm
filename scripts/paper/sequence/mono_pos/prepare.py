#!/usr/bin/env python3
"""Release only the three frozen GDN Mono POS units for official evaluation."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path('/scratch/lmbanr001/masters')
BUNDLE = ROOT / 'sallm_snapshots/full-matrix-execution-20260916-v1'
RESULT = ROOT / 'sallm/results/full_matrix_execution_20260916_v1'
SELECTION = ROOT / 'sallm/results/general_sequence_validation_20260916_hex_v5_final_amendment_v2/selection/SELECTION.json'
OUTPUT = RESULT / 'gdn-mono-pos-official-20260920-v2'
SCOPE = {27: 'tsn', 28: 'xho', 29: 'zul'}
KNOWN = ('config.json', 'adapter_config.json', 'adapter_model.safetensors',
         'adapter_model.bin', 'model.safetensors', 'pytorch_model.bin',
         'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root):
    files = sorted(path for path in root.rglob('*')
                   if path.is_file() and '.cache' not in path.relative_to(root).parts)
    if not files:
        raise ValueError(f'Empty model tree: {root}')
    payload = ''.join(f'{sha256(path)}  ./{path.relative_to(root).as_posix()}\n'
                      for path in files)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    units = json.loads((BUNDLE / 'frozen-inventory/official_units.json').read_text())
    template = json.loads((BUNDLE / 'general_sequence_protocol_template.json').read_text())
    selection = json.loads(SELECTION.read_text())
    prompts = selection['selected_prompts']['pos']
    if set(prompts) != {'tsn', 'xho', 'zul'} or any(not 1 <= int(p) <= 4 for p in prompts.values()):
        raise ValueError('Frozen POS prompt selection is incomplete')
    source = Path(template['source_snapshot']['path'])
    if sha256(source / 'SNAPSHOT_MANIFEST.sha256') != template['source_snapshot']['manifest_sha256']:
        raise ValueError('Source snapshot manifest changed')
    protocols = {}
    for index, language in SCOPE.items():
        unit = units[index]
        if (unit['array_index'], unit['architecture'], unit['task_group'],
            unit['regime'], unit['languages'], unit['cell_ids']) != (
            index, 'gdn', 'pos', 'Mono', [language], [f'gdn:pos:{language}:mono']):
            raise ValueError(f'Frozen inventory changed at {index}')
        if (RESULT / f'official/sequence/{index}.json').exists():
            raise FileExistsError(f'Official unit {index} already exists')
        base_ref = unit['binding']['base']
        base = Path(base_ref['path'])
        adapter_ref = unit['binding']['adapter_candidates']
        if len(adapter_ref) != 1:
            raise ValueError(f'Ambiguous adapter for unit {index}')
        raw_adapter = adapter_ref[0]['path']
        adapter_path, declared_weight = raw_adapter.split('; adapter_model_sha256=')
        adapter = Path(adapter_path)
        if not base.is_dir() or not adapter.is_dir():
            raise FileNotFoundError(f'Missing base or adapter for unit {index}')
        if tree_sha256(base) != base_ref['expected_tree_sha256']:
            raise ValueError(f'Base tree hash mismatch for unit {index}')
        if sha256(adapter / 'adapter_model.safetensors') != declared_weight:
            raise ValueError(f'Adapter weight hash mismatch for unit {index}')
        protocol = deepcopy(template)
        protocol['models'] = {'gdn': {
            'dtype': 'bfloat16', 'merge_lora': False, 'tie_word_embeddings': None,
            'execution_status': 'runnable',
            'base_path': str(base), 'adapter_path': str(adapter),
            'base_tree_sha256': tree_sha256(base),
            'adapter_tree_sha256': tree_sha256(adapter),
            'base_files': {name: sha256(base / name) for name in KNOWN if (base / name).is_file()},
            'adapter_files': {name: sha256(adapter / name) for name in KNOWN if (adapter / name).is_file()},
        }}
        protocols[index] = protocol
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for index, protocol in protocols.items():
        write_once(OUTPUT / f'protocols/{index}.json', protocol)
    write_once(OUTPUT / 'release/POS_TEST_ACCESS_RELEASED.json', {
        'schema': 'sallm.general_sequence_test_release/v1',
        'task': 'pos', 'selection_sha256': sha256(SELECTION),
        'authorized': True, 'one_time_test': True, 'no_score_based_retry': True,
        'scope': sorted(SCOPE), 'reason': 'missing frozen GDN Mono POS official cells',
    })
    write_once(OUTPUT / 'PREPARED.json', {
        'schema': 'sallm.gdn_mono_pos_official_preparation/v1',
        'status': 'PREPARED', 'indices': sorted(SCOPE), 'test_accessed': False,
        'selection_sha256': sha256(SELECTION),
        'protocol_sha256': {str(index): sha256(OUTPUT / f'protocols/{index}.json')
                            for index in SCOPE},
    })
    print('GDN_MONO_POS_PREPARED indices=27,28,29 test_accessed=false')


if __name__ == '__main__':
    main()
