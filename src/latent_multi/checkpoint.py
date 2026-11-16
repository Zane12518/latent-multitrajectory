"""Resume identity checks. Content fingerprints detect changes, not authorship."""
import hashlib
import json
import platform
import random
import warnings
from pathlib import Path

import numpy as np
import torch

FORMAT = 'latent-multitrajectory-resume-v2'
LEGACY = 'latent-multitrajectory-resume-v1'
PROTOCOL = dict(version=1, native_maps_per_microbatch=2, prior_K=32,
                prior_outer_steps=16, prior_grad_outer_steps=[15],
                quality='native-stablemax-kl-act+shortest-DAG-mass',
                auxiliary='length-normalized', temperature=.5, sinkhorn_iterations=12,
                assignment_gradient='stopped', clip_norm=1., optimizer='AdamW')
REQUIRED = ('model', 'optimizer', 'rng', 'prior_rng', 'step', 'order_rng', 'order',
            'cursor', 'reference_visits', 'carry', 'actual_ids', 'calls',
            'model_training', 'initial_seed', 'stream', 'release_config', 'train_ids',
            'device_type', 'scheduler', 'scaler', 'quality_mode')


def content_identity(rows):
    # Includes row order and all input/reference values, not only puzzle IDs.
    payload = json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(algorithm='blake2b-256-canonical-json-v1', rows=len(rows),
                digest=hashlib.blake2b(payload, digest_size=32).hexdigest())


def environment(device):
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, device_type=device.type,
                cuda=torch.version.cuda if device.type == 'cuda' else None,
                cuda_devices=torch.cuda.device_count() if device.type == 'cuda' else 0)


def implementation_identity():
    root = Path(__file__).parent
    names = ('runtime.py', 'objective.py', 'maze.py', 'checkpoint.py',
             'vendor/gram/recursive_reasoning/gram.py', 'vendor/gram/losses.py',
             'vendor/gram/layers.py', 'vendor/gram/common.py', 'vendor/gram/sparse_embedding.py')
    return {name:hashlib.blake2b((root/name).read_bytes(),digest_size=32).hexdigest() for name in names}


def config_diff(old, new, prefix=''):
    changes = {}
    for key in sorted(set(old) | set(new)):
        path = prefix + key
        if key not in old or key not in new:
            changes[path] = dict(before=old.get(key), after=new.get(key))
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            changes.update(config_diff(old[key], new[key], path + '.'))
        elif old[key] != new[key]:
            changes[path] = dict(before=old[key], after=new[key])
    return changes


def require_keys(obj, keys, label):
    missing = set(keys) - set(obj)
    if missing:
        raise ValueError(f'{label}: missing required state {sorted(missing)}')


def validate(session, c, mode, arm, allow_changes):
    if mode not in ('exact', 'fork', 'new-experiment', 'legacy'):
        raise ValueError('unknown resume mode')
    legacy = c.get('format') == LEGACY
    if c.get('format') not in (FORMAT, LEGACY):
        raise ValueError('not a full training checkpoint; inference weights cannot resume training')
    if legacy != (mode == 'legacy'):
        raise ValueError('v1 requires explicit legacy mode; it cannot verify data content or exact recovery')
    require_keys(c, REQUIRED, 'checkpoint')
    if not legacy:
        require_keys(c, ('identity', 'arm', 'lineage'), 'v2 checkpoint')
        if c['identity'].get('data') != content_identity(session.train):
            raise ValueError('training input/reference content identity mismatch')
    if c['device_type'] != session.device.type:
        raise ValueError('resume requires the same CPU/CUDA RNG backend; use weights for inference')
    if c['train_ids'] != [r['id'] for r in session.train]:
        raise ValueError('training population/order mismatch')
    changes = config_diff(c['release_config'], session.config)
    allowed = set(allow_changes)
    if mode != 'new-experiment' and (changes or allowed):
        raise ValueError(f'scientific configuration mismatch: {sorted(changes)}; use explicit new-experiment mode')
    if mode == 'new-experiment':
        permitted = {'training.schedule', 'training.aux_coef', 'training.microbatches', 'training.updates'}
        if set(changes) != allowed or not allowed <= permitted:
            raise ValueError(f'list exactly the changed supported fields: {sorted(changes)}; allowed: {sorted(permitted)}')
    if c['quality_mode'] != 'canonical' or c['scheduler'] is not None or c['scaler'] is not None:
        raise ValueError('unsupported quality/scheduler/scaler state')
    if not isinstance(c['step'], int) or c['step'] < 0:
        raise ValueError('invalid training progress')
    if c['initial_seed'] != c['release_config']['training']['initial_seed']:
        raise ValueError('initial seed/configuration mismatch')
    if c['order'] and sorted(c['order']) != list(range(len(session.train))):
        raise ValueError('invalid saved data order')
    if not 0 <= c['cursor'] <= len(c['order']):
        raise ValueError('invalid data cursor')
    ids = set(c['train_ids'])
    if any(k not in ids or not isinstance(v, int) or v < 0 for k, v in c['reference_visits'].items()):
        raise ValueError('invalid reference visits')
    if any(k not in ids for k in c['actual_ids']):
        raise ValueError('invalid ACT exposure IDs')
    require_keys(c['calls'], ('native_outer_lanes', 'prior_outer_lanes'), 'call counters')
    if c['step'] > 0 and (c['carry'] is None or not c['order'] or not c['optimizer'].get('state')):
        raise ValueError('trained checkpoint is missing carry/order/Adam state')
    model = session.m.state_dict()
    if set(c['model']) != set(model):
        raise ValueError('model parameter names differ')
    for name, value in model.items():
        saved = c['model'][name]
        if saved.shape != value.shape or saved.dtype != value.dtype or not torch.isfinite(saved).all():
            raise ValueError(f'model tensor mismatch/nonfinite: {name}')
    opt = c['optimizer']; require_keys(opt, ('state', 'param_groups'), 'optimizer')
    expected = session.opt.state_dict()['param_groups']
    if [g['params'] for g in opt['param_groups']] != [g['params'] for g in expected]:
        raise ValueError('optimizer parameter binding differs')
    for saved, fresh in zip(opt['param_groups'], expected):
        for key in ('betas', 'eps', 'weight_decay', 'amsgrad', 'maximize'):
            if saved.get(key) != fresh.get(key):
                raise ValueError(f'optimizer configuration mismatch: {key}')
    params = [p for g in session.opt.param_groups for p in g['params']]
    for index, state in opt['state'].items():
        require_keys(state, ('step', 'exp_avg', 'exp_avg_sq'), 'Adam slot')
        if index not in range(len(params)):
            raise ValueError('unknown optimizer parameter')
        for key in ('exp_avg', 'exp_avg_sq'):
            if state[key].shape != params[index].shape or not torch.isfinite(state[key]).all():
                raise ValueError(f'invalid Adam {key}')
        if not 0 < float(state['step']) <= c['step']:
            raise ValueError('invalid Adam progress')
    r = c['rng']; require_keys(r, ('py', 'np', 'cpu', 'cuda'), 'RNG')
    random.Random().setstate(r['py']); random.Random().setstate(c['order_rng'])
    ns = r['np']; np.random.RandomState().set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.Generator().set_state(r['cpu'])
    torch.Generator(device=session.device).set_state(c['prior_rng'])
    if session.device.type == 'cuda' and len(r['cuda']) != torch.cuda.device_count():
        raise ValueError('CUDA RNG device count mismatch')
    if c['carry'] is not None:
        data = c['carry']['current_data']
        by_id = {row['id']: row for row in session.train}
        if data['ids'].numel() != 2:
            raise ValueError('ACT carry lane mismatch')
        for i, rid in enumerate(data['ids'].tolist()):
            if rid not in by_id or data['inputs'][i].tolist() != by_id[rid]['x']:
                raise ValueError('ACT carry input/content mismatch')
    previous_arm = c.get('arm')
    if previous_arm is not None and arm is not None and previous_arm != arm and mode == 'exact':
        raise ValueError('changing the training arm requires explicit fork mode')
    if not legacy:
        identity = c['identity']
        if identity.get('protocol') != PROTOCOL:
            raise ValueError('training implementation protocol mismatch')
        if identity.get('implementation') != implementation_identity():
            raise ValueError('training implementation content changed')
        if identity.get('resolved_model_config') != session.m.config.model_dump():
            raise ValueError('resolved model configuration changed')
        if identity.get('environment') != environment(session.device):
            raise ValueError('runtime environment mismatch; exact recovery is not verified')
        if identity.get('optimizer_slots') != sorted(opt['state']):
            raise ValueError('missing or unexpected optimizer slot')
    else:
        warnings.warn('LEGACY: data/reference content and historical implementation identity cannot be verified. This is not verified exact recovery.', RuntimeWarning, stacklevel=2)
    event = None
    if mode != 'exact':
        event = dict(mode=mode, parent_step=c['step'], parent_stream=c['stream'],
                     from_arm=previous_arm, to_arm=arm, config_changes=changes,
                     legacy_identity_unverified=legacy)
    return event
