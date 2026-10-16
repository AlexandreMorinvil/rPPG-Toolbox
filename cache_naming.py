"""Shared cache identities for the toolbox and dependency-light launcher.

Names describe the main settings; the digest distinguishes the complete preprocessing
configuration. Split selection and runtime worker counts are not clip identities.
"""

import hashlib
import json
import os
import re
from pathlib import Path

SETTINGS_FILENAME = 'preprocessing_settings.json'
TYPE_CODES = {'DiffNormalized': 'DN', 'Standardized': 'STD', 'Raw': 'RAW', '': 'NA'}


def flatten_settings(node, prefix=''):
    result = {}
    for key, value in node.items():
        dotted = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            result.update(flatten_settings(value, dotted))
        else:
            result[dotted] = value
    return result


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Yacs uses [''] placeholders; the launcher normalizes these to [].
        return [_canonical(item) for item in value if item != '']
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


UV_KEYS = ('LAYOUT', 'CHARTS', 'EXCLUDE_CHARTS', 'MIN_COVERAGE', 'FILL', 'MIN_PIXEL_VALID_FRACTION', 'RESAMPLE_TO_FS')


def _uv_values(values, prefix):
    """UV settings that change the clips, or None when the UV input mode is off."""
    if not values.get(prefix + '.UV.ENABLED'):
        return None
    uv = {}
    for key in UV_KEYS:
        value = values.get(f'{prefix}.UV.{key}')
        if isinstance(value, (list, tuple)):
            value = sorted(str(item).strip().lower() for item in value if str(item).strip())
        elif isinstance(value, str):
            value = value.strip().lower()
        uv[key.lower()] = value
    return _canonical(uv)


def uv_identity(data):
    """UV settings of a DATA node (None when off); shared by the vHRM cache name."""
    return _uv_values(flatten_settings(data, 'DATA'), 'DATA')


def preprocessing_identity(values, prefix):
    """All preprocessing settings, including dataset-specific transforms and true H."""
    start = prefix + '.PREPROCESS.'
    settings = {key[len(start):]: value for key, value in values.items()
                if key.startswith(start) and key != start + 'NUM_WORKERS'}
    uv = _uv_values(values, prefix)
    if uv:
        # The face crop does not apply to UV maps (config.py turns it off).
        settings = {key: value for key, value in settings.items() if not key.startswith('CROP_FACE.')}
    identity = {'version': 1, 'dataset': values.get(prefix + '.DATASET', ''),
                'fs': values[prefix + '.FS'], 'preprocess': settings}
    if uv:  # absent when off, so face-crop identities (and their caches) are unchanged
        identity['uv'] = uv
    return _canonical(identity)


def settings_path(directory):
    """Long-path access only for metadata in retained legacy Windows caches."""
    path = Path(directory) / SETTINGS_FILENAME
    if os.name == 'nt' and len(str(path.absolute())) >= 260:
        absolute = str(path.absolute())
        if not absolute.startswith('\\\\?\\'):
            absolute = ('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                        else '\\\\?\\' + absolute)
        return Path(absolute)
    return path


def legacy_cache_matches(directory, identity):
    """Old caches without metadata remain usable; recorded mismatches are not adopted."""
    directory = Path(directory)
    if not directory.is_dir():
        return False
    metadata = settings_path(directory)
    if not metadata.is_file():
        return True
    try:
        return json.loads(metadata.read_text(encoding='utf-8')) == identity
    except (OSError, ValueError):
        return False


def _token(value, limit=18):
    return re.sub(r'[^A-Za-z0-9+-]', '-', str(value)).strip('-')[:limit] or 'NA'


def compact_name(values, prefix):
    identity = preprocessing_identity(values, prefix)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]
    p = prefix + '.PREPROCESS.'
    types = '+'.join(TYPE_CODES.get(t, _token(t, 6)) for t in values[p + 'DATA_TYPE'])
    label = TYPE_CODES.get(values[p + 'LABEL_TYPE'], _token(values[p + 'LABEL_TYPE'], 6))
    if 'uv' in identity:
        crop = 'UV' if identity['uv'].get('layout') == 'packed' else 'UVSTD'
    else:
        crop = values[p + 'CROP_FACE.BACKEND'] if values[p + 'CROP_FACE.DO_CROP_FACE'] else 'FULL'
    chunk = values[p + 'CHUNK_LENGTH'] if values[p + 'DO_CHUNK'] else 'FULL'
    mode = '_UNSUP' if prefix == 'UNSUPERVISED.DATA' else ''
    return (f"{_token(identity['dataset'])}_{values[p + 'RESIZE.W']}x{values[p + 'RESIZE.H']}"
            f"_C{chunk}_{_token(types)}_L{label}_{_token(crop, 8)}_F{identity['fs']}{mode}_{digest}")


def legacy_name(values, prefix):
    """Reproduce old generic names exactly (including the old non-TEST height bug)."""
    p = prefix + '.PREPROCESS.'
    height = values[p + 'RESIZE.H'] if prefix == 'TEST.DATA' else values[p + 'RESIZE.W']
    parts = [values.get(prefix + '.DATASET', ''), f"SizeW{values[p + 'RESIZE.W']}",
             f'SizeH{height}', f"ClipLength{values[p + 'CHUNK_LENGTH']}",
             'DataType' + '_'.join(values[p + 'DATA_TYPE']),
             'DataAug' + '_'.join(values[p + 'DATA_AUG']), f"LabelType{values[p + 'LABEL_TYPE']}",
             f"Crop_face{values[p + 'CROP_FACE.DO_CROP_FACE']}", f"Backend{values[p + 'CROP_FACE.BACKEND']}",
             f"Large_box{values[p + 'CROP_FACE.USE_LARGE_FACE_BOX']}",
             f"Large_size{values[p + 'CROP_FACE.LARGE_BOX_COEF']}",
             f"Dyamic_Det{values[p + 'CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION']}",
             f"det_len{values[p + 'CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY']}",
             f"Median_face_box{values[p + 'CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX']}"]
    if prefix == 'UNSUPERVISED.DATA':
        parts.append('unsupervised')
    return '_'.join(str(part) for part in parts)