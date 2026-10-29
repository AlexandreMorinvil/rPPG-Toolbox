'use strict';

const TOKEN = document.querySelector('meta[name="launcher-token"]').content;
const $ = id => document.getElementById(id);

const S = {
  boot: null,
  fields: [],
  fieldMap: {},
  values: null,
  extras: {},
  backend: 'docker',
  source: null,
  syncLinks: {},
  validation: null,
  validateTimer: null,
  validateSeq: 0,
  preflight: null,
  jobs: [],
  groups: [],
  gpus: [],
  selectedJob: null,
  selectedGroup: null,
  log: null,
  cache: { sets: [], samples: [], sample: null, frames: null, meta: null, frame: 0, timer: null, label: null },
  results: { experiments: [], selected: null, detail: null, analysis: null, recording: null, sort: 'mae', compare: [] },
};

// ------------------------------------------------------------------ helpers
function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
}

async function api(path, { method = 'GET', body, params } = {}) {
  const query = params ? '?' + new URLSearchParams(params).toString() : '';
  const init = { method, headers: { 'X-Launcher-Token': TOKEN } };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  const response = await fetch(path + query, init);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 403 && /token/i.test(payload.error || '') && !S.stale) {
      S.stale = true;
      toast('The launcher was restarted. Reload this page (F5) to reconnect.', 'error', 3600000);
    }
    const error = new Error(payload.error || response.statusText);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function fileUrl(path, download) {
  const params = new URLSearchParams({ path, t: TOKEN });
  if (download) params.set('download', '1');
  return '/api/results/file?' + params.toString();
}

function toast(message, kind = 'info', ms = 4500) {
  const el = $('toast');
  el.textContent = message;
  el.className = 'toast ' + kind;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.add('hidden'), ms);
}

function fmtBytes(bytes) {
  if (!Number.isFinite(bytes)) return '';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (bytes >= 1024 && i < units.length - 1) { bytes /= 1024; i++; }
  return `${bytes.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function fmtDate(value) {
  if (!value) return '';
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
  return date.toLocaleString();
}

function fmtDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '';
  const h = Math.floor(seconds / 3600), m = Math.floor(seconds % 3600 / 60), s = Math.floor(seconds % 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}

const num = (v, digits = 2) => (v === null || v === undefined || !Number.isFinite(v)) ? '–' : (+v).toFixed(digits);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'text') el.textContent = value;
    else if (key === 'html') el.innerHTML = value;
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function openModal(title, body, buttons = [], wide = false) {
  $('modalTitle').textContent = title;
  const container = $('modalBody');
  container.replaceChildren(typeof body === 'string' ? h('div', { html: body }) : body);
  const foot = $('modalFoot');
  foot.replaceChildren(...buttons.map(b => h('button', {
    type: 'button', class: b.primary ? 'primary' : (b.danger ? 'danger' : ''), id: b.id, disabled: b.disabled,
    onclick: b.onClick,
  }, b.label)));
  $('modal').classList.toggle('wide', wide);
  $('modal').classList.remove('hidden');
}

function closeModal() {
  $('modal').classList.add('hidden');
  S.preflight = null;
}

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  if (name === 'jobs') pollJobs();
  if (name === 'cache' && !S.cache.sets.length) loadCacheSets();
  if (name === 'results') loadResults();
  if (name === 'settings') renderSettings();
}

// --------------------------------------------------------- path conversion
function normHost(path) {
  let p = String(path).replace(/\\/g, '/').replace(/\/+$/, '');
  if (S.boot.platform === 'nt') p = p.toLowerCase();
  return p;
}

function hostToContainer(hostPath) {
  const target = normHost(hostPath);
  let best = null;
  for (const mount of S.boot.docker_mounts) {
    const root = normHost(mount.host);
    if (target === root || target.startsWith(root + '/')) {
      if (!best || root.length > best.root.length) best = { root, mount };
    }
  }
  if (!best) return null;
  const rest = String(hostPath).replace(/\\/g, '/').slice(best.root.length).replace(/^\/+/, '');
  return best.mount.container + (rest ? '/' + rest : '');
}

function containerToHost(value) {
  for (const mount of S.boot.docker_mounts) {
    if (value === mount.container || value.startsWith(mount.container + '/')) {
      const rest = value.slice(mount.container.length).replace(/^\/+/, '');
      return mount.host + (rest ? (S.boot.platform === 'nt' ? '\\' : '/') + rest.split('/').join(S.boot.platform === 'nt' ? '\\' : '/') : '');
    }
  }
  return null;
}

// ------------------------------------------------------------------ schema
function isActive(field, v) {
  const when = field.when || {};
  if (when.modes && !when.modes.includes(v.TOOLBOX_MODE)) return false;
  if (when.needs_valid && v['TEST.USE_LAST_EPOCH']) return false;
  if (when.models && !when.models.includes(v['MODEL.NAME'])) return false;
  if (when.dataset && !when.dataset[1].includes(v[when.dataset[0] + '.DATASET'])) return false;
  if (when.wandb && !v['WANDB.ENABLED']) return false;
  if (when.equals && Object.entries(when.equals).some(([key, allowed]) => !allowed.includes(v[key]))) return false;
  return true;
}

// Mirrors schema.cross_validation_enabled: the config runs as one training per fold.
function cvEnabled(v = S.values) {
  const field = S.fieldMap['CROSS_VALIDATION.ENABLED'];
  return !!(v && field && isActive(field, v) && v['CROSS_VALIDATION.ENABLED']);
}

function activeSplits(v) {
  if (v.TOOLBOX_MODE === 'train_and_test') return ['TRAIN.DATA', ...(v['TEST.USE_LAST_EPOCH'] ? [] : ['VALID.DATA']), 'TEST.DATA'];
  if (v.TOOLBOX_MODE === 'only_test') return ['TEST.DATA'];
  if (v.TOOLBOX_MODE === 'unsupervised_method') return ['UNSUPERVISED.DATA'];
  return [];
}

function sameValue(a, b) {
  const clean = x => Array.isArray(x) ? JSON.stringify(x.filter(i => i !== '')) : JSON.stringify(x);
  return clean(a) === clean(b);
}

const isStructural = key => key === 'TOOLBOX_MODE' || key === 'MODEL.NAME' || key.endsWith('.DATASET')
  || key === 'TEST.USE_LAST_EPOCH' || key === 'WANDB.ENABLED'
  || S.fields.some(field => field.when?.equals && key in field.when.equals);

const fieldId = key => 'f_' + key.replace(/[^A-Za-z0-9]/g, '_');
const slug = text => text.toLowerCase().replace(/[^a-z0-9]+/g, '-');

// ------------------------------------------------------------------ editor
function populateConfigSelect() {
  const selected = $('configSelect').value;
  const first = S.boot.configs.find(c => c.group === 'docker') || S.boot.configs[0];
  selectConfig(S.boot.configs.some(c => c.path === selected) ? selected : first?.path || '');
  const blank = $('blankMode');
  blank.replaceChildren(...S.boot.schema.modes.map(m => h('option', { value: m }, S.boot.schema.mode_labels[m])));
}

function selectConfig(path) {
  $('configSelect').value = path;
  $('configSelectionLabel').textContent = path || 'No configs available';
  $('configPicker').querySelector('summary').title = path || 'No configs available';
  $('loadConfig').disabled = !path;
  renderConfigTree();
}

function renderConfigTree() {
  const tree = $('configTree');
  const expanded = new Set([...tree.querySelectorAll('details[open]')].map(el => el.dataset.path));
  const query = $('configSearch').value.trim().toLowerCase();
  const selected = $('configSelect').value;
  const groups = { docker: ['Docker configs', 'docker/configs'], local: ['Local configs', 'configs'] };
  let count = 0;
  tree.replaceChildren();

  const renderFolder = (folder, name, path, root = false) => {
    const details = h('details', {
      class: 'config-folder', 'data-path': path,
      open: !!query || root || expanded.has(path) || selected.startsWith(path + '/'),
    });
    details.append(h('summary', {}, h('span', { class: 'config-folder-name' }, name),
      root ? h('span', { class: 'config-root-path' }, path + '/') : null,
      h('span', { class: 'config-folder-count' }, folder.count)));
    const children = h('div', { class: 'config-folder-children' });
    for (const [childName, child] of [...folder.folders].sort(([a], [b]) => a.localeCompare(b))) {
      children.append(renderFolder(child, childName, path + '/' + childName));
    }
    for (const item of folder.files.sort((a, b) => a.path.localeCompare(b.path))) {
      const detail = item.error ? 'Unreadable config' : [S.boot.schema.mode_labels[item.mode] || item.mode,
        item.model, item.train, item.test !== item.train ? item.test : ''].filter(Boolean).join(' · ');
      children.append(h('button', {
        type: 'button', class: 'config-file' + (item.path === selected ? ' selected' : ''),
        'aria-pressed': String(item.path === selected), title: item.path + (item.error ? '\n' + item.error : ''),
        onclick: () => {
          selectConfig(item.path);
          $('configPicker').open = false;
          $('configPicker').querySelector('summary').focus();
        },
      }, h('span', { class: 'config-file-name' }, item.path.split('/').pop()),
      h('span', { class: 'config-file-detail' + (item.error ? ' bad' : '') }, detail)));
    }
    details.append(children);
    return details;
  };

  for (const [group, [label, rootPath]] of Object.entries(groups)) {
    const root = { folders: new Map(), files: [], count: 0 };
    for (const item of S.boot.configs.filter(c => c.group === group)) {
      const searchable = [item.path, label, item.mode, S.boot.schema.mode_labels[item.mode], item.model, item.train, item.test].join(' ').toLowerCase();
      if (query && !query.split(/\s+/).every(term => searchable.includes(term))) continue;
      count++;
      const parts = item.path.slice(rootPath.length + 1).split('/');
      let folder = root;
      folder.count++;
      for (const part of parts.slice(0, -1)) {
        if (!folder.folders.has(part)) folder.folders.set(part, { folders: new Map(), files: [], count: 0 });
        folder = folder.folders.get(part);
        folder.count++;
      }
      folder.files.push(item);
    }
    if (root.count) tree.append(renderFolder(root, label, rootPath, true));
  }
  $('configCount').textContent = query ? `${count} of ${S.boot.configs.length} configs match` : `${count} configs · Expand a folder to browse`;
  if (!count) tree.append(h('p', { class: 'muted pad' }, query ? 'No matching configs. Try another search.' : 'No config files found.'));
}

async function loadConfig(path) {
  try {
    const data = await api('/api/config', { params: { path } });
    setValues(data.values, data.extras, data.backend, `Loaded ${data.path}`, data.path);
    if (Object.keys(data.type_errors || {}).length) toast('Some values in the file have the wrong type; see validation.', 'warn');
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function newBlank(mode) {
  const data = await api('/api/defaults');
  const v = data.values;
  v.TOOLBOX_MODE = mode;
  const docker = S.backend === 'docker';
  v['LOG.PATH'] = docker ? '/runs/my_experiment' : 'runs/my_experiment';
  for (const split of ['TRAIN.DATA', 'VALID.DATA', 'TEST.DATA', 'UNSUPERVISED.DATA']) {
    v[split + '.CACHED_PATH'] = docker ? '/cache' : 'PreprocessedData';
    v[split + '.DATA_PATH'] = docker ? '/data/' : '';
    v[split + '.FS'] = 30;
    v[split + '.DO_PREPROCESS'] = true;
  }
  if (mode === 'train_and_test') { v['TRAIN.DATA.END'] = 0.7; v['VALID.DATA.BEGIN'] = 0.7; v['VALID.DATA.END'] = 0.8; v['TEST.DATA.BEGIN'] = 0.8; }
  v['TEST.METRICS'] = ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA'];
  v['UNSUPERVISED.METRICS'] = ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR'];
  v['UNSUPERVISED.DATA.PREPROCESS.DATA_TYPE'] = ['Raw'];
  v['UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE'] = 'Raw';
  setValues(v, {}, S.backend, `New ${S.boot.schema.mode_labels[mode]} config`, null);
}

function setValues(values, extras, backend, label, source) {
  S.values = values;
  S.syncLinks = {};
  S.resumeFrom = null;
  S.extras = extras || {};
  S.source = source;
  setBackend(backend, false);
  $('sourceInfo').textContent = label || '';
  $('modelBanner').classList.add('hidden');
  renderForm();
  scheduleValidate(0);
}

function setBackend(backend, revalidate = true) {
  S.backend = backend;
  document.querySelectorAll('input[name="backend"]').forEach(r => { r.checked = r.value === backend; });
  if (revalidate && S.values) scheduleValidate(0);
}

function fieldVisible(field) {
  if (!isActive(field, S.values)) return false;
  if (!field.advanced || $('showAdvanced').checked) return true;
  if (!sameValue(S.values[field.key], field.default)) return true;
  const messages = S.validation ? [...S.validation.errors, ...S.validation.warnings] : [];
  return messages.some(m => m.key === field.key);
}

function renderForm() {
  updateSyncControls();
  const form = $('configForm');
  const scroll = form.scrollTop;
  form.replaceChildren();
  const nav = $('sectionNav');
  nav.replaceChildren();
  if (!S.values) { form.append(h('p', { class: 'muted' }, 'Load a config or start a blank one.')); return; }
  const sections = new Map();
  for (const field of S.fields) {
    if (!sections.has(field.section)) sections.set(field.section, []);
    sections.get(field.section).push(field);
  }
  for (const [name, fields] of sections) {
    const visible = fields.filter(fieldVisible);
    if (!visible.length) continue;
    const id = 'sec-' + slug(name);
    const section = h('section', { class: 'form-section', id }, h('h3', {}, name), h('div', { class: 'derived', 'data-section': name }));
    let group = null;
    let grid = h('div', { class: 'field-grid' });
    section.append(grid);
    for (const field of visible) {
      if (field.group && field.group !== group) {
        group = field.group;
        section.append(h('h4', {}, group));
        grid = h('div', { class: 'field-grid' });
        section.append(grid);
      }
      grid.append(renderField(field));
    }
    form.append(section);
    nav.append(h('a', { href: '#' + id, 'data-section': name, onclick: e => { e.preventDefault(); $(id).scrollIntoView({ behavior: 'smooth' }); } },
      h('span', {}, name), h('span', { class: 'count' })));
  }
  form.scrollTop = scroll;
  renderValidation();
}

function renderField(field) {
  const key = field.key;
  const value = S.values[key];
  const id = fieldId(key);
  const wrap = h('div', { class: 'field' + (field.type === 'multi' || field.type === 'text' ? ' wide' : ''), 'data-key': key });
  wrap.append(h('label', { for: id }, field.label, field.required ? h('span', { class: 'req' }, ' *') : null,
    field.advanced ? h('span', { class: 'adv' }, ' advanced') : null));
  const control = h('div', { class: 'control' });
  const changed = v => onFieldChange(key, v);
  let input;
  switch (field.type) {
    case 'bool':
      input = h('input', { type: 'checkbox', id, onchange: e => changed(e.target.checked) });
      input.checked = !!value;
      control.append(h('label', { class: 'switch' }, input, h('span', {}, value ? 'On' : 'Off')));
      input.addEventListener('change', e => { e.target.nextSibling.textContent = e.target.checked ? 'On' : 'Off'; });
      break;
    case 'enum':
    case 'enum_list': {
      input = h('select', { id, onchange: e => changed(field.type === 'enum_list' ? [e.target.value] : e.target.value) });
      const current = field.type === 'enum_list' ? (value || [])[0] : value;
      if (!current || !field.options.includes(current)) input.append(h('option', { value: current || '' }, current ? `${current} (unsupported)` : '— choose —'));
      for (const option of field.options) input.append(h('option', { value: option }, (field.option_labels || {})[option] || option));
      input.value = current || '';
      control.append(input);
      break;
    }
    case 'multi': {
      const list = h('div', { class: 'chips', id });
      const selected = value || [];
      for (const option of field.options) {
        const box = h('input', { type: 'checkbox' });
        box.checked = selected.includes(option);
        box.addEventListener('change', () => {
          const current = (S.values[key] || []).filter(x => x !== option);
          if (box.checked) current.push(option);
          changed(current);
          renderFieldInPlace(key);
        });
        const order = selected.indexOf(option);
        list.append(h('label', { class: 'chip' + (box.checked ? ' on' : '') }, box, option, order >= 0 && selected.length > 1 ? h('b', {}, String(order + 1)) : null));
      }
      for (const extra of selected.filter(x => !field.options.includes(x))) list.append(h('span', { class: 'chip bad' }, extra));
      control.append(list);
      break;
    }
    case 'int':
      input = h('input', { type: 'number', id, step: '1', min: field.min, max: field.max, onchange: e => changed(e.target.value === '' ? '' : (Number.isInteger(+e.target.value) ? +e.target.value : e.target.value)) });
      input.value = value ?? '';
      control.append(input);
      break;
    case 'float':
      input = h('input', { type: 'text', id, inputmode: 'decimal', onchange: e => { const t = e.target.value.trim(); changed(t !== '' && Number.isFinite(+t) ? +t : t); } });
      input.value = value ?? '';
      control.append(input);
      break;
    case 'floatpair':
      input = h('input', { type: 'text', id, onchange: e => changed(e.target.value) });
      input.value = Array.isArray(value) ? value.join(', ') : value;
      control.append(input);
      break;
    case 'list':
      input = h('input', { type: 'text', id, placeholder: 'comma separated', onchange: e => changed(e.target.value.split(',').map(s => s.trim()).filter(Boolean)) });
      input.value = (value || []).join(', ');
      control.append(input);
      break;
    case 'yamllist':
      input = h('input', { type: 'text', id, onchange: e => changed(e.target.value) });
      input.value = JSON.stringify(value);
      control.append(input);
      break;
    case 'text':
      input = h('textarea', { id, rows: '2', onchange: e => changed(e.target.value) });
      input.value = value ?? '';
      control.append(input);
      break;
    default:
      input = h('input', { type: 'text', id, spellcheck: 'false', onchange: e => changed(e.target.value.trim()) });
      input.value = value ?? '';
      control.append(input);
      if (field.type === 'path') control.append(h('button', { type: 'button', class: 'small', onclick: () => browseDialog(field) }, 'Browse…'));
  }
  if (syncEnabled() && S.syncLinks[key]) {
    control.querySelectorAll('input, select, textarea, button').forEach(el => { el.disabled = true; });
    wrap.classList.add('linked-field');
    wrap.append(h('div', { class: 'sync-link' }, h('span', {}, 'Linked to training'),
      h('button', { type: 'button', class: 'small', 'aria-label': `Unlink ${field.label} from training`, onclick: () => {
        delete S.syncLinks[key];
        renderForm();
      } }, 'Unlink')));
  }
  wrap.append(control);
  if (field.help) wrap.append(h('div', { class: 'help' }, field.help));
  wrap.append(h('div', { class: 'path-info' }), h('div', { class: 'msgs' }));
  return wrap;
}

function renderFieldInPlace(key) {
  const old = document.querySelector(`.field[data-key="${CSS.escape(key)}"]`);
  if (old) old.replaceWith(renderField(S.fieldMap[key]));
  renderValidation();
}

function onFieldChange(key, value) {
  if (syncEnabled() && S.syncLinks[key]) return;
  const previous = S.values[key];
  S.values[key] = value;
  if (key === 'TOOLBOX_MODE' && value !== 'train_and_test') S.syncLinks = {};
  const linked = syncTrainingValue(key, value);
  if (key === 'MODEL.NAME' && value && value !== previous) offerModelDefaults(value);
  if (isStructural(key) || linked.length) renderForm();
  scheduleValidate();
}

// Links are editor-session state only; saved values remain ordinary YAML.
const SYNC_TARGETS = ['VALID.DATA', 'TEST.DATA'];
const SYNC_VHRM_FIELDS = new Set([
  'VHRM.VIEWS', 'VHRM.CROP_MODE_BY_VIEW', 'VHRM.PPG_SOURCE',
  'VHRM.RPEAK_PULSE_DELAY_SEC', 'VHRM.RPEAK_GAUSSIAN_SIGMA',
  'VHRM.HR_SOURCE', 'VHRM.VIDEO_DECODER', 'VHRM.CACHE_DTYPE',
  'VHRM.SHOW_VIDEO_PROGRESS',
]);

function syncEnabled() {
  return S.values?.TOOLBOX_MODE === 'train_and_test';
}

function preprocessingSyncField(field) {
  const suffix = field.key.slice('TRAIN.DATA.'.length);
  return suffix.startsWith('PREPROCESS.') || suffix === 'DATA_FORMAT' || suffix === 'FS';
}

function syncFields() {
  return S.fields.filter(field => {
    if (!field.key.startsWith('TRAIN.DATA.')) return false;
    const suffix = field.key.slice('TRAIN.DATA.'.length);
    return (preprocessingSyncField(field) || suffix === 'DO_PREPROCESS' || SYNC_VHRM_FIELDS.has(suffix))
      && SYNC_TARGETS.every(split => S.fieldMap[`${split}.${suffix}`]?.type === field.type);
  });
}

function syncTarget(source, split) {
  return split + source.slice('TRAIN.DATA'.length);
}

function cloneSyncValue(value) {
  if (Array.isArray(value)) return value.map(cloneSyncValue);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, cloneSyncValue(v)]));
  return value;
}

function syncTrainingValue(key, value) {
  if (!syncEnabled()) return [];
  const changed = [];
  for (const [target, source] of Object.entries(S.syncLinks)) {
    if (source !== key) continue;
    S.values[target] = cloneSyncValue(value);
    changed.push(target);
  }
  return changed;
}

function updateSyncControls() {
  const preset = syncFields().filter(preprocessingSyncField);
  const total = preset.length * SYNC_TARGETS.length;
  const linked = preset.reduce((n, field) => n + SYNC_TARGETS.filter(split => S.syncLinks[syncTarget(field.key, split)] === field.key).length, 0);
  const count = Object.keys(S.syncLinks).length;
  const box = $('linkPreprocess');
  box.disabled = !syncEnabled();
  box.checked = total > 0 && linked === total;
  box.indeterminate = count > 0 && !box.checked;
  $('syncSettings').disabled = !syncEnabled();
  $('syncSummary').textContent = count ? `${count} linked field${count === 1 ? '' : 's'}` : '';
}

function applySyncLinks(links) {
  if (!syncEnabled()) return;
  S.syncLinks = { ...links };
  for (const [target, source] of Object.entries(S.syncLinks)) S.values[target] = cloneSyncValue(S.values[source]);
  closeModal();
  renderForm();
  scheduleValidate(0);
}

function previewSyncLinks(links, back) {
  if (!syncEnabled()) return;
  const changes = Object.entries(links).filter(([target, source]) => !sameValue(S.values[target], S.values[source]));
  const removed = Object.keys(S.syncLinks).filter(target => !links[target]).length;
  const body = h('div', { class: 'form-stack' },
    h('p', {}, `${Object.keys(links).length} destination fields will follow training. ${changes.length} current values will change. ${removed} links will be removed (values retained).`),
    h('p', { class: 'muted small' }, 'Check frame rates and transforms carefully when datasets differ. Links also apply to currently hidden validation/model-specific fields.'));
  if (changes.length) {
    const table = h('table', { class: 'grid-table sync-preview' }, h('thead', {}, h('tr', {},
      h('th', {}, 'Destination'), h('th', {}, 'Current value'), h('th', {}, 'Training value'))));
    const rows = h('tbody');
    for (const [target, source] of changes) rows.append(h('tr', {}, h('td', {}, target),
      h('td', {}, JSON.stringify(S.values[target]) ?? 'unset'), h('td', {}, JSON.stringify(S.values[source]) ?? 'unset')));
    table.append(rows);
    body.append(h('div', { class: 'table-wrap' }, table));
  }
  openModal('Preview training sync', body, [
    { label: back ? 'Back' : 'Cancel', onClick: back || closeModal },
    { label: 'Apply links', primary: true, onClick: () => applySyncLinks(links) },
  ], true);
}

function togglePreprocessingSync() {
  if (!syncEnabled()) return;
  const enable = $('linkPreprocess').checked;
  const links = { ...S.syncLinks };
  for (const field of syncFields().filter(preprocessingSyncField)) {
    for (const split of SYNC_TARGETS) {
      const target = syncTarget(field.key, split);
      if (enable) links[target] = field.key;
      else delete links[target];
    }
  }
  // The checkbox reflects committed links, even if the preview is cancelled.
  updateSyncControls();
  if (enable) previewSyncLinks(links);
  else applySyncLinks(links);
}

function syncSettingsDialog(draft = { ...S.syncLinks }) {
  if (!syncEnabled()) return;
  const body = h('div', { class: 'form-stack' },
    h('p', {}, 'Choose which training parameters validation and test should follow. Linked destination fields are read-only until unlinked.'),
    h('p', { class: 'muted small' }, 'Dataset identities, participants, split boundaries, paths and cache/output identifiers are excluded. Links reset when you load/create a config or leave training mode; they are not saved in YAML.'),
    S.values['TEST.USE_LAST_EPOCH'] ? h('p', { class: 'msg info' }, 'Validation is currently unused. Its selected links still update its stored values.') : null);
  const fields = syncFields();
  const table = h('table', { class: 'grid-table sync-options' }, h('thead', {}, h('tr', {},
    h('th', {}, 'Training parameter'), h('th', {}, 'Validation'), h('th', {}, 'Test'))));
  const rows = h('tbody');
  const boxes = [];
  for (const field of fields) {
    const suffix = field.key.slice('TRAIN.DATA.'.length);
    const row = h('tr', {}, h('td', {}, h('strong', {}, field.label), h('div', { class: 'muted small' }, suffix),
      !isActive(field, S.values) ? h('div', { class: 'muted small' }, 'Not used by the current training model/dataset') : null));
    for (const split of SYNC_TARGETS) {
      const target = syncTarget(field.key, split);
      const box = h('input', { type: 'checkbox', 'aria-label': `${field.label}: sync ${split === 'VALID.DATA' ? 'validation' : 'test'} with training`, onchange: () => {
        if (box.checked) draft[target] = field.key;
        else delete draft[target];
      } });
      box.checked = draft[target] === field.key;
      boxes.push({ box, field, target });
      row.append(h('td', {}, box));
    }
    rows.append(row);
  }
  const selectPreset = () => {
    for (const { box, field, target } of boxes) {
      if (!preprocessingSyncField(field)) continue;
      box.checked = true;
      draft[target] = field.key;
    }
  };
  body.append(h('div', { class: 'row' },
    h('button', { type: 'button', class: 'small', onclick: selectPreset }, 'Select preprocessing settings'),
    h('button', { type: 'button', class: 'small', onclick: () => {
      for (const { box, target } of boxes) { box.checked = false; delete draft[target]; }
    } }, 'Clear all')));
  table.append(rows);
  body.append(h('div', { class: 'sync-options-wrap' }, table));
  openModal('Sync settings', body, [
    { label: 'Cancel', onClick: closeModal },
    { label: 'Preview changes', primary: true, onClick: () => previewSyncLinks(draft, () => syncSettingsDialog(draft)) },
  ], true);
}

function offerModelDefaults(model) {
  const profile = S.boot.schema.model_profiles[model];
  const banner = $('modelBanner');
  if (!profile || profile.bigsmall) { banner.classList.add('hidden'); return; }
  banner.replaceChildren(
    h('span', {}, `Apply the usual ${model} preprocessing (${profile.format}, ${profile.types.join(' + ')}, label ${profile.label}, ${profile.chunk} frames, ${profile.size}×${profile.size})?`),
    h('button', { type: 'button', class: 'small primary', onclick: () => { applyModelDefaults(model); banner.classList.add('hidden'); } }, 'Apply'),
    h('button', { type: 'button', class: 'small', onclick: () => banner.classList.add('hidden') }, 'Dismiss'));
  banner.classList.remove('hidden');
}

function applyModelDefaults(model) {
  const p = S.boot.schema.model_profiles[model];
  const v = S.values;
  for (const split of activeSplits(v)) {
    v[split + '.DATA_FORMAT'] = p.format;
    v[split + '.PREPROCESS.DATA_TYPE'] = [...p.types];
    v[split + '.PREPROCESS.RESIZE.H'] = p.size;
    v[split + '.PREPROCESS.RESIZE.W'] = p.size;
    v[split + '.PREPROCESS.LABEL_TYPE'] = p.label;
    v[split + '.PREPROCESS.CHUNK_LENGTH'] = p.chunk;
    if (split === 'TEST.DATA' && v[split + '.DATASET'] === 'vHRM') v[split + '.VHRM.PREDICTION_IS_DIFF'] = p.label === 'DiffNormalized';
  }
  if (p.frame_num) v[p.frame_num] = p.chunk;
  if (p.channels_key) v[p.channels_key] = 3 * p.types.length;
  if (model === 'FactorizePhys' && p.size === 72) v['MODEL.FactorizePhys.TYPE'] = 'Standard';
  for (const source of new Set(Object.values(S.syncLinks))) syncTrainingValue(source, v[source]);
  renderForm();
  scheduleValidate(0);
  toast(`Applied ${model} preprocessing defaults.`, 'ok');
}

function scheduleValidate(delay = 350) {
  clearTimeout(S.validateTimer);
  S.validateTimer = setTimeout(validateNow, delay);
}

async function validateNow() {
  if (!S.values) return;
  const seq = ++S.validateSeq;
  $('validationSummary').textContent = 'Validating…';
  try {
    const report = await api('/api/validate', { method: 'POST', body: { values: S.values, extras: S.extras, backend: S.backend } });
    if (seq !== S.validateSeq) return;
    S.validation = report;
    renderValidation();
  } catch (error) {
    $('validationSummary').textContent = 'Validation failed: ' + error.message;
  }
}

function renderValidation() {
  const report = S.validation;
  document.querySelectorAll('.field .msgs').forEach(el => el.replaceChildren());
  document.querySelectorAll('.field').forEach(el => el.classList.remove('has-error', 'has-warn'));
  document.querySelectorAll('.field .path-info').forEach(el => el.replaceChildren());
  document.querySelectorAll('.derived').forEach(el => el.replaceChildren());
  if (!report) return;
  const list = $('validationList');
  list.replaceChildren();
  const counts = {};
  const groups = [['error', report.errors], ['warn', report.warnings], ['info', report.info]];
  for (const [level, items] of groups) {
    for (const item of items) {
      const fieldEl = item.key ? document.querySelector(`.field[data-key="${CSS.escape(item.key)}"]`) : null;
      if (fieldEl) {
        fieldEl.querySelector('.msgs').append(h('div', { class: 'msg ' + level }, item.message));
        if (level !== 'info') fieldEl.classList.add(level === 'error' ? 'has-error' : 'has-warn');
        const section = S.fieldMap[item.key]?.section;
        if (section && level !== 'info') {
          counts[section] = counts[section] || { error: 0, warn: 0 };
          counts[section][level]++;
        }
      }
      list.append(h('div', {
        class: 'vitem ' + level + (fieldEl ? ' link' : ''),
        onclick: fieldEl ? () => { fieldEl.scrollIntoView({ behavior: 'smooth', block: 'center' }); fieldEl.classList.add('flash'); setTimeout(() => fieldEl.classList.remove('flash'), 1200); } : null,
      }, item.message));
    }
  }
  for (const [key, info] of Object.entries(report.paths || {})) {
    const target = document.querySelector(`.field[data-key="${CSS.escape(key)}"] .path-info`);
    if (!target || !info.host) continue;
    const field = S.fieldMap[key];
    const mustExist = field?.path?.must_exist;
    const state = info.exists ? 'exists' : (mustExist ? 'missing' : 'will be created');
    target.append(h('span', { class: info.exists ? 'ok' : (mustExist ? 'bad' : 'muted') }, `${S.backend === 'docker' ? 'Host: ' : ''}${info.host} — ${state}${info.found ? ' · ' + info.found : ''}`));
  }
  const splitSections = S.boot.schema.splits;
  for (const [prefix, spec] of Object.entries(splitSections)) {
    const derived = report.derived?.[prefix];
    const el = document.querySelector(`.derived[data-section="${CSS.escape(spec.section)}"]`);
    if (!derived || !el) continue;
    el.append(h('div', {}, 'Cache: ', h('code', {}, derived.cache_dir_host || derived.cache_dir), ' · file list ',
      h('span', { class: derived.file_list_exists ? 'ok' : 'muted' }, derived.file_list_exists ? 'exists' : 'not created yet')));
  }
  const outputs = report.derived?.outputs;
  const exp = document.querySelector('.derived[data-section="Experiment"]');
  if (outputs && exp) {
    for (const [key, label] of [['model_dir_host', 'Checkpoints'], ['test_outputs_host', 'Test outputs'], ['outputs_host', 'Outputs']]) {
      if (outputs[key]) exp.append(h('div', {}, `${label}: `, h('code', {}, outputs[key])));
    }
  }
  const cv = report.derived?.cross_validation;
  const cvSection = document.querySelector('.derived[data-section="Cross-validation"]');
  if (cv && cvSection) {
    const host = p => (S.backend === 'docker' && cv.root_host ? p.replace(cv.root, cv.root_host) : p);
    cvSection.append(h('div', {}, 'Each fold: ', h('code', {}, host(cv.folds)), ' · merged results: ', h('code', {}, host(cv.summary))),
      h('div', { class: 'row' }, h('button', { type: 'button', class: 'small', onclick: () => previewFolds() }, 'Preview folds…'),
        h('span', { class: 'muted small' }, 'Lists the participants of every fold (reads the dataset folders, a few seconds).')));
  }
  document.querySelectorAll('#sectionNav a').forEach(a => {
    const c = counts[a.dataset.section];
    const badge = a.querySelector('.count');
    badge.textContent = c ? (c.error ? `${c.error}✕` : '') + (c.warn ? ` ${c.warn}!` : '') : '';
    badge.className = 'count' + (c?.error ? ' error' : c?.warn ? ' warn' : '');
  });
  const summary = $('validationSummary');
  summary.className = 'validation-summary ' + (report.errors.length ? 'error' : report.warnings.length ? 'warn' : 'ok');
  summary.textContent = report.errors.length ? `${report.errors.length} error(s) — fix before launching`
    : report.warnings.length ? `Valid, ${report.warnings.length} warning(s)` : 'Valid';
  $('launchRun').disabled = report.errors.length > 0;
  $('launchRun').textContent = cvEnabled() ? 'Launch cross-validation…' : 'Launch…';
}

async function convertPaths() {
  if (!S.values) return;
  try {
    const result = await api('/api/config/convert-paths', { method: 'POST', body: { values: S.values, target: S.backend } });
    S.values = result.values;
    renderForm();
    scheduleValidate(0);
    toast(result.notes.length ? result.notes.join(' ') : `Paths converted for ${S.backend}.`, result.notes.length ? 'warn' : 'ok', 8000);
  } catch (error) { toast(error.message, 'error'); }
}

async function previewYaml() {
  if (!S.values) return;
  const result = await api('/api/config/yaml', { method: 'POST', body: { values: S.values, extras: S.extras } });
  openModal('YAML preview', h('pre', { class: 'code' }, result.yaml), [{ label: 'Close', onClick: closeModal }], true);
}

function saveConfigDialog() {
  if (!S.values) return;
  const location = h('select', {}, ...Object.entries(S.boot.save_locations).map(([k, v]) => h('option', { value: k }, `${v}/`)));
  location.value = S.backend;
  const suggested = S.source ? S.source.split('/').pop().replace(/\.ya?ml$/, '') : 'my_experiment';
  const name = h('input', { type: 'text', value: 'launcher/' + suggested.replace(/^launcher\//, '') });
  const message = h('div', { class: 'muted small' }, 'Docker configs must be saved in docker/configs to be visible inside the container.');
  const save = async overwrite => {
    try {
      const result = await api('/api/config/save', { method: 'POST', body: { location: location.value, name: name.value, values: S.values, extras: S.extras, overwrite } });
      S.boot.configs = result.configs;
      populateConfigSelect();
      selectConfig(result.path);
      S.source = result.path;
      $('sourceInfo').textContent = `Saved as ${result.path}`;
      closeModal();
      toast(`Saved ${result.path}`, 'ok');
    } catch (error) {
      if (error.status === 409 && confirm(`${error.message} exists. Overwrite?`)) return save(true);
      message.textContent = error.message;
      message.className = 'msg error';
    }
  };
  openModal('Save config', h('div', { class: 'form-stack' }, h('label', {}, 'Folder', location), h('label', {}, 'Name (sub-folders allowed)', name), message),
    [{ label: 'Cancel', onClick: closeModal }, { label: 'Save', primary: true, onClick: () => save(false) }]);
}

async function browseDialog(field) {
  const kind = field.path?.kind === 'file' ? 'file' : 'dir';
  let current = S.values[field.key] || '';
  if (S.backend === 'docker' && current.startsWith('/')) current = containerToHost(current) || '';
  const pathInput = h('input', { type: 'text', value: current });
  const list = h('div', { class: 'browse-list' });
  const note = h('div', { class: 'small muted' });
  const choose = hostPath => {
    let value = hostPath;
    if (S.backend === 'docker') {
      value = hostToContainer(hostPath);
      if (!value) {
        note.className = 'msg error';
        note.textContent = 'This folder is not inside a Docker mount. Mounted host folders: ' +
          S.boot.docker_mounts.map(m => `${m.container} ← ${m.host}`).join('; ') + '. Change them in Settings.';
        return;
      }
    }
    S.values[field.key] = value;
    closeModal();
    renderForm();
    scheduleValidate(0);
  };
  const load = async path => {
    try {
      const data = await api('/api/browse', { params: { path, kind } });
      pathInput.value = data.path;
      list.replaceChildren();
      if (data.parent !== null) list.append(h('div', { class: 'browse-item dir', onclick: () => load(data.parent) }, '⬑ ..'));
      for (const entry of data.entries) {
        list.append(h('div', {
          class: 'browse-item ' + (entry.dir ? 'dir' : 'file'),
          onclick: () => entry.dir ? load(entry.path) : choose(entry.path),
        }, (entry.dir ? '📁 ' : '📄 ') + entry.name));
      }
    } catch (error) { note.textContent = error.message; }
  };
  pathInput.addEventListener('keydown', e => { if (e.key === 'Enter') load(pathInput.value); });
  if (S.backend === 'docker') note.textContent = 'Docker: pick a host folder inside a mount; it is converted to the container path.';
  openModal(`Browse — ${field.label}`, h('div', { class: 'form-stack' }, pathInput, list, note), [
    { label: 'Cancel', onClick: closeModal },
    kind === 'dir' ? { label: 'Use this folder', primary: true, onClick: () => choose(pathInput.value) } : null,
  ].filter(Boolean), true);
  load(current);
}

// ------------------------------------------------------------ cross-validation
// Table of the participants of every fold (plan from /api/cv/plan or a launched group).
function foldTable(folds, onSelect) {
  const list = items => h('span', { title: items.join(', ') }, items.length ? items.join(', ') : '–');
  return h('div', { class: 'table-wrap' }, h('table', { class: 'grid-table fold-table' + (onSelect ? ' clickable' : '') },
    h('tr', {}, ...['Fold', 'Test', 'Validation', 'Training'].map(t => h('th', {}, t))),
    ...folds.map(fold => h('tr', { onclick: onSelect ? () => onSelect(fold) : null },
      h('th', {}, fold.name), h('td', {}, list(fold.test)), h('td', {}, list(fold.valid)),
      h('td', { title: fold.train.join(', ') }, `${fold.train.length} participants`)))));
}

async function previewFolds(refresh = false) {
  if (!S.values) return;
  const body = h('div', {}, h('p', { class: 'muted' }, 'Listing the participants of the training selection…'));
  openModal('Cross-validation folds', body, [{ label: 'Refresh', onClick: () => previewFolds(true) }, { label: 'Close', onClick: closeModal }], true);
  try {
    const { plan } = await api('/api/cv/plan', { method: 'POST', body: { values: S.values, extras: S.extras, backend: S.backend, refresh } });
    const units = Object.values(plan.units);
    body.replaceChildren(
      h('p', {}, `${plan.folds.length} folds over ${units.length} participants`,
        units.length !== plan.pool.length ? ` (${plan.pool.length} participant folders; sessions of a person stay together)` : '', '. ',
        h('span', { class: 'muted' }, 'Each fold is a training run; every participant is tested once.')),
      ...plan.warnings.map(w => h('div', { class: 'msg warn' }, w)),
      foldTable(plan.folds),
      h('p', { class: 'muted small' }, 'Pool: the participants the training data selects: ', plan.pool.join(', ')));
  } catch (error) {
    body.replaceChildren(h('div', { class: 'msg error pre' }, error.message));
  }
}

// ------------------------------------------------------------------ launch
async function launchDialog(initial = {}) {
  if (!S.values) return;
  if (S.validation?.errors.length) { toast('Fix the validation errors first.', 'error'); return; }
  const cv = cvEnabled();
  S.preflight = { answers: initial.answers || {}, secrets: {}, name: initial.name || '', description: S.values.DESCRIPTION || '', result: null, busy: false, cv };
  openModal(cv ? 'Pre-launch checks — cross-validation' : 'Pre-launch checks',
    h('div', { id: 'preflightBody' }, h('p', { class: 'muted' }, cv ? 'Checking environment and planning the folds…' : 'Checking environment…')),
    [{ label: 'Cancel', onClick: closeModal }, { label: 'Re-check', id: 'pfRecheck', onClick: () => refreshPreflight() },
      { label: cv ? 'Launch all folds' : 'Launch', primary: true, id: 'pfLaunch', disabled: true, onClick: doLaunch }], true);
  await refreshPreflight();
}

async function refreshPreflight(autofill = true) {
  const pf = S.preflight;
  if (!pf) return;
  pf.busy = true;
  $('pfLaunch').disabled = true;
  try {
    const result = await api(pf.cv ? '/api/cv/preflight' : '/api/preflight', { method: 'POST', body: { values: S.values, extras: S.extras, backend: S.backend, answers: pf.answers, secrets: pf.secrets } });
    if (!S.preflight) return;
    let filled = false;
    if (autofill) {
      for (const check of result.checks) {
        if (check.status === 'question' && check.default && pf.answers[check.id] === undefined) {
          pf.answers[check.id] = check.default;
          filled = true;
        }
      }
    }
    if (filled) return refreshPreflight(false);
    pf.result = result;
    renderPreflight();
  } catch (error) {
    $('preflightBody').replaceChildren(h('div', { class: 'msg error' }, error.message));
  } finally {
    pf.busy = false;
  }
}

function renderPreflight() {
  const pf = S.preflight;
  const body = $('preflightBody');
  const icons = { ok: '✔', info: 'ℹ', warn: '!', error: '✕', question: '?', answered: '✔' };
  body.replaceChildren();
  for (const check of pf.result.checks) {
    const item = h('div', { class: 'check ' + check.status },
      h('div', { class: 'check-head' }, h('span', { class: 'icon' }, icons[check.status] || '•'), h('strong', {}, check.title)),
      h('div', { class: 'check-msg' }, check.message));
    if (check.details?.length) item.append(h('ul', { class: 'details' }, ...check.details.slice(0, 12).map(d => h('li', {}, d))));
    if (check.options) {
      const options = h('div', { class: 'options' });
      for (const option of check.options) {
        const radio = h('input', { type: 'radio', name: 'pf_' + check.id, value: option.value });
        radio.checked = pf.answers[check.id] === option.value;
        radio.addEventListener('change', () => { pf.answers[check.id] = option.value; refreshPreflight(false); });
        options.append(h('label', { class: 'option' }, radio, h('span', {}, option.label, option.description ? h('small', {}, option.description) : null)));
      }
      item.append(options);
      if (check.secret && pf.answers[check.id] === check.secret.when) {
        const input = h('input', { type: 'password', autocomplete: 'off', placeholder: check.secret.label, value: pf.secrets[check.secret.name] || '' });
        input.addEventListener('input', () => { pf.secrets[check.secret.name] = input.value; });
        item.append(h('div', { class: 'row' }, input, h('button', { type: 'button', class: 'small', onclick: () => refreshPreflight(false) }, 'Verify')));
      }
    }
    body.append(item);
  }
  if (pf.result.cv?.folds) {
    body.append(h('details', { class: 'check-folds', open: pf.result.cv.folds.length <= 10 },
      h('summary', {}, `Participants of the ${pf.result.cv.folds.length} folds`), foldTable(pf.result.cv.folds)));
  }
  const name = h('input', { type: 'text', placeholder: pf.cv ? 'Optional name (each fold job is "<name> · fold N")' : 'Optional job name', value: pf.name });
  name.addEventListener('input', () => { pf.name = name.value; });
  body.append(h('label', { class: 'job-name' }, 'Job name', name));
  const description = h('textarea', { rows: '3', placeholder: 'Optional description: what this run tests or changes' });
  description.value = pf.description;
  description.addEventListener('input', () => { pf.description = description.value; });
  // Keep the editor in sync so a later "Save config" or relaunch keeps the text.
  description.addEventListener('change', () => { S.values.DESCRIPTION = description.value; renderFieldInPlace('DESCRIPTION'); });
  body.append(h('label', { class: 'job-name' }, 'Description', description));
  $('pfLaunch').disabled = !pf.result.can_launch;
}

async function doLaunch() {
  const pf = S.preflight;
  if (!pf?.result?.can_launch) return;
  $('pfLaunch').disabled = true;
  try {
    const body = { values: S.values, extras: S.extras, backend: S.backend, answers: pf.answers, secrets: pf.secrets, name: pf.name, description: pf.description, source: S.source, resume_from: S.resumeFrom };
    if (pf.cv) {
      const { group } = await api('/api/cv/groups', { method: 'POST', body });
      closeModal();
      selectGroup(group.id);
      toast(`Cross-validation "${group.name}": ${group.folds.length} fold jobs queued.`, 'ok');
      showTab('jobs');
      return;
    }
    const result = await api('/api/jobs', { method: 'POST', body });
    closeModal();
    S.selectedJob = result.job.id;
    S.selectedGroup = null;
    S.log = null;
    toast(`Job "${result.job.name}" ${result.job.status}.`, 'ok');
    showTab('jobs');
  } catch (error) {
    toast(error.message, 'error', 10000);
    $('pfLaunch').disabled = false;
  }
}

// -------------------------------------------------------------------- jobs
const PHASES = { starting: 'Starting', preprocessing: 'Preprocessing', training: 'Training', validating: 'Validating', testing: 'Testing', unsupervised: 'Unsupervised', succeeded: 'Succeeded', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted', pausing: 'Finishing epoch before pause', paused: 'Paused (GPU released)' };

function jobFraction(job) {
  const p = job.progress || {};
  if (job.status === 'succeeded') return 1;
  if (p.phase === 'training' && job.epochs_total && p.epoch !== null && p.epoch !== undefined) {
    const inner = p.bar && /^train epoch/i.test(p.bar.desc) && p.bar.total ? p.bar.n / p.bar.total : 0;
    return Math.min(1, (p.epoch + inner) / job.epochs_total);
  }
  if (p.bar && p.bar.total) return p.bar.n / p.bar.total;
  return null;
}

function elapsed(job) {
  if (!job.started) return '';
  const end = job.finished ? new Date(job.finished) : new Date();
  return fmtDuration((end - new Date(job.started)) / 1000);
}

async function pollJobs() {
  try {
    const data = await api('/api/jobs');
    S.jobs = data.jobs;
    S.groups = data.groups || [];
    S.gpus = data.gpus;
  } catch (error) { return; }
  const active = S.jobs.filter(j => ['queued', 'starting', 'running', 'pausing', 'stopping'].includes(j.status)).length;
  $('jobsBadge').textContent = active;
  $('jobsBadge').classList.toggle('hidden', !active);
  $('gpuStrip').replaceChildren(...S.gpus.map(g => h('span', { class: 'gpu', title: g.name },
    `GPU${g.index} ${g.utilization}% · ${(g.memory_used / 1024).toFixed(1)}/${(g.memory_total / 1024).toFixed(0)} GB`)));
  if ($('tab-jobs').classList.contains('active')) {
    renderJobList();
    updateJobDetail();
  }
}

const ACTIVE_STATUSES = ['queued', 'starting', 'running', 'pausing', 'stopping'];

function selectJob(id) {
  S.selectedJob = id;
  S.selectedGroup = null;
  S.log = null;
  renderJobList();
  updateJobDetail(true);
}

function selectGroup(id) {
  S.selectedGroup = id;
  S.selectedJob = null;
  S.log = null;
  if ($('tab-jobs').classList.contains('active')) { renderJobList(); updateJobDetail(true); }
}

// Folds done (with the running fold's own progress), average duration of finished folds and the ETA.
function groupProgress(group) {
  const done = group.folds.filter(f => f.status === 'succeeded').length;
  const failed = group.folds.filter(f => ['failed', 'interrupted', 'cancelled', 'removed'].includes(f.status)).length;
  const running = group.folds.find(f => ['starting', 'running', 'pausing', 'stopping'].includes(f.status));
  const runningJob = running && S.jobs.find(j => j.id === running.job);
  const current = runningJob ? (jobFraction(runningJob) || 0) : 0;
  const durations = group.folds.filter(f => f.status === 'succeeded' && f.started && f.finished)
    .map(f => (new Date(f.finished) - new Date(f.started)) / 1000);
  const average = durations.length ? durations.reduce((a, b) => a + b, 0) / durations.length : null;
  const remaining = group.folds.length - done - failed;
  let eta = null;
  if (average !== null && group.active) {
    const spent = runningJob?.started ? (Date.now() - new Date(runningJob.started)) / 1000 : 0;
    eta = Math.max(0, remaining * average - Math.min(spent, average));
  }
  return { done, failed, running, fraction: (done + current) / group.folds.length, average, eta };
}

function jobItem(job, nested = false) {
  const fraction = jobFraction(job);
  const phase = PHASES[job.progress?.phase] || job.status;
  return h('button', { type: 'button', class: 'item job ' + job.status + (nested ? ' fold' : '') + (job.id === S.selectedJob ? ' active' : ''), onclick: () => selectJob(job.id) },
    h('div', { class: 'item-title' }, h('span', { class: 'dot' }), nested ? `Fold ${job.group.fold}` : job.name,
      nested && job.resume_from ? h('span', { class: 'muted small' }, ' (resumed)') : null),
    h('div', { class: 'item-sub' }, `${nested ? '' : job.backend + ' · '}${job.status === 'running' ? phase : job.status} · ${elapsed(job)}`),
    !nested && job.description ? h('div', { class: 'item-desc', title: job.description }, job.description) : null,
    fraction !== null && job.status === 'running' ? h('div', { class: 'progress' }, h('div', { style: `width:${(fraction * 100).toFixed(1)}%` })) : null);
}

function groupItem(group) {
  const progress = groupProgress(group);
  S.collapsedGroups = S.collapsedGroups || {};
  const collapsed = S.collapsedGroups[group.id] ?? !group.active;
  const toggle = h('span', { class: 'group-toggle', role: 'button', title: collapsed ? 'Show the folds' : 'Hide the folds',
    onclick: e => { e.stopPropagation(); S.collapsedGroups[group.id] = !collapsed; renderJobList(); } }, collapsed ? '▸' : '▾');
  const header = h('button', { type: 'button', class: 'item job group ' + group.status + (group.id === S.selectedGroup ? ' active' : ''), onclick: () => selectGroup(group.id) },
    h('div', { class: 'item-title' }, toggle, h('span', { class: 'dot' }), group.name),
    h('div', { class: 'item-sub' }, `cross-validation · ${progress.done}/${group.folds.length} folds done`,
      progress.failed ? ` · ${progress.failed} not completed` : '', ` · ${group.status}`, progress.eta !== null ? ` · ~${fmtDuration(progress.eta)} left` : ''),
    group.description ? h('div', { class: 'item-desc', title: group.description }, group.description) : null,
    group.active ? h('div', { class: 'progress' }, h('div', { style: `width:${(progress.fraction * 100).toFixed(1)}%` })) : null);
  const attempts = S.jobs.filter(j => j.group?.id === group.id).sort((a, b) => a.group.index - b.group.index || a.created.localeCompare(b.created));
  return h('div', { class: 'job-group' }, header, collapsed ? null : h('div', { class: 'group-folds' }, ...attempts.map(job => jobItem(job, true))));
}

function renderJobList() {
  const filter = $('jobFilter').value;
  const list = $('jobList');
  const keep = active => filter === 'all' || (filter === 'active') === active;
  const groups = (S.groups || []).filter(g => keep(g.active));
  const known = new Set((S.groups || []).map(g => g.id));
  const jobs = S.jobs.filter(j => !known.has(j.group?.id) && keep(ACTIVE_STATUSES.includes(j.status)));
  const entries = [...groups.map(g => ({ created: g.created, node: () => groupItem(g) })),
    ...jobs.map(j => ({ created: j.created, node: () => jobItem(j) }))].sort((a, b) => b.created.localeCompare(a.created));
  if (!S.selectedJob && !S.selectedGroup) {
    if (groups.length && (!jobs.length || groups[0].created >= jobs[0].created)) S.selectedGroup = groups[0].id;
    else if (jobs.length) S.selectedJob = jobs[0].id;
  }
  list.replaceChildren(...entries.map(e => e.node()));
  if (!entries.length) list.append(h('p', { class: 'muted pad' }, 'No jobs yet. Launch one from "New run".'));
}

function updateJobDetail(force = false) {
  if (S.selectedGroup) { updateGroupDetail(force); return; }
  const job = S.jobs.find(j => j.id === S.selectedJob);
  const panel = $('jobDetail');
  if (!job) { panel.replaceChildren(h('p', { class: 'muted' }, 'Select a job.')); return; }
  if (force || panel.dataset.job !== job.id) {
    panel.dataset.job = job.id;
    panel.replaceChildren(
      h('div', { id: 'jdHead' }), h('div', { id: 'jdDescription', class: 'card' }), h('div', { id: 'jdProgress', class: 'card' }), h('div', { id: 'jdError' }),
      h('div', { class: 'two-col' }, h('div', { class: 'card' }, h('h4', {}, 'Loss per epoch'), h('canvas', { id: 'jdLoss', class: 'chart' })),
        h('div', { class: 'card' }, h('h4', {}, 'Results'), h('div', { id: 'jdMetrics' }))),
      h('div', { class: 'card' }, h('div', { class: 'log-head' }, h('h4', {}, 'Console output'),
        h('label', { class: 'check' }, h('input', { type: 'checkbox', id: 'logFollow', checked: true }), 'Follow')),
        h('pre', { id: 'jobLog', class: 'log' })));
    S.log = { jobId: job.id, next: null, lines: [], partial: '', done: false };
    S.editingDescription = null;
    pollLog();
  }
  if (S.editingDescription !== job.id) renderJobDescription(job);
  const p = job.progress || {};
  const actions = [];
  if (job.can_pause) actions.push(h('button', { type: 'button', onclick: () => pauseJob(job) }, 'Pause after epoch'));
  if (job.can_resume) actions.push(h('button', { type: 'button', class: 'primary', onclick: () => resumeJob(job) }, job.recovery?.training_complete ? 'Retry plots / test' : 'Resume training'));
  if (['queued', 'running', 'starting', 'pausing'].includes(job.status)) actions.push(h('button', { type: 'button', class: 'danger', onclick: () => cancelJob(job) }, job.status === 'queued' ? 'Cancel' : 'Stop now'));
  actions.push(h('button', { type: 'button', onclick: () => viewJobConfig(job) }, 'View config'));
  actions.push(h('button', { type: 'button', onclick: () => editJobConfig(job) }, 'Open in editor'));
  if ((job.outputs || {}).log_path_host) actions.push(h('button', { type: 'button', onclick: () => openResultsForJob(job) }, 'Results'));
  if (!['queued', 'running', 'starting', 'pausing', 'stopping'].includes(job.status)) actions.push(h('button', { type: 'button', onclick: () => removeJob(job) }, 'Remove'));
  $('jdHead').replaceChildren(
    h('div', { class: 'detail-head' }, h('div', {}, h('h2', {}, job.name),
      h('div', { class: 'muted' }, `${job.mode} · ${job.model || ''} · ${[job.datasets?.train, job.datasets?.test].filter(Boolean).join(' → ')} · ${job.backend}${job.gpu !== null && job.gpu !== undefined ? ' · GPU ' + job.gpu : ''} · created ${fmtDate(job.created)}`)),
      h('div', { class: 'actions' }, ...actions)));
  const fraction = jobFraction(job);
  const bar = p.bar;
  const rows = [
    ['Status', h('span', { class: 'status ' + job.status }, job.status)],
    ['Phase', PHASES[p.phase] || p.phase || '–'],
    job.epochs_total ? ['Epoch', p.epoch !== null && p.epoch !== undefined ? `${p.epoch + 1} / ${job.epochs_total}` : `– / ${job.epochs_total}`] : null,
    bar ? ['Current step', `${bar.desc || 'progress'}: ${bar.n}/${bar.total}${bar.eta ? ' · ETA ' + bar.eta : ''}`] : null,
    p.best_epoch !== null && p.best_epoch !== undefined ? ['Best epoch', String(p.best_epoch)] : null,
    ['Elapsed', elapsed(job) || '–'],
    job.status === 'pausing' ? ['Pause', 'Waiting for the current epoch and validation to finish; Stop now discards the unfinished epoch.'] : null,
    job.recovery ? ['Recovery', `${job.recovery.next_epoch} / ${job.recovery.epochs_total} epochs committed${job.recovery.training_complete ? ' — training complete' : ''}`] : null,
    job.resume_from ? ['Resumed from', job.resume_from] : null,
    job.group ? ['Cross-validation', h('a', { href: '#', onclick: e => { e.preventDefault(); selectGroup(job.group.id); } },
      `${job.group.name} — fold ${job.group.fold} (${job.group.index + 1} of ${job.group.count})`)] : null,
    job.wandb?.enabled ? ['W&B', p.wandb_url ? h('a', { href: p.wandb_url, target: '_blank', rel: 'noopener' }, p.wandb_url) : job.wandb.mode] : null,
    (job.outputs || {}).log_path_host ? ['Output folder', h('code', {}, job.outputs.log_path_host)] : null,
    p.last_line ? ['Last line', h('code', { class: 'clip' }, p.last_line)] : null,
  ].filter(Boolean);
  $('jdProgress').replaceChildren(...[
    fraction !== null ? h('div', { class: 'progress big' }, h('div', { style: `width:${(fraction * 100).toFixed(1)}%` }), h('span', {}, `${(fraction * 100).toFixed(0)}%`)) : null,
    h('table', { class: 'kv' }, ...rows.map(([k, v]) => h('tr', {}, h('th', {}, k), h('td', {}, v))))].filter(Boolean));
  const err = $('jdError');
  err.replaceChildren();
  if (job.error || p.hints?.length) {
    err.append(h('div', { class: 'card error-card' },
      job.error ? h('div', { class: 'msg error' }, job.error) : null,
      ...(p.hints || []).map(hint => h('div', { class: 'msg warn' }, hint)),
      p.traceback?.length ? h('details', {}, h('summary', {}, 'Traceback'), h('pre', { class: 'code' }, p.traceback.join('\n'))) : null));
  }
  const epochs = [...new Set([...Object.keys(p.train_loss || {}), ...Object.keys(p.valid_loss || {})])].map(Number).sort((a, b) => a - b);
  Charts.draw($('jdLoss'), {
    height: 200, xLabel: 'epoch', yLabel: 'loss', empty: 'No epochs yet',
    series: [
      { label: 'train (sampled)', x: epochs, y: epochs.map(e => p.train_loss?.[e] ?? null), markers: true },
      { label: 'validation', x: epochs, y: epochs.map(e => p.valid_loss?.[e] ?? null), markers: true },
    ],
    vlines: p.best_epoch !== null && p.best_epoch !== undefined ? [{ x: p.best_epoch, dash: [4, 3], color: '#30a46c' }] : [],
  });
  const metrics = p.metrics || {};
  const groups = Object.keys(metrics);
  const summary = job.summary ? jsonTable(job.summary) : null;
  $('jdMetrics').replaceChildren(...[groups.length ? h('table', { class: 'grid-table' },
    h('tr', {}, h('th', {}, ''), ...['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'MACC'].map(m => h('th', {}, m))),
    ...groups.map(g => h('tr', {}, h('th', {}, g), ...['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'MACC'].map(m => {
      const v = metrics[g][m];
      return h('td', {}, v ? `${num(v.value, 3)} ± ${num(v.se, 3)}` : '–');
    })))) : (summary ? null : h('p', { class: 'muted' }, 'Metrics appear after the test phase.')), summary].filter(Boolean));
}

function renderJobDescription(job) {
  const card = $('jdDescription');
  const head = (...buttons) => h('div', { class: 'log-head' }, h('h4', {}, 'Description'), h('div', { class: 'actions' }, ...buttons));
  if (S.editingDescription !== job.id) {
    card.replaceChildren(head(h('button', { type: 'button', class: 'small', onclick: () => { S.editingDescription = job.id; renderJobDescription(job); } }, 'Edit')),
      job.description ? h('div', { class: 'description' }, job.description) : h('p', { class: 'muted' }, 'No description. Click Edit to note what this run tests or changes.'));
    return;
  }
  const input = h('textarea', { rows: '4', placeholder: 'What this run tests or changes, observations…' });
  input.value = job.description || '';
  const stop = () => { S.editingDescription = null; renderJobDescription(S.jobs.find(j => j.id === job.id) || job); };
  const save = async () => {
    try {
      const result = await api(`/api/jobs/${job.id}/description`, { method: 'POST', body: { description: input.value } });
      const index = S.jobs.findIndex(j => j.id === job.id);
      if (index >= 0) S.jobs[index] = { ...S.jobs[index], description: result.job.description };
      S.editingDescription = null;
      renderJobList();
      renderJobDescription(S.jobs[index] || result.job);
    } catch (error) { toast(error.message, 'error'); }
  };
  card.replaceChildren(head(h('button', { type: 'button', class: 'small', onclick: stop }, 'Cancel'),
    h('button', { type: 'button', class: 'small primary', onclick: save }, 'Save')), input);
  input.focus();
}

function lastSegment(line) {
  const parts = line.split('\r').filter(s => s.trim() !== '');
  return parts.length ? parts[parts.length - 1] : '';
}

async function pollLog() {
  const log = S.log;
  if (!log || log.busy || !$('tab-jobs').classList.contains('active')) return;
  const job = S.jobs.find(j => j.id === log.jobId);
  log.busy = true;
  try {
    const params = log.next === null ? {} : { offset: log.next };
    const data = await api(`/api/jobs/${log.jobId}/log`, { params });
    if (S.log !== log) return;
    log.next = data.next;
    if (data.text) {
      const text = log.partial + data.text;
      const parts = text.split('\n');
      log.partial = parts.pop();
      for (const part of parts) log.lines.push(lastSegment(part));
      if (log.lines.length > 4000) log.lines.splice(0, log.lines.length - 4000);
      const pre = $('jobLog');
      if (pre) {
        pre.textContent = log.lines.join('\n') + '\n' + lastSegment(log.partial);
        if ($('logFollow')?.checked) pre.scrollTop = pre.scrollHeight;
      }
    }
    if (job && !['queued', 'starting', 'running', 'pausing', 'stopping'].includes(job.status) && data.next >= data.size) log.done = true;
  } catch (error) {
    /* job removed or log not yet available */
  } finally {
    log.busy = false;
  }
}

async function cancelJob(job) {
  if (!confirm(`Stop "${job.name}" immediately? Any unfinished epoch is discarded. A full-state checkpoint, if available, can resume from the last completed epoch.`)) return;
  try { await api(`/api/jobs/${job.id}/cancel`, { method: 'POST', body: {} }); pollJobs(); } catch (error) { toast(error.message, 'error'); }
}

async function pauseJob(job) {
  if (!confirm(`Pause "${job.name}" after the current epoch and validation? The checkpoint is saved and the GPU is released. This can take the rest of the epoch.`)) return;
  try { await api(`/api/jobs/${job.id}/pause`, { method: 'POST', body: {} }); pollJobs(); } catch (error) { toast(error.message, 'error'); }
}

async function resumeJob(job) {
  try {
    const data = await api(`/api/jobs/${job.id}/resume`, { method: 'POST', body: {} });
    setValues(data.values, data.extras, data.backend, `Resume "${job.name}" from its last committed epoch`, data.path);
    S.resumeFrom = data.resume_from;
    S.validation = null;
    showTab('editor');
    await launchDialog(data);
  } catch (error) { toast(error.message, 'error'); }
}

async function removeJob(job) {
  if (!confirm(`Remove "${job.name}" from the list? Its log and config snapshot are deleted; run outputs are kept.`)) return;
  try { await api(`/api/jobs/${job.id}/remove`, { method: 'POST', body: {} }); S.selectedJob = null; pollJobs(); } catch (error) { toast(error.message, 'error'); }
}

async function viewJobConfig(job) {
  const response = await fetch(`/api/jobs/${job.id}/config`, { headers: { 'X-Launcher-Token': TOKEN } });
  openModal(`Config — ${job.name}`, h('pre', { class: 'code' }, await response.text()), [{ label: 'Close', onClick: closeModal }], true);
}

async function editJobConfig(job) {
  try {
    const data = await api('/api/config', { params: { path: `launcher_data/jobs/${job.id}/config.yaml` } });
    if (job.description !== undefined) data.values.DESCRIPTION = job.description;
    setValues(data.values, data.extras, job.backend, `Config of job "${job.name}" (edit and launch again)`, job.source_config);
    showTab('editor');
  } catch (error) { toast(error.message, 'error'); }
}

function openResultsForJob(job) {
  S.results.pendingJob = job.id;
  showTab('results');
}

// ------------------------------------------------------ cross-validation jobs
const SUMMARY_COLUMNS = [['windows', 'Windows', 0], ['mae_bpm', 'MAE', 2], ['rmse_bpm', 'RMSE', 2], ['mape_percent', 'MAPE %', 1],
  ['pearson', 'Pearson', 3], ['mean_snr_db', 'SNR dB', 1], ['mae_vs_label_bpm', 'MAE vs label', 2]];

function summaryTable(rows, keyColumn) {
  if (!rows?.length) return h('p', { class: 'muted' }, 'No rows.');
  const columns = SUMMARY_COLUMNS.filter(([key]) => key in rows[0]);
  return h('div', { class: 'table-wrap' }, h('table', { class: 'grid-table' },
    h('tr', {}, keyColumn ? h('th', {}, keyColumn) : null, ...columns.map(([, label]) => h('th', {}, label))),
    ...rows.map(row => h('tr', {}, keyColumn ? h('th', {}, String(row[keyColumn.toLowerCase()] ?? '')) : null,
      ...columns.map(([key, , digits]) => h('td', {}, num(row[key], digits)))))));
}

async function fetchGroupDetail(id) {
  try {
    const { group } = await api(`/api/cv/groups/${id}`);
    S.groupDetail = { id, group, stamp: group.summary?.modified ?? null, merge: group.merge?.status };
    if (S.selectedGroup === id) updateGroupDetail();
  } catch (error) { toast(error.message, 'error'); }
}

function updateGroupDetail(force = false) {
  const group = (S.groups || []).find(g => g.id === S.selectedGroup);
  const panel = $('jobDetail');
  if (!group) { panel.replaceChildren(h('p', { class: 'muted' }, 'This cross-validation was removed.')); return; }
  const detail = S.groupDetail?.id === group.id ? S.groupDetail : null;
  // The summary tables come from the full record; reload it when a merge finished or the summary changed.
  if (force || !detail || detail.stamp !== (group.summary?.modified ?? null) || detail.merge !== group.merge?.status) {
    if (!S.groupFetching) { S.groupFetching = true; fetchGroupDetail(group.id).finally(() => { S.groupFetching = false; }); }
  }
  if (force || panel.dataset.job !== 'group:' + group.id) {
    panel.dataset.job = 'group:' + group.id;
    panel.replaceChildren(h('div', { id: 'gdHead' }), h('div', { id: 'gdProgress', class: 'card' }), h('div', { id: 'gdFolds', class: 'card' }), h('div', { id: 'gdSummary', class: 'card' }));
  }
  const progress = groupProgress(group);
  const actions = [];
  if (group.active) actions.push(h('button', { type: 'button', class: 'danger', onclick: () => groupAction(group, 'cancel') }, 'Stop remaining folds'));
  if (progress.done) actions.push(h('button', { type: 'button', onclick: () => groupAction(group, 'merge') }, 'Merge results now'));
  actions.push(h('button', { type: 'button', onclick: () => editGroupConfig(group) }, 'Open in editor'));
  if (group.summary) actions.push(h('button', { type: 'button', onclick: () => { S.results.pendingPath = group.root_host; showTab('results'); } }, 'Results'));
  if (!group.active) actions.push(h('button', { type: 'button', onclick: () => groupAction(group, 'remove') }, 'Remove'));
  const scheme = S.fieldMap['CROSS_VALIDATION.SCHEME']?.option_labels?.[group.scheme] || group.scheme;
  $('gdHead').replaceChildren(h('div', { class: 'detail-head' }, h('div', {}, h('h2', {}, group.name),
    h('div', { class: 'muted' }, `Cross-validation · ${scheme} · ${group.folds.length} folds · ${group.pool.length} participants · ${group.backend} · created ${fmtDate(group.created)}`)),
  h('div', { class: 'actions' }, ...actions)));

  const merge = group.merge || {};
  const mergeText = { none: 'after the first finished fold', running: 'merging…', succeeded: `merged ${fmtDate(merge.finished)}`, failed: `failed ${fmtDate(merge.finished)}` }[merge.status] || merge.status;
  const rows = [
    ['Status', h('span', { class: 'status ' + group.status }, group.status)],
    ['Folds', `${progress.done} of ${group.folds.length} done${progress.failed ? ` · ${progress.failed} not completed` : ''}${progress.running ? ` · running fold ${progress.running.name}` : ''}`],
    progress.average !== null ? ['Fold duration', `~${fmtDuration(progress.average)} on average`] : null,
    progress.eta !== null ? ['Time left', `~${fmtDuration(progress.eta)} (estimate from the finished folds)`] : null,
    ['Output folder', h('code', {}, group.root_host || group.root)],
    ['Merged summary', mergeText],
    group.description ? ['Description', h('div', { class: 'description' }, group.description)] : null,
  ].filter(Boolean);
  $('gdProgress').replaceChildren(...[
    group.active ? h('div', { class: 'progress big' }, h('div', { style: `width:${(progress.fraction * 100).toFixed(1)}%` }), h('span', {}, `${(progress.fraction * 100).toFixed(0)}%`)) : null,
    h('table', { class: 'kv' }, ...rows.map(([k, v]) => h('tr', {}, h('th', {}, k), h('td', {}, v)))),
    ...(group.warnings || []).map(w => h('div', { class: 'msg warn' }, w)),
    merge.status === 'failed' && merge.output ? h('details', {}, h('summary', {}, 'Merge output'), h('pre', { class: 'code' }, merge.output)) : null].filter(Boolean));

  const cell = items => h('td', { title: items.join(', ') }, items.join(', ') || '–');
  $('gdFolds').replaceChildren(h('h4', {}, 'Folds'), h('div', { class: 'table-wrap' }, h('table', { class: 'grid-table clickable' },
    h('tr', {}, ...['Fold', 'Test', 'Validation', 'Training', 'Status', 'Epoch', 'MAE', 'Duration'].map(t => h('th', {}, t))),
    ...group.folds.map(fold => h('tr', { onclick: fold.job ? () => selectJob(fold.job) : null, title: fold.job ? 'Open this fold job' : '' },
      h('th', {}, fold.name), cell(fold.test), cell(fold.valid), h('td', { title: fold.train.join(', ') }, String(fold.train.length)),
      h('td', {}, h('span', { class: 'status ' + fold.status }, fold.status), fold.attempts > 1 ? h('span', { class: 'muted small' }, ` (${fold.attempts} attempts)`) : null),
      h('td', {}, fold.epochs_total ? `${fold.epoch !== null && fold.epoch !== undefined ? fold.epoch + 1 : '–'} / ${fold.epochs_total}` : '–'),
      h('td', {}, num(fold.summary?.mae_bpm)),
      h('td', {}, fold.started ? fmtDuration(((fold.finished ? new Date(fold.finished) : new Date()) - new Date(fold.started)) / 1000) : '–'))))));

  const summary = detail?.group.summary;
  const parts = [h('h4', {}, 'Merged results (all folds)')];
  if (!summary) parts.push(h('p', { class: 'muted' }, 'The fold results are merged into cv_summary/ each time a fold finishes.'));
  else {
    parts.push(summaryTable([summary.overall]));
    for (const [name, key] of [['by_fold', 'Fold'], ['by_participant', 'Participant'], ['by_movement', 'Movement']]) {
      if (summary.tables?.[name]?.length) parts.push(h('h4', {}, `By ${key.toLowerCase()}`), summaryTable(summary.tables[name], key));
    }
  }
  $('gdSummary').replaceChildren(...parts);
}

async function groupAction(group, action) {
  const prompts = {
    cancel: `Stop "${group.name}"? Queued folds are cancelled and the running fold stops immediately (its unfinished epoch is discarded).`,
    remove: `Remove "${group.name}" and its ${group.folds.length} fold jobs from the list? Logs and config snapshots are deleted; run outputs and the merged summary are kept.`,
  };
  if (prompts[action] && !confirm(prompts[action])) return;
  try {
    await api(`/api/cv/groups/${group.id}/${action}`, { method: 'POST', body: {} });
    if (action === 'remove') S.selectedGroup = null;
    if (action === 'merge') toast('Merging the fold results…', 'ok');
    await pollJobs();
  } catch (error) { toast(error.message, 'error'); }
}

async function editGroupConfig(group) {
  try {
    const data = await api('/api/config', { params: { path: `launcher_data/cv/${group.id}/config.yaml` } });
    setValues(data.values, data.extras, group.backend, `Cross-validation config of "${group.name}" (edit and launch again)`, group.source_config);
    showTab('editor');
  } catch (error) { toast(error.message, 'error'); }
}

// ------------------------------------------------------- preprocessed data
async function loadCacheSets() {
  try {
    const data = await api('/api/cache');
    S.cache.sets = data.sets;
    $('cacheRoots').textContent = data.roots.join('; ') || 'none (add folders in Settings)';
    const select = $('cacheList');
    select.replaceChildren(h('option', { value: '' }, data.sets.length ? '— choose a file list —' : 'No caches found'));
    for (const set of data.sets) {
      const og = h('optgroup', { label: set.name });
      for (const list of set.lists) og.append(h('option', { value: list.path }, list.name));
      select.append(og);
    }
  } catch (error) { toast(error.message, 'error'); }
}

async function loadSamples(listPath) {
  stopPlayback();
  if (!listPath) return;
  $('sampleCount').textContent = 'Loading…';
  try {
    const data = await api('/api/cache/samples', { params: { list: listPath } });
    S.cache.samples = data.samples;
    $('sampleCount').textContent = `${data.samples.length} samples${data.missing ? ` · ${data.missing} entries not found on disk` : ''}`;
    renderSamples();
  } catch (error) { $('sampleCount').textContent = error.message; }
}

function renderSamples() {
  const query = $('sampleFilter').value.trim().toLowerCase();
  const visible = S.cache.samples.filter(s => !query || s.name.toLowerCase().includes(query));
  const shown = visible.slice(0, 1500);
  $('sampleList').replaceChildren(...shown.map(sample => h('button', {
    type: 'button', class: 'item' + (S.cache.sample?.input === sample.input ? ' active' : ''),
    onclick: () => selectSample(sample),
  }, h('div', { class: 'item-title' }, sample.name.split('/').pop()), h('div', { class: 'item-sub' }, sample.name))));
  if (visible.length > shown.length) $('sampleList').append(h('p', { class: 'muted pad' }, `${visible.length - shown.length} more — refine the filter.`));
}

async function selectSample(sample) {
  stopPlayback();
  S.cache.sample = sample;
  renderSamples();
  $('cacheEmpty').classList.add('hidden');
  $('cacheViewer').classList.remove('hidden');
  try {
    const info = await api('/api/cache/sample', { params: { path: sample.input } });
    S.cache.info = info;
    const rows = [['File', sample.name], ['Shape', info.shape.join(' × ')], ['dtype', info.dtype], ['Size', fmtBytes(info.size_bytes)]];
    if (info.label) rows.push(['Label', `${info.label.shape.join(' × ')} ${info.label.dtype}`]);
    else rows.push(['Label', 'missing']);
    const table = h('table', { class: 'kv' }, ...rows.map(([k, v]) => h('tr', {}, h('th', {}, k), h('td', {}, v))));
    const stats = info.groups ? h('table', { class: 'grid-table' }, h('tr', {}, ...['Channels', 'min', 'max', 'mean', 'std'].map(t => h('th', {}, t))),
      ...info.groups.map(g => h('tr', {}, h('td', {}, `${g.start}–${g.start + g.channels - 1}`), h('td', {}, num(g.min, 3)), h('td', {}, num(g.max, 3)), h('td', {}, num(g.mean, 3)), h('td', {}, num(g.std, 3))))) : null;
    $('sampleInfo').replaceChildren(table, stats || '');
    const select = $('channelGroup');
    select.replaceChildren(...(info.groups || []).map((g, i) => h('option', { value: i }, `${g.start}–${g.start + g.channels - 1}`)));
    await loadFrames();
    await loadLabel();
  } catch (error) { toast(error.message, 'error'); }
}

async function loadFrames() {
  const sample = S.cache.sample;
  if (!sample) return;
  const params = new URLSearchParams({ path: sample.input, group: $('channelGroup').value || '0', stride: '1' });
  const response = await fetch('/api/cache/frames?' + params, { headers: { 'X-Launcher-Token': TOKEN } });
  if (!response.ok) { const p = await response.json().catch(() => ({})); toast(p.error || 'Could not load frames', 'error'); return; }
  S.cache.meta = JSON.parse(response.headers.get('X-Frames-Meta'));
  S.cache.frames = new Uint8ClampedArray(await response.arrayBuffer());
  const slider = $('frameSlider');
  slider.max = Math.max(0, S.cache.meta.frames - 1);
  S.cache.frame = Math.min(S.cache.frame, S.cache.meta.frames - 1);
  slider.value = S.cache.frame;
  drawFrame();
}

function drawFrame() {
  const { meta, frames } = S.cache;
  if (!meta || !frames) return;
  const index = S.cache.frame;
  const size = meta.width * meta.height * 3;
  const rgba = new Uint8ClampedArray(meta.width * meta.height * 4);
  for (let i = 0, j = index * size; i < meta.width * meta.height; i++, j += 3) {
    rgba[i * 4] = frames[j]; rgba[i * 4 + 1] = frames[j + 1]; rgba[i * 4 + 2] = frames[j + 2]; rgba[i * 4 + 3] = 255;
  }
  const off = drawFrame.off || (drawFrame.off = document.createElement('canvas'));
  off.width = meta.width; off.height = meta.height;
  off.getContext('2d').putImageData(new ImageData(rgba, meta.width, meta.height), 0, 0);
  const canvas = $('frameCanvas');
  const scale = Math.max(1, Math.floor(320 / Math.max(meta.width, meta.height)));
  canvas.width = meta.width * scale; canvas.height = meta.height * scale;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  $('frameLabel').textContent = `${index + 1} / ${meta.frames}`;
  drawLabelChart();
}

function stopPlayback() {
  clearInterval(S.cache.timer);
  S.cache.timer = null;
  $('playToggle').textContent = 'Play';
}

function togglePlayback() {
  if (S.cache.timer) { stopPlayback(); return; }
  if (!S.cache.meta) return;
  $('playToggle').textContent = 'Pause';
  S.cache.timer = setInterval(() => {
    S.cache.frame = (S.cache.frame + 1) % S.cache.meta.frames;
    $('frameSlider').value = S.cache.frame;
    drawFrame();
  }, 1000 / +$('playFps').value);
}

async function loadLabel(diffOverride) {
  const sample = S.cache.sample;
  if (!sample) return;
  const diff = diffOverride === undefined ? 'auto' : (diffOverride ? '1' : '0');
  try {
    S.cache.label = await api('/api/cache/label', { params: { path: sample.input, fs: $('sampleFs').value || '30', diff } });
    if (S.cache.label.is_diff !== undefined) $('labelDiff').checked = S.cache.label.is_diff;
    drawLabelChart();
  } catch (error) { toast(error.message, 'error'); }
}

function drawLabelChart() {
  const label = S.cache.label;
  const canvas = $('labelChart');
  const extra = $('labelExtra');
  if (!label || !label.available) {
    Charts.draw(canvas, { series: [], empty: 'No label file', height: 160 });
    extra.replaceChildren();
    return;
  }
  const frame = S.cache.frame;
  const fs = label.fs || 30;
  if (label.kind === 'physiology') {
    const names = Object.keys(label.channels);
    const n = label.channels[names[0]].length;
    const x = Array.from({ length: n }, (_, i) => i);
    Charts.draw(canvas, { height: 180, xLabel: 'frame', yLabel: 'bpm', series: [{ label: names[0], x, y: label.channels[names[0]] }], vlines: [{ x: frame, color: '#888' }] });
    if (!extra.dataset.built || extra.dataset.built !== S.cache.sample.input) {
      extra.dataset.built = S.cache.sample.input;
      extra.replaceChildren(h('p', { class: 'muted' }, `Mean heart rate: ${num(label.mean_hr, 1)} bpm`),
        ...names.slice(1).map(name => h('canvas', { class: 'chart', 'data-channel': name })));
    }
    extra.querySelectorAll('canvas').forEach(c => Charts.draw(c, { height: 130, xLabel: 'frame', series: [{ label: c.dataset.channel, x, y: label.channels[c.dataset.channel], color: '#3e63dd' }], vlines: [{ x: frame, color: '#888' }] }));
    return;
  }
  const x = label.signal.map((_, i) => i);
  Charts.draw(canvas, { height: 180, xLabel: 'frame', yLabel: 'label', series: [{ label: 'label', x, y: label.signal }], vlines: [{ x: frame, color: '#888' }] });
  if (extra.dataset.built !== S.cache.sample.input + label.is_diff + fs) {
    extra.dataset.built = S.cache.sample.input + label.is_diff + fs;
    const spectrum = h('canvas', { class: 'chart' });
    extra.replaceChildren(h('p', { class: 'hr-readout' }, `Estimated heart rate: ${num(label.hr, 1)} bpm`), spectrum);
    Charts.draw(spectrum, { height: 150, xLabel: 'Hz', yLabel: 'power', series: [{ label: 'spectrum', x: label.spectrum.freqs, y: label.spectrum.power, color: '#3e63dd' }], vlines: Number.isFinite(label.hr) ? [{ x: label.hr / 60, color: '#e5484d', dash: [4, 3] }] : [] });
  }
}

// ----------------------------------------------------------------- results
async function loadResults() {
  try {
    const data = await api('/api/results');
    S.results.experiments = data.experiments;
    $('resultRoots').textContent = data.roots.join('; ') || 'none';
    renderResultList();
    if (S.results.pendingJob) {
      const match = data.experiments.find(e => e.jobs.includes(S.results.pendingJob));
      S.results.pendingJob = null;
      if (match) selectExperiment(match.path);
      else toast('No outputs found yet for this job.', 'warn');
    }
    if (S.results.pendingPath) {
      const match = data.experiments.find(e => normHost(e.path) === normHost(S.results.pendingPath));
      S.results.pendingPath = null;
      if (match) selectExperiment(match.path);
      else toast('This folder is not under a results folder; add it in Settings → Folders to browse.', 'warn', 8000);
    }
  } catch (error) { toast(error.message, 'error'); }
}

function renderResultList() {
  const query = $('resultFilter').value.trim().toLowerCase();
  const descriptions = e => e.jobs.map(id => S.jobs.find(j => j.id === id)?.description || '').join('\n').toLowerCase();
  const items = S.results.experiments.filter(e => !query || e.name.toLowerCase().includes(query) || descriptions(e).includes(query));
  $('resultList').replaceChildren(...items.map(exp => {
    const jobNames = exp.jobs.map(id => S.jobs.find(j => j.id === id)?.name).filter(Boolean);
    const mae = exp.summary?.mae_bpm;
    return h('button', { type: 'button', class: 'item' + (exp.path === S.results.selected ? ' active' : ''), onclick: () => selectExperiment(exp.path) },
      h('div', { class: 'item-title' }, exp.name),
      h('div', { class: 'item-sub' }, exp.markers.map(m => m.replace('saved_', '').replace('PreTrainedModels', 'checkpoints').replace('cv_summary', 'cross-validation')).join(' · '),
        exp.checkpoints ? ` · ${exp.checkpoints} ckpt` : '', Number.isFinite(mae) ? ` · MAE ${num(mae)}` : ''),
      h('div', { class: 'item-sub' }, fmtDate(exp.modified), jobNames.length ? ' · job: ' + jobNames.join(', ') : ''));
  }));
  if (!items.length) $('resultList').append(h('p', { class: 'muted pad' }, 'No experiment folders found.'));
}

async function selectExperiment(path) {
  S.results.selected = path;
  S.results.analysis = null;
  renderResultList();
  const panel = $('resultDetail');
  panel.replaceChildren(h('p', { class: 'muted' }, 'Loading…'));
  try {
    S.results.detail = await api('/api/results/detail', { params: { path } });
  } catch (error) { panel.replaceChildren(h('div', { class: 'msg error' }, error.message)); return; }
  const exp = S.results.experiments.find(e => e.path === path);
  const tabs = ['Overview', 'Plots', 'Predictions', 'Files'];
  const body = h('div', { id: 'resultBody' });
  const bar = h('div', { class: 'subtabs' }, ...tabs.map(t => h('button', { type: 'button', 'data-sub': t, onclick: () => showResultTab(t) }, t)));
  panel.replaceChildren(h('div', { class: 'detail-head' }, h('div', {}, h('h2', {}, exp?.name || path), h('div', { class: 'muted' }, h('code', {}, path)))), bar, body);
  showResultTab('Overview');
}

function showResultTab(name) {
  document.querySelectorAll('.subtabs button').forEach(b => b.classList.toggle('active', b.dataset.sub === name));
  const body = $('resultBody');
  const files = S.results.detail.files;
  if (name === 'Overview' && S.results.detail.cross_validation) {
    body.replaceChildren(...crossValidationOverview(files));
  } else if (name === 'Overview') {
    const exp = S.results.experiments.find(e => e.path === S.results.selected);
    const parts = [];
    for (const id of exp?.jobs || []) {
      const job = S.jobs.find(j => j.id === id);
      if (!job) continue;
      parts.push(h('div', { class: 'card' }, h('h4', {}, 'Launcher job: ', job.name), h('div', { class: 'muted' }, `${job.status} · ${fmtDate(job.created)}`),
        job.description ? h('div', { class: 'description' }, job.description) : null, metricsTable(job.progress?.metrics)));
    }
    const checkpoints = files.filter(f => f.kind === 'checkpoint');
    if (checkpoints.length) {
      parts.push(h('div', { class: 'card' }, h('h4', {}, `Checkpoints (${checkpoints.length})`),
        h('table', { class: 'grid-table' }, ...checkpoints.map(f => h('tr', {}, h('td', {}, f.name), h('td', {}, fmtBytes(f.size)), h('td', {}, fmtDate(f.modified)),
          h('td', {}, h('button', { type: 'button', class: 'small', onclick: () => useCheckpoint(f.path) }, 'Evaluate…')))))));
    }
    for (const f of files.filter(f => f.content && (f.kind === 'json' || (f.kind === 'csv' && !/window_results/.test(f.name))))) {
      parts.push(h('div', { class: 'card' }, h('h4', {}, f.name), f.kind === 'json' ? jsonTable(f.content) : csvTable(f.content)));
    }
    if (!parts.length) parts.push(h('p', { class: 'muted' }, 'No summaries in this folder.'));
    body.replaceChildren(...parts);
  } else if (name === 'Plots') {
    const images = files.filter(f => f.kind === 'image');
    const pdfs = files.filter(f => f.kind === 'pdf');
    body.replaceChildren(...[
      images.length ? h('div', { class: 'gallery' }, ...images.map(f => h('figure', {}, h('a', { href: fileUrl(f.path), target: '_blank' }, h('img', { src: fileUrl(f.path), alt: f.name, loading: 'lazy' })), h('figcaption', {}, f.name)))) : h('p', { class: 'muted' }, 'No images.'),
      pdfs.length ? h('div', { class: 'card' }, h('h4', {}, 'PDF plots'), ...pdfs.map(f => h('div', {}, h('a', { href: fileUrl(f.path), target: '_blank' }, f.name)))) : null].filter(Boolean));
  } else if (name === 'Predictions') {
    renderPredictionControls(body, files);
  } else {
    body.replaceChildren(h('table', { class: 'grid-table files' }, h('tr', {}, ...['File', 'Type', 'Size', 'Modified', ''].map(t => h('th', {}, t))),
      ...files.map(f => h('tr', {}, h('td', {}, f.name), h('td', {}, f.kind), h('td', {}, fmtBytes(f.size)), h('td', {}, fmtDate(f.modified)),
        h('td', {}, h('a', { href: fileUrl(f.path, !['image', 'pdf', 'csv', 'json', 'text'].includes(f.kind)), target: '_blank' }, 'open'))))));
  }
}

// Overview of a cross-validation root: merged metrics, the folds and the launcher group.
function crossValidationOverview(files) {
  const table = name => files.find(f => f.name === `cv_summary/${name}.csv`)?.content;
  const rows = content => content ? content.rows.map(r => Object.fromEntries(content.header.map((c, i) => [c, r[i] !== '' && Number.isFinite(+r[i]) ? +r[i] : r[i]]))) : [];
  const parts = [];
  const group = (S.groups || []).find(g => g.root_host && normHost(g.root_host) === normHost(S.results.selected));
  if (group) {
    parts.push(h('div', { class: 'card' }, h('h4', {}, 'Launcher cross-validation: ', group.name),
      h('div', { class: 'muted' }, `${group.status} · ${group.folds.filter(f => f.status === 'succeeded').length} of ${group.folds.length} folds done · ${fmtDate(group.created)}`),
      group.description ? h('div', { class: 'description' }, group.description) : null,
      h('button', { type: 'button', class: 'small', onclick: () => { selectGroup(group.id); showTab('jobs'); } }, 'Open in Jobs')));
  }
  const overall = rows(table('overall_summary'));
  parts.push(h('div', { class: 'card' }, h('h4', {}, 'All folds'), overall.length ? summaryTable(overall) : h('p', { class: 'muted' }, 'Not merged yet.'),
    h('p', { class: 'muted small' }, 'Every evaluation window of every fold, against the measured HR. Predictions → all_windows.csv plots them.')));
  const plan = files.find(f => f.name === 'cv_plan.json')?.content;
  if (plan?.folds) parts.push(h('div', { class: 'card' }, h('h4', {}, 'Folds'), foldTable(plan.folds)));
  for (const [name, key] of [['by_fold', 'Fold'], ['by_participant', 'Participant'], ['by_movement', 'Movement'], ['by_view', 'View'], ['by_experiment', 'Experiment']]) {
    const data = rows(table(name));
    if (data.length) parts.push(h('div', { class: 'card' }, h('h4', {}, `By ${key.toLowerCase()}`), summaryTable(data, key)));
  }
  return parts;
}

function metricsTable(metrics) {
  const groups = Object.keys(metrics || {});
  if (!groups.length) return h('p', { class: 'muted' }, 'No metrics parsed from the log.');
  const names = ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'MACC'];
  return h('table', { class: 'grid-table' }, h('tr', {}, h('th', {}, ''), ...names.map(m => h('th', {}, m))),
    ...groups.map(g => h('tr', {}, h('th', {}, g), ...names.map(m => h('td', {}, metrics[g][m] ? num(metrics[g][m].value, 3) : '–')))));
}

function jsonTable(content) {
  if (content && typeof content === 'object' && !Array.isArray(content)) {
    return h('table', { class: 'kv' }, ...Object.entries(content).map(([k, v]) => h('tr', {}, h('th', {}, k), h('td', {}, typeof v === 'number' ? num(v, 3) : JSON.stringify(v)))));
  }
  return h('pre', { class: 'code' }, JSON.stringify(content, null, 2));
}

function csvTable(content) {
  const fmt = v => (v !== '' && Number.isFinite(+v) && /\./.test(v)) ? num(+v, 3) : v;
  return h('div', { class: 'table-wrap' }, h('table', { class: 'grid-table' }, h('tr', {}, ...content.header.map(c => h('th', {}, c))),
    ...content.rows.slice(0, 200).map(r => h('tr', {}, ...r.map(c => h('td', {}, fmt(c)))))));
}

function useCheckpoint(hostPath) {
  if (!S.values) { toast('Load an evaluation config in the editor first.', 'warn'); return; }
  let value = hostPath;
  if (S.backend === 'docker') {
    value = hostToContainer(hostPath);
    if (!value) { toast('This checkpoint is not inside a Docker mount (/runs or /checkpoints). Switch the editor to Local or adjust Settings.', 'error', 9000); return; }
  }
  S.values['INFERENCE.MODEL_PATH'] = value;
  if (S.values.TOOLBOX_MODE !== 'only_test') {
    S.values.TOOLBOX_MODE = 'only_test';
    toast('Editor switched to "Evaluation of a checkpoint"; review the test data section.', 'warn', 8000);
  } else toast('Checkpoint set in the editor.', 'ok');
  renderForm();
  scheduleValidate(0);
  showTab('editor');
}

// ------------------------------------------------------- prediction viewer
const LINE = { gt: '#1c2024', label: '#8b8d98', pred: '#e5484d', tracked: '#3e63dd', raw: '#b9bbc6', band: '#30a46c' };
// Index 0 is the current run (same red as its prediction); others avoid the tracked-HR blue.
const RUN_COLORS = ['#e5484d', '#f76b15', '#8e4ec6', '#12a594', '#ad7f58', '#d6409f', '#30a46c'];

function sourceKind(name) {
  if (/(^|\/)saved_valid_outputs\//.test(name)) return 'validation, best epoch';
  if (/(^|\/)saved_outputs\//.test(name)) return 'unsupervised';
  if (/(^|\/)saved_test_outputs\//.test(name)) return 'test';
  return '';
}

// Light-to-red background for an error value (0 = neutral, >= max = darkest).
function errorColor(value, max) {
  if (!Number.isFinite(value)) return {};
  const t = Math.min(1, Math.max(0, value / (max || 1)));
  const from = [241, 243, 245], to = [205, 43, 49];
  const rgb = from.map((c, i) => Math.round(c + (to[i] - c) * t));
  return { style: `background: rgb(${rgb.join(',')}); color: ${t > 0.55 ? '#fff' : 'inherit'}` };
}

function chartBlock(title, canvas, fileName, ...extra) {
  return h('div', { class: 'chart-block' },
    h('div', { class: 'chart-head' }, h('h4', {}, title), ...extra,
      h('button', { type: 'button', class: 'small', title: 'Download this chart as PNG', onclick: () => Charts.download(canvas, fileName) }, 'PNG')),
    canvas);
}

function renderPredictionControls(body, files) {
  const sources = files.filter(f => f.kind === 'pickle' || /(window_results|cv_summary\/all_windows)\.csv$/.test(f.name))
    .sort((a, b) => (a.kind === 'pickle' ? 0 : 1) - (b.kind === 'pickle' ? 0 : 1));
  if (!sources.length) { body.replaceChildren(h('p', { class: 'muted' }, 'No *_outputs.pickle, window_results.csv or merged all_windows.csv in this folder.')); return; }
  S.results.historyPath = files.find(f => /probe_history\.npz$/.test(f.name))?.path || null;
  S.results.compare = [];
  const source = h('select', {}, ...sources.map(f => h('option', { value: f.path }, f.name + (sourceKind(f.name) ? `  (${sourceKind(f.name)})` : ''))));
  const windowInput = h('input', { type: 'number', min: '0', step: '1', value: '10', title: '0 = whole recording' });
  const stepInput = h('input', { type: 'number', min: '0', step: '1', value: '2', title: '0 = same as the window' });
  const truth = h('select', { title: 'Reference heart rate' }, h('option', { value: 'measured' }, 'measured HR (when saved)'), h('option', { value: 'label' }, 'HR of the PPG label'));
  const diff = h('select', {}, h('option', { value: 'auto' }, 'auto'), h('option', { value: '1' }, 'yes'), h('option', { value: '0' }, 'no'));
  const track = h('input', { type: 'checkbox', title: 'Add a causal Viterbi-tracked HR line (limits jumps to ~8 bpm/s)' });
  const out = h('div', { id: 'analysisOut' });
  const run = async () => {
    out.replaceChildren(h('p', { class: 'muted' }, 'Analysing…'));
    try {
      const params = { path: source.value, window: windowInput.value || '0', step: stepInput.value || '0', diff: diff.value,
        truth: truth.value, track: track.checked ? '1' : '0' };
      S.results.analysis = await api('/api/results/predictions', { params });
      S.results.analysisParams = params;
      renderAnalysis(out);
      if (S.results.compare.length) loadComparison();
    } catch (error) { out.replaceChildren(h('div', { class: 'msg error' }, error.message)); }
  };
  body.replaceChildren(h('div', { class: 'toolbar' },
    h('label', {}, 'Source ', source), h('label', {}, 'Window (s) ', windowInput), h('label', {}, 'Step (s) ', stepInput),
    h('label', {}, 'Ground truth ', truth), h('label', {}, 'Prediction is differenced ', diff),
    h('label', {}, track, ' Tracked HR'), h('button', { type: 'button', class: 'primary', onclick: run }, 'Analyse')),
    h('p', { class: 'muted small' }, 'Heart rate per sliding window with the toolbox method (integrate if differenced, detrend, 0.6–3.3 Hz band-pass, FFT peak). ',
      'Measured HR is the vHRM sensor HR saved with newer runs; older outputs fall back to the HR of the PPG label. ',
      'Use ↑/↓ to step through recordings.'), out);
  run();
}

function renderAnalysis(out) {
  const a = S.results.analysis;
  const o = a.overall || {};
  const cards = [['Windows', o.windows, 0], ['MAE (bpm)', o.mae], ['RMSE (bpm)', o.rmse], ['MAPE (%)', o.mape], ['Pearson', o.pearson, 3],
    ['Bias (bpm)', o.bias], ['Within ±5 bpm (%)', o.within5, 0], ['SNR (dB)', o.snr]];
  const reference = a.source === 'csv' ? 'window_results.csv' : a.truth === 'measured' ? 'measured HR' : 'HR of the PPG label';
  const info = [a.split && `split: ${a.split}`, a.meta?.epoch !== undefined && a.meta?.epoch !== null && `epoch ${a.meta.epoch}`,
    a.meta?.model && `model: ${a.meta.model}`, `reference: ${reference}`, a.has_gt === false && a.source !== 'csv' && 'no measured HR saved in this file'].filter(Boolean);
  const scatter = h('canvas', { class: 'chart' });
  const ba = h('canvas', { class: 'chart' });
  const filters = h('div', { class: 'toolbar flat' });
  const heat = h('div', { class: 'table-wrap' });
  const grouping = h('select', {}, ...[['movement', 'view'], ['participant', 'movement'], ['participant', 'view']]
    .map(([r, c]) => h('option', { value: `${r}|${c}` }, `${r} × ${c}`)));
  const table = h('div', { class: 'table-wrap' });
  const compareCard = h('div', { id: 'compareCard' });
  const detail = h('div', { id: 'recordingDetail' });
  const hasMeta = a.recordings.some(r => r.participant);
  out.replaceChildren(h('div', { class: 'cards' }, ...cards.map(([label, v, d]) => h('div', { class: 'metric' }, h('span', {}, label), h('b', {}, num(v, d ?? 2))))),
    h('p', { class: 'muted small' }, info.join(' · ')),
    h('div', { class: 'two-col' }, h('div', { class: 'card' }, chartBlock('Predicted vs ground-truth HR', scatter, 'scatter')),
      h('div', { class: 'card' }, chartBlock('Bland-Altman', ba, 'bland_altman'))),
    hasMeta ? h('div', { class: 'card' }, h('div', { class: 'chart-head' }, h('h4', {}, 'MAE by scenario (bpm)'), grouping,
      h('span', { class: 'muted small' }, 'Click a cell to filter; click again to clear.')), heat) : null,
    compareCard,
    h('div', { class: 'card' }, h('div', { class: 'chart-head' }, h('h4', {}, 'Recordings'),
      h('button', { type: 'button', class: 'small', onclick: () => pickExtreme(-1) }, 'Worst'),
      h('button', { type: 'button', class: 'small', onclick: () => pickExtreme(1) }, 'Best'),
      a.source === 'csv' ? null : h('button', { type: 'button', class: 'small', onclick: openComparePicker }, 'Compare with…')),
    filters, table), detail);
  const state = S.results.filter = { participant: '', movement: '', view: '' };
  const selects = {};
  if (hasMeta) {
    for (const key of ['participant', 'movement', 'view']) {
      const values = [...new Set(a.recordings.map(r => r[key]))].sort();
      selects[key] = h('select', { onchange: e => { state[key] = e.target.value; draw(); } }, h('option', { value: '' }, `All ${key}s`), ...values.map(v => h('option', { value: v }, v)));
      filters.append(selects[key]);
    }
    grouping.addEventListener('change', () => draw());
  }
  const meta = Object.fromEntries(a.recordings.map(r => [r.id, r]));
  const drawHeat = rows => {
    const [rowKey, colKey] = grouping.value.split('|');
    const ids = new Set(rows.map(r => r.id));
    const cells = {};
    for (const p of a.points) {
      if (!ids.has(p.recording)) continue;
      const m = meta[p.recording];
      const k = `${m[rowKey]}|${m[colKey]}`;
      (cells[k] ||= { sum: 0, n: 0, recs: new Set() });
      cells[k].sum += Math.abs(p.pred - p.gt); cells[k].n += 1; cells[k].recs.add(p.recording);
    }
    const rowValues = [...new Set(rows.map(r => r[rowKey]))].sort();
    const colValues = [...new Set(rows.map(r => r[colKey]))].sort();
    const maes = Object.values(cells).map(c => c.sum / c.n);
    const max = Math.max(10, ...maes);
    const toggle = (r, c) => {
      const active = state[rowKey] === r && state[colKey] === c;
      state[rowKey] = active ? '' : r; state[colKey] = active ? '' : c;
      for (const [k, sel] of Object.entries(selects)) sel.value = state[k];
      draw();
    };
    heat.replaceChildren(h('table', { class: 'grid-table heat' }, h('tr', {}, h('th', {}, `${rowKey} \\ ${colKey}`), ...colValues.map(c => h('th', {}, c))),
      ...rowValues.map(r => h('tr', {}, h('th', {}, r), ...colValues.map(c => {
        const cell = cells[`${r}|${c}`];
        if (!cell) return h('td', { class: 'muted' }, '');
        const mae = cell.sum / cell.n;
        return h('td', { ...errorColor(mae, max), class: 'heat-cell' + (state[rowKey] === r && state[colKey] === c ? ' active' : ''),
          title: `${cell.n} windows, ${cell.recs.size} recordings`, onclick: () => toggle(r, c) }, num(mae, 1));
      })))));
  };
  const draw = () => {
    const rows = a.recordings.filter(r => Object.entries(state).every(([k, v]) => !v || r[k] === v));
    if (hasMeta) {
      // The grid keeps all its own rows/columns; only the other filters narrow it.
      const [rowKey, colKey] = grouping.value.split('|');
      drawHeat(a.recordings.filter(r => Object.entries(state).every(([k, v]) => !v || k === rowKey || k === colKey || r[k] === v)));
    }
    const ids = new Set(rows.map(r => r.id));
    const pts = a.points.filter(p => ids.has(p.recording));
    const selected = S.results.recording;
    const highlight = pts.filter(p => p.recording === selected);
    const gt = pts.map(p => p.gt), pred = pts.map(p => p.pred);
    Charts.draw(scatter, { height: 260, identity: true, xLabel: 'ground truth (bpm)', yLabel: 'predicted (bpm)', legend: false,
      series: [{ type: 'points', x: gt, y: pred, labels: pts.map(p => p.recording), color: '#3e63dd' },
        { type: 'points', x: highlight.map(p => p.gt), y: highlight.map(p => p.pred), color: '#e5484d', radius: 4, labels: highlight.map(p => p.recording) }] });
    const mean = pts.map(p => (p.gt + p.pred) / 2), d = pts.map(p => p.pred - p.gt);
    const bias = d.reduce((s, v) => s + v, 0) / (d.length || 1);
    const sd = Math.sqrt(d.reduce((s, v) => s + (v - bias) ** 2, 0) / Math.max(1, d.length - 1));
    Charts.draw(ba, { height: 260, xLabel: 'mean of GT and prediction (bpm)', yLabel: 'prediction − GT (bpm)', legend: false,
      series: [{ type: 'points', x: mean, y: d, labels: pts.map(p => p.recording), color: '#3e63dd' },
        { type: 'points', x: highlight.map(p => (p.gt + p.pred) / 2), y: highlight.map(p => p.pred - p.gt), color: '#e5484d', radius: 4, labels: highlight.map(p => p.recording) }],
      hlines: [{ y: bias, color: '#e5484d', label: `bias ${num(bias)}` }, { y: bias + 1.96 * sd, dash: [4, 3], label: `+1.96 SD ${num(bias + 1.96 * sd)}` }, { y: bias - 1.96 * sd, dash: [4, 3], label: `−1.96 SD ${num(bias - 1.96 * sd)}` }] });
    const sortKey = S.results.sort;
    const ascending = ['id', 'within5'].includes(sortKey);
    rows.sort((x, y) => sortKey === 'id' ? x.id.localeCompare(y.id) : ascending ? (x[sortKey] ?? Infinity) - (y[sortKey] ?? Infinity) : (y[sortKey] ?? -Infinity) - (x[sortKey] ?? -Infinity));
    S.results.visible = rows.map(r => r.id);
    const cols = [['id', 'Recording'], ['windows', 'Windows'], ['duration', 'Duration (s)'], ['mean_gt', 'GT HR'], ['mean_pred', 'Pred HR'], ['mae', 'MAE'],
      ...(a.track ? [['mae_tracked', 'MAE tracked']] : []), ['rmse', 'RMSE'], ['bias', 'Bias'], ['within5', '±5 bpm %'], ['max_jump', 'Max jump'], ['pearson', 'r'], ['snr', 'SNR']];
    const maxMae = Math.max(10, ...rows.map(r => r.mae).filter(Number.isFinite));
    table.replaceChildren(h('table', { class: 'grid-table clickable' },
      h('tr', {}, ...cols.map(([k, label]) => h('th', { class: k === sortKey ? 'sorted' : '', onclick: () => { S.results.sort = k; draw(); } }, label))),
      ...rows.map(r => h('tr', { class: r.id === selected ? 'active' : '', onclick: () => selectRecording(r.id) },
        ...cols.map(([k]) => h('td', k === 'mae' ? errorColor(r.mae, maxMae) : {}, k === 'id' ? r.id : num(r[k], k === 'windows' || k === 'within5' ? 0 : k === 'pearson' ? 3 : 1)))))));
  };
  S.results.redraw = draw;
  draw();
}

function pickExtreme(direction) {
  const a = S.results.analysis;
  const ids = new Set(S.results.visible || []);
  const rows = a.recordings.filter(r => ids.has(r.id) && Number.isFinite(r.mae)).sort((x, y) => (y.mae - x.mae) * -direction);
  if (rows.length) selectRecording(rows[0].id);
}

function stepRecording(delta) {
  const ids = S.results.visible || [];
  if (!ids.length) return;
  const index = ids.indexOf(S.results.recording);
  const next = ids[Math.max(0, Math.min(ids.length - 1, index < 0 ? 0 : index + delta))];
  if (next !== S.results.recording) selectRecording(next);
}

async function openComparePicker() {
  let list;
  try { list = (await api('/api/results/sources')).sources; } catch (error) { toast(error.message, 'error'); return; }
  const current = S.results.analysisParams.path;
  const choices = list.filter(s => s.path !== current);
  if (!choices.length) { toast('No other prediction outputs found under the results folders.', 'warn'); return; }
  const chosen = new Set(S.results.compare);
  const filter = h('input', { type: 'search', placeholder: 'Filter…' });
  const box = h('div', { class: 'browse-list' });
  const render = () => {
    const q = filter.value.toLowerCase();
    box.replaceChildren(...choices.filter(s => !q || (s.experiment + ' ' + s.name).toLowerCase().includes(q)).map(s => h('label', { class: 'browse-item option' },
      h('input', { type: 'checkbox', checked: chosen.has(s.path), onchange: e => { e.target.checked ? chosen.add(s.path) : chosen.delete(s.path); } }),
      h('span', {}, h('b', {}, s.experiment), ' / ', s.name, h('small', {}, fmtDate(s.modified) + (sourceKind(s.name) ? ' · ' + sourceKind(s.name) : ''))))));
  };
  filter.addEventListener('input', render);
  render();
  openModal('Compare with other runs', h('div', { class: 'form-stack' },
    h('p', { class: 'muted small' }, 'Runs are compared on the recordings they share, with the current window, step and ground-truth settings.'), filter, box), [
    { label: 'Clear comparison', onClick: () => { S.results.compare = []; closeModal(); $('compareCard').replaceChildren(); if (S.results.recording) selectRecording(S.results.recording); } },
    { label: 'Compare', primary: true, onClick: () => { S.results.compare = [...chosen]; closeModal(); loadComparison(); } }], true);
}

async function loadComparison() {
  const card = $('compareCard');
  if (!card || !S.results.compare.length) return;
  card.replaceChildren(h('div', { class: 'card' }, h('p', { class: 'muted' }, 'Comparing runs…')));
  let data;
  try {
    data = await api('/api/results/compare', { params: { ...S.results.analysisParams, paths: [S.results.analysisParams.path, ...S.results.compare].join('|') } });
  } catch (error) { card.replaceChildren(h('div', { class: 'msg error' }, error.message)); return; }
  S.results.compareRuns = data.runs;
  const max = Math.max(10, ...data.recordings.flatMap(r => r.mae).filter(Number.isFinite));
  const meanOf = i => { const v = data.recordings.map(r => r.mae[i]).filter(Number.isFinite); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN; };
  card.replaceChildren(h('div', { class: 'card' }, h('div', { class: 'chart-head' }, h('h4', {}, `Model comparison — MAE (bpm) on ${data.recordings.length} shared recordings`),
    h('button', { type: 'button', class: 'small', onclick: openComparePicker }, 'Change…')),
    h('div', { class: 'table-wrap' }, h('table', { class: 'grid-table clickable' },
      h('tr', {}, h('th', {}, 'Recording'), ...data.runs.map((run, i) => h('th', { title: run.path }, h('span', { class: 'swatch', style: `background:${RUN_COLORS[i % RUN_COLORS.length]}` }), run.name))),
      h('tr', {}, h('th', {}, 'Mean'), ...data.runs.map((_, i) => h('th', errorColor(meanOf(i), max), num(meanOf(i), 1)))),
      ...data.recordings.map(r => h('tr', { class: r.id === S.results.recording ? 'active' : '', onclick: () => selectRecording(r.id) }, h('td', {}, r.id),
        ...r.mae.map(v => h('td', errorColor(v, max), num(v, 1)))))))));
  if (S.results.recording) selectRecording(S.results.recording);
}

async function selectRecording(id) {
  S.results.recording = id;
  S.results.redraw?.();
  const a = S.results.analysis;
  const detail = $('recordingDetail');
  detail.replaceChildren(h('p', { class: 'muted' }, 'Loading recording…'));
  let data, compare = null, history = null;
  if (a.source === 'csv') {
    data = { id, windows: a.windows_by_recording[id] || [], metrics: a.recordings.find(r => r.id === id) };
  } else {
    try {
      const params = { ...S.results.analysisParams, id };
      [data, compare, history] = await Promise.all([
        api('/api/results/recording', { params }),
        S.results.compare.length ? api('/api/results/compare', { params: { ...params, paths: [params.path, ...S.results.compare].join('|') } }) : null,
        S.results.historyPath && (a.split === 'valid' || /saved_valid_outputs/.test(a.path)) ? api('/api/results/history', { params: { ...params, path: S.results.historyPath } }) : null,
      ]);
    } catch (error) { detail.replaceChildren(h('div', { class: 'msg error' }, error.message)); return; }
  }
  if (S.results.recording !== id) return;  // a newer selection is loading
  renderRecording(detail, data, compare?.runs?.slice(1) || [], history?.epochs?.length ? history : null);
}

function renderRecording(detail, data, others, history) {
  const view = S.results.view ||= { span: 30, offset: 0, raw: false, beats: true };
  const state = { epoch: null };
  const duration = data.duration || Math.max(0, ...data.windows.map(w => w.end));
  view.offset = Math.min(view.offset, Math.max(0, duration - (view.span || duration)));
  const m = data.metrics || {}, mt = data.metrics_tracked;
  const chips = [['MAE', m.mae, 1], ['bias', m.bias, 1], ['RMSE', m.rmse, 1], ['r', m.pearson, 2], ['within ±5', m.within5, 0, '%'], ['max jump', m.max_jump, 1],
    ...(mt ? [['tracked MAE', mt.mae, 1], ['tracked max jump', mt.max_jump, 1]] : [])];
  const centre = w => (w.start + w.end) / 2;
  const hr = h('canvas', { class: 'chart' }), err = h('canvas', { class: 'chart' }), wave = h('canvas', { class: 'chart' }), spec = h('canvas', { class: 'chart' });
  const caption = h('span', { class: 'muted small' });
  const hasGtLine = data.truth === 'measured' && data.gt_channels?.heart_rate_bpm;
  const activeWindows = () => state.epoch ? state.epoch.windows : data.windows;
  const activePrediction = () => state.epoch ? state.epoch.prediction : data.prediction;
  const viewRange = () => view.span ? [view.offset, Math.min(duration, view.offset + view.span)] : [0, duration];
  const invalid = (data.invalid_spans || []).map(([x0, x1]) => ({ x0, x1, color: '#8b8d98', alpha: 0.18 }));

  const drawHr = () => {
    const windows = activeWindows();
    const series = [];
    if (hasGtLine) {
      const values = data.gt_channels.heart_rate_bpm;
      series.push({ label: 'measured HR', x: values.map((_, i) => i * duration / values.length), y: values, color: LINE.gt, width: 2 });
    } else {
      series.push({ label: data.truth === 'measured' ? 'ground truth' : 'label HR (reference)', x: data.windows.map(centre), y: data.windows.map(w => w.gt), color: LINE.gt, width: 2, markers: true });
    }
    if (hasGtLine && data.windows.some(w => Number.isFinite(w.label_hr))) series.push({ label: 'label HR', x: data.windows.map(centre), y: data.windows.map(w => w.label_hr), color: LINE.label, dash: [5, 4] });
    series.push({ label: state.epoch ? `predicted (epoch ${state.epoch.epoch})` : 'predicted', x: windows.map(centre), y: windows.map(w => w.pred), color: LINE.pred, markers: true });
    if (!state.epoch && windows.some(w => Number.isFinite(w.tracked))) series.push({ label: 'tracked', x: windows.map(centre), y: windows.map(w => w.tracked), color: LINE.tracked });
    others.forEach((run, i) => series.push({ label: run.name, x: run.windows.map(centre), y: run.windows.map(w => w.pred), color: RUN_COLORS[(i + 1) % RUN_COLORS.length], dash: [3, 2] }));
    const [x0, x1] = viewRange();
    Charts.draw(hr, { height: 250, xLabel: 'time (s)', yLabel: 'heart rate (bpm)', xMin: 0, xMax: duration, series,
      vspans: [...invalid, ...(data.prediction ? [{ x0, x1, color: '#3e63dd', alpha: 0.1 }] : [])],
      onClick: data.prediction ? x => { if (view.span) { view.offset = Math.max(0, Math.min(duration - view.span, x - view.span / 2)); offset.value = view.offset; drawWave(); drawHr(); } } : undefined });
    const errors = windows.map(w => w.pred - w.gt);
    const errSeries = [{ label: 'prediction − GT', x: windows.map(centre), y: errors, color: LINE.pred, markers: true }];
    if (!state.epoch && windows.some(w => Number.isFinite(w.tracked))) errSeries.push({ label: 'tracked − GT', x: windows.map(centre), y: windows.map(w => w.tracked - w.gt), color: LINE.tracked });
    Charts.draw(err, { height: 130, xLabel: 'time (s)', yLabel: 'error (bpm)', xMin: 0, xMax: duration, series: errSeries,
      hbands: [{ y0: -5, y1: 5, color: LINE.band, alpha: 0.15 }], hlines: [{ y: 0 }], vspans: invalid });
  };

  const valueAt = (values, t) => values[Math.max(0, Math.min(values.length - 1, Math.round(t / duration * values.length)))];
  const beatRate = times => times.length > 1 ? 60 / ((times[times.length - 1] - times[0]) / (times.length - 1)) : NaN;
  const drawWave = () => {
    const prediction = activePrediction();
    if (!prediction) return;
    const [x0, x1] = viewRange();
    const t = prediction.map((_, i) => i * duration / prediction.length);
    const series = [];
    if (view.raw && data.prediction_raw && !state.epoch) series.push({ label: 'prediction (unfiltered)', x: t, y: data.prediction_raw, color: LINE.raw, width: 1 });
    if (data.label) series.push({ label: 'PPG label', x: data.label.map((_, i) => i * duration / data.label.length), y: data.label, color: LINE.gt });
    series.push({ label: state.epoch ? `prediction (epoch ${state.epoch.epoch})` : 'prediction', x: t, y: prediction, color: LINE.pred });
    others.forEach((run, i) => run.prediction && series.push({ label: run.name, x: run.prediction.map((_, k) => k * run.duration / run.prediction.length), y: run.prediction, color: RUN_COLORS[(i + 1) % RUN_COLORS.length], width: 1 }));
    const inView = times => (times || []).filter(v => v >= x0 && v <= x1);
    const parts = [];
    if (view.beats && !state.epoch) {
      const pb = inView(data.pred_beats);
      series.push({ type: 'marks', shape: 'up', x: pb, y: pb.map(v => valueAt(data.prediction, v)), color: LINE.pred });
      parts.push(`prediction: ${pb.length} beats, ${num(beatRate(pb), 0)} bpm`);
      if (data.label_beats) {
        const lb = inView(data.label_beats);
        series.push({ type: 'marks', shape: 'down', x: lb, y: lb.map(v => valueAt(data.label, v)), color: LINE.gt });
        parts.push(`label: ${lb.length} beats, ${num(beatRate(lb), 0)} bpm`);
      }
      const windows = activeWindows().filter(w => centre(w) >= x0 && centre(w) <= x1 && Number.isFinite(w.gt));
      if (windows.length) parts.push(`reference ${num(windows.reduce((s, w) => s + w.gt, 0) / windows.length, 0)} bpm`);
    }
    caption.textContent = `${num(x0, 0)}–${num(x1, 0)} s` + (parts.length ? ' · in view: ' + parts.join(' · ') : '');
    Charts.draw(wave, { height: 230, xLabel: 'time (s)', yLabel: 'z', series, xMin: x0, xMax: x1, vspans: invalid });
  };

  const parts = [h('div', { class: 'chart-head' }, h('h3', {}, data.id),
    h('div', { class: 'chips' }, ...chips.map(([label, v, d, unit]) => h('span', { class: 'stat' }, label, ' ', h('b', {}, num(v, d) + (unit && Number.isFinite(v) ? unit : '')))))),
    chartBlock('Heart rate over time' + (data.prediction ? ' — click to move the waveform window' : ''), hr, `${data.id}_hr`),
    chartBlock('Error per window (green band = ±5 bpm)', err, `${data.id}_error`)];

  let offset = h('input', { type: 'range' });
  if (data.prediction) {
    const span = h('select', {}, ...[10, 20, 30, 60, 0].map(s => h('option', { value: s }, s ? `${s} s` : 'all')));
    span.value = String(view.span);
    offset = h('input', { type: 'range', min: '0', max: String(Math.max(0, Math.floor(duration))), value: String(view.offset), step: '1' });
    const raw = h('input', { type: 'checkbox', checked: view.raw });
    const beats = h('input', { type: 'checkbox', checked: view.beats });
    span.addEventListener('change', () => { view.span = +span.value; view.offset = Math.min(view.offset, Math.max(0, duration - view.span)); drawWave(); drawHr(); });
    offset.addEventListener('input', () => { view.offset = +offset.value; drawWave(); drawHr(); });
    raw.addEventListener('change', () => { view.raw = raw.checked; drawWave(); });
    beats.addEventListener('change', () => { view.beats = beats.checked; drawWave(); });
    parts.push(chartBlock('Pulse waveform (z-scored, band-passed)', wave, `${data.id}_bvp`,
      h('label', { class: 'small' }, 'Span ', span), h('label', { class: 'small grow' }, 'Start ', offset),
      h('label', { class: 'small' }, raw, ' unfiltered'), h('label', { class: 'small' }, beats, ' beats')), caption);
  }
  if (history) {
    const select = h('select', {}, h('option', { value: '' }, 'best epoch (saved file)'), ...history.epochs.map((e, i) => h('option', { value: i }, `epoch ${e.epoch} — MAE ${num(e.metrics?.mae, 1)}`)));
    select.addEventListener('change', () => { state.epoch = select.value === '' ? null : history.epochs[+select.value]; drawHr(); drawWave(); });
    const trend = h('canvas', { class: 'chart' });
    parts.splice(1, 0, h('div', { class: 'card inset' }, h('div', { class: 'chart-head' }, h('h4', {}, 'Probe recording — predictions during training'), h('label', { class: 'small' }, 'Show ', select)), trend));
    requestAnimationFrame(() => Charts.draw(trend, { height: 120, xLabel: 'epoch', yLabel: 'MAE (bpm)', legend: false,
      series: [{ x: history.epochs.map(e => e.epoch), y: history.epochs.map(e => e.metrics?.mae ?? NaN), color: LINE.pred, markers: true }] }));
  }
  if (data.spectrogram?.times?.length) parts.push(chartBlock('Prediction spectrogram (white: measured HR and 2× harmonic)', spec, `${data.id}_spectrogram`));
  const aux = Object.entries(data.gt_channels || {}).filter(([name]) => name !== 'heart_rate_bpm');
  const auxCanvases = aux.map(() => h('canvas', { class: 'chart' }));
  aux.forEach(([name], i) => parts.push(chartBlock(name, auxCanvases[i], `${data.id}_${name}`)));
  if (!data.prediction) parts.push(h('p', { class: 'muted small' }, 'Waveforms need the output pickle.'));
  detail.replaceChildren(h('div', { class: 'card' }, ...parts));

  drawHr();
  drawWave();
  if (data.spectrogram?.times?.length) {
    const sp = data.spectrogram;
    const series = [];
    if (hasGtLine) {
      const values = data.gt_channels.heart_rate_bpm;
      const x = values.map((_, i) => i * duration / values.length);
      series.push({ label: 'measured HR', x, y: values, color: '#ffffff', width: 1.5 }, { label: '2×', x, y: values.map(v => v * 2), color: '#ffffff', dash: [3, 3], width: 1 });
    } else {
      series.push({ label: 'reference HR', x: data.windows.map(centre), y: data.windows.map(w => w.gt), color: '#ffffff', width: 1.5 });
    }
    Charts.draw(spec, { height: 210, xLabel: 'time (s)', yLabel: 'frequency (bpm)', xMin: 0, xMax: duration, yMin: 0, yMax: 240,
      heatmap: { x: sp.times, y: sp.freqs, z: sp.power }, series, legend: false });
  }
  aux.forEach(([name, values], i) => Charts.draw(auxCanvases[i], { height: 130, xLabel: 'time (s)', legend: false, xMin: 0, xMax: duration,
    series: [{ x: values.map((_, k) => k * duration / values.length), y: values, color: '#30a46c' }] }));
}

// ---------------------------------------------------------------- settings
async function renderSettings() {
  const data = await api('/api/settings');
  const s = data.settings;
  $('setPython').value = s.local_python;
  $('setPolicy').value = s.gpu_policy;
  $('setMaxJobs').value = s.max_parallel_jobs;
  $('setBackend').value = s.default_backend;
  $('setResultsRoots').value = s.results_roots.join('\n');
  $('setCacheRoots').value = s.cache_roots.join('\n');
  const env = data.docker_env;
  const descriptions = {
    DATA_ROOT: 'Host folder mounted read-only at /data', CHECKPOINT_ROOT: 'Host folder mounted read-only at /checkpoints',
    CACHE_PATH: 'Host folder mounted at /cache', RUNS_PATH: 'Host folder mounted at /runs', GPU_DEVICE_ID: 'Default host GPU',
    DOCKER_SHM_SIZE: 'Shared memory, e.g. 8gb', TOOLBOX_IMAGE: 'Image tag',
  };
  $('envFields').replaceChildren(...Object.entries(env).map(([k, v]) => h('label', {}, `${k} `, h('small', { class: 'muted' }, descriptions[k] || ''), h('input', { type: 'text', 'data-env': k, value: v }))));
  $('envFields').prepend(h('p', { class: 'muted small' }, data.env_file_exists ? 'Editing .env (other lines are preserved).' : 'No .env yet: saving creates it from docker/.env.example.'));
  renderMounts();
  $('wandbKeyState').textContent = data.wandb_key_in_env ? 'A WANDB_API_KEY is present in the environment or .env.' : 'No WANDB_API_KEY in the environment or .env.';
}

function renderMounts() {
  $('mountTable').replaceChildren(h('table', { class: 'grid-table' }, h('tr', {}, h('th', {}, 'Container'), h('th', {}, 'Host'), h('th', {}, '')),
    ...S.boot.docker_mounts.map(m => h('tr', {}, h('td', {}, m.container), h('td', {}, h('code', {}, m.host)), h('td', { class: m.exists ? 'ok' : (m.read_only ? 'bad' : 'muted') }, m.exists ? 'exists' : (m.read_only ? 'missing' : 'created on first run'))))));
}

async function saveSettings() {
  const lines = id => $(id).value.split('\n').map(s => s.trim()).filter(Boolean);
  const dockerEnv = {};
  document.querySelectorAll('[data-env]').forEach(input => { dockerEnv[input.dataset.env] = input.value; });
  try {
    const result = await api('/api/settings', { method: 'POST', body: {
      settings: { local_python: $('setPython').value, gpu_policy: $('setPolicy').value, max_parallel_jobs: +$('setMaxJobs').value,
        default_backend: $('setBackend').value, results_roots: lines('setResultsRoots'), cache_roots: lines('setCacheRoots') },
      docker_env: dockerEnv,
    } });
    S.boot.settings = result.settings;
    S.boot.docker_mounts = result.docker_mounts;
    S.boot.docker_env = result.docker_env;
    renderMounts();
    S.cache.sets = [];
    $('settingsResult').textContent = 'Saved.';
    toast('Settings saved.', 'ok');
    if (S.values) scheduleValidate(0);
  } catch (error) {
    $('settingsResult').textContent = error.message;
    toast(error.message, 'error');
  }
}

async function envTest(target, out) {
  out.textContent = target === 'docker_gpu' ? 'Starting a container (can take a while)…' : 'Testing…';
  try {
    const r = await api('/api/env/test', { method: 'POST', body: { target } });
    if (target === 'local') {
      out.textContent = r.ok ? `Python ${r.python}, torch ${r.torch_version || 'missing'}, CUDA ${r.cuda ? 'yes: ' + r.gpus.join(', ') : 'no'}; ` +
        ['yacs', 'cv2', 'scipy', 'wandb'].map(k => `${k} ${r[k] ? '✓' : '✗'}`).join(' ') : r.error;
    } else if (target === 'docker') {
      out.textContent = r.ok ? `Docker ${r.server}, Compose ${r.compose}; image ${r.image?.exists ? 'present (' + r.image.created + ')' : 'not built'}` : r.error;
    } else {
      out.textContent = (r.ok ? 'OK: ' : 'Failed: ') + r.output;
    }
    out.className = r.ok ? 'ok' : 'bad';
  } catch (error) { out.textContent = error.message; out.className = 'bad'; }
}

// -------------------------------------------------------------------- init
async function init() {
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    if (S.boot) showTab(t.dataset.tab); else S.pendingTab = t.dataset.tab;
  }));
  S.boot = await api('/api/bootstrap');
  S.fields = S.boot.schema.fields;
  S.fieldMap = Object.fromEntries(S.fields.map(f => [f.key, f]));
  S.backend = S.boot.settings.default_backend;
  populateConfigSelect();
  setBackend(S.backend, false);
  renderForm();

  $('loadConfig').addEventListener('click', () => loadConfig($('configSelect').value));
  $('configSearch').addEventListener('input', renderConfigTree);
  $('configPicker').addEventListener('toggle', () => {
    if ($('configPicker').open) $('configSearch').focus();
  });
  document.addEventListener('click', e => {
    if (!$('configPicker').contains(e.target)) $('configPicker').open = false;
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && $('configPicker').open) {
      $('configPicker').open = false;
      $('configPicker').querySelector('summary').focus();
      e.preventDefault();
    }
  });
  $('newBlank').addEventListener('click', () => newBlank($('blankMode').value));
  document.querySelectorAll('input[name="backend"]').forEach(r => r.addEventListener('change', () => setBackend(r.value)));
  $('convertPaths').addEventListener('click', convertPaths);
  $('linkPreprocess').addEventListener('change', togglePreprocessingSync);
  $('syncSettings').addEventListener('click', () => syncSettingsDialog());
  $('showAdvanced').addEventListener('change', renderForm);
  $('previewYaml').addEventListener('click', previewYaml);
  $('saveConfig').addEventListener('click', saveConfigDialog);
  $('launchRun').addEventListener('click', launchDialog);
  $('modalClose').addEventListener('click', closeModal);
  $('modal').addEventListener('click', e => { if (e.target === $('modal')) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !$('modal').classList.contains('hidden')) closeModal(); });

  $('jobFilter').addEventListener('change', renderJobList);

  $('cacheList').addEventListener('change', e => loadSamples(e.target.value));
  $('sampleFilter').addEventListener('input', renderSamples);
  $('frameSlider').addEventListener('input', e => { S.cache.frame = +e.target.value; drawFrame(); });
  $('playToggle').addEventListener('click', togglePlayback);
  $('playFps').addEventListener('change', () => { if (S.cache.timer) { stopPlayback(); togglePlayback(); } });
  $('channelGroup').addEventListener('change', loadFrames);
  $('sampleFs').addEventListener('change', () => loadLabel($('labelDiff').checked));
  $('labelDiff').addEventListener('change', () => loadLabel($('labelDiff').checked));
  document.addEventListener('keydown', e => {
    if (!$('tab-cache').classList.contains('active') || !S.cache.meta || ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      S.cache.frame = Math.max(0, Math.min(S.cache.meta.frames - 1, S.cache.frame + (e.key === 'ArrowRight' ? 1 : -1)));
      $('frameSlider').value = S.cache.frame;
      drawFrame();
    } else if (e.key === ' ') { e.preventDefault(); togglePlayback(); }
  });

  document.addEventListener('keydown', e => {
    if (!$('tab-results').classList.contains('active') || !S.results.analysis || !$('recordingDetail') ||
        ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) || !$('modal').classList.contains('hidden')) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); stepRecording(e.key === 'ArrowDown' ? 1 : -1); }
  });
  $('resultFilter').addEventListener('input', renderResultList);
  $('refreshResults').addEventListener('click', loadResults);

  $('saveSettings').addEventListener('click', saveSettings);
  $('testLocal').addEventListener('click', () => envTest('local', $('localResult')));
  $('testDocker').addEventListener('click', () => envTest('docker', $('dockerResult')));
  $('testDockerGpu').addEventListener('click', () => envTest('docker_gpu', $('dockerResult')));

  window.addEventListener('resize', () => { clearTimeout(init.resize); init.resize = setTimeout(() => { if (S.cache.label) drawLabelChart(); updateJobDetail(); }, 200); });
  setInterval(() => { if (!S.stale) pollJobs(); }, 2000);
  setInterval(() => { if (!S.stale && S.log && !S.log.done) pollLog(); }, 1500);
  pollJobs();
  if (S.pendingTab) showTab(S.pendingTab);
}

init().catch(error => {
  document.body.prepend(h('div', { class: 'msg error pad' }, 'Could not start the launcher UI: ' + error.message));
});
