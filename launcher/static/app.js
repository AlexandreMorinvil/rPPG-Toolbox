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
  validation: null,
  validateTimer: null,
  validateSeq: 0,
  preflight: null,
  jobs: [],
  gpus: [],
  selectedJob: null,
  log: null,
  cache: { sets: [], samples: [], sample: null, frames: null, meta: null, frame: 0, timer: null, label: null },
  results: { experiments: [], selected: null, detail: null, analysis: null, recording: null, sort: 'mae' },
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
  return true;
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
  || key === 'TEST.USE_LAST_EPOCH' || key === 'WANDB.ENABLED';

const fieldId = key => 'f_' + key.replace(/[^A-Za-z0-9]/g, '_');
const slug = text => text.toLowerCase().replace(/[^a-z0-9]+/g, '-');

// ------------------------------------------------------------------ editor
function populateConfigSelect() {
  const select = $('configSelect');
  const groups = { docker: 'Docker configs (docker/configs)', local: 'Local configs (configs)' };
  select.replaceChildren();
  for (const [group, label] of Object.entries(groups)) {
    const og = h('optgroup', { label });
    for (const item of S.boot.configs.filter(c => c.group === group)) {
      const rel = item.path.replace(/^(docker\/)?configs\//, '');
      const detail = item.error ? 'unreadable' : [item.mode, item.model, item.train || item.test].filter(Boolean).join(' · ');
      og.append(h('option', { value: item.path }, `${rel}  —  ${detail}`));
    }
    if (og.children.length) select.append(og);
  }
  const blank = $('blankMode');
  blank.replaceChildren(...S.boot.schema.modes.map(m => h('option', { value: m }, S.boot.schema.mode_labels[m])));
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
  const previous = S.values[key];
  S.values[key] = value;
  const linked = linkPreprocessing(key, value);
  if (key === 'MODEL.NAME' && value && value !== previous) offerModelDefaults(value);
  if (isStructural(key) || linked.length) renderForm();
  scheduleValidate();
}

function linkPreprocessing(key, value) {
  if (!$('linkPreprocess').checked || !key.startsWith('TRAIN.DATA.')) return [];
  const suffix = key.slice('TRAIN.DATA.'.length);
  if (!(suffix.startsWith('PREPROCESS.') || suffix === 'DATA_FORMAT' || suffix === 'FS')) return [];
  const changed = [];
  for (const split of ['VALID.DATA', 'TEST.DATA']) {
    const target = `${split}.${suffix}`;
    if (!(target in S.fieldMap)) continue;
    if (split === 'TEST.DATA' && S.values['TEST.DATA.DATASET'] === 'vHRM' &&
        ['PREPROCESS.LABEL_TYPE', 'PREPROCESS.CHUNK_LENGTH', 'PREPROCESS.DO_CHUNK', 'PREPROCESS.NUM_WORKERS'].includes(suffix)) continue;
    S.values[target] = Array.isArray(value) ? [...value] : value;
    changed.push(target);
  }
  return changed;
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
    const vhrm = v[split + '.DATASET'] === 'vHRM';
    v[split + '.DATA_FORMAT'] = p.format;
    v[split + '.PREPROCESS.DATA_TYPE'] = [...p.types];
    v[split + '.PREPROCESS.RESIZE.H'] = p.size;
    v[split + '.PREPROCESS.RESIZE.W'] = p.size;
    if (!vhrm) v[split + '.PREPROCESS.LABEL_TYPE'] = p.label;
    if (!vhrm || p.frame_num) v[split + '.PREPROCESS.CHUNK_LENGTH'] = p.chunk;
    if (vhrm) v[split + '.VHRM.PREDICTION_IS_DIFF'] = p.label === 'DiffNormalized';
  }
  if (p.frame_num) v[p.frame_num] = p.chunk;
  if (p.channels_key) v[p.channels_key] = 3 * p.types.length;
  if (model === 'FactorizePhys' && p.size === 72) v['MODEL.FactorizePhys.TYPE'] = 'Standard';
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
      $('configSelect').value = result.path;
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

// ------------------------------------------------------------------ launch
async function launchDialog() {
  if (!S.values) return;
  if (S.validation?.errors.length) { toast('Fix the validation errors first.', 'error'); return; }
  S.preflight = { answers: {}, secrets: {}, name: '', result: null, busy: false };
  openModal('Pre-launch checks', h('div', { id: 'preflightBody' }, h('p', { class: 'muted' }, 'Checking environment…')),
    [{ label: 'Cancel', onClick: closeModal }, { label: 'Re-check', id: 'pfRecheck', onClick: () => refreshPreflight() },
      { label: 'Launch', primary: true, id: 'pfLaunch', disabled: true, onClick: doLaunch }], true);
  await refreshPreflight();
}

async function refreshPreflight(autofill = true) {
  const pf = S.preflight;
  if (!pf) return;
  pf.busy = true;
  $('pfLaunch').disabled = true;
  try {
    const result = await api('/api/preflight', { method: 'POST', body: { values: S.values, extras: S.extras, backend: S.backend, answers: pf.answers, secrets: pf.secrets } });
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
  const name = h('input', { type: 'text', placeholder: 'Optional job name', value: pf.name });
  name.addEventListener('input', () => { pf.name = name.value; });
  body.append(h('label', { class: 'job-name' }, 'Job name', name));
  $('pfLaunch').disabled = !pf.result.can_launch;
}

async function doLaunch() {
  const pf = S.preflight;
  if (!pf?.result?.can_launch) return;
  $('pfLaunch').disabled = true;
  try {
    const result = await api('/api/jobs', { method: 'POST', body: { values: S.values, extras: S.extras, backend: S.backend, answers: pf.answers, secrets: pf.secrets, name: pf.name, source: S.source } });
    closeModal();
    S.selectedJob = result.job.id;
    S.log = null;
    toast(`Job "${result.job.name}" ${result.job.status}.`, 'ok');
    showTab('jobs');
  } catch (error) {
    toast(error.message, 'error', 10000);
    $('pfLaunch').disabled = false;
  }
}

// -------------------------------------------------------------------- jobs
const PHASES = { starting: 'Starting', preprocessing: 'Preprocessing', training: 'Training', validating: 'Validating', testing: 'Testing', unsupervised: 'Unsupervised', succeeded: 'Succeeded', failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted' };

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
    S.gpus = data.gpus;
  } catch (error) { return; }
  const active = S.jobs.filter(j => ['queued', 'starting', 'running', 'stopping'].includes(j.status)).length;
  $('jobsBadge').textContent = active;
  $('jobsBadge').classList.toggle('hidden', !active);
  $('gpuStrip').replaceChildren(...S.gpus.map(g => h('span', { class: 'gpu', title: g.name },
    `GPU${g.index} ${g.utilization}% · ${(g.memory_used / 1024).toFixed(1)}/${(g.memory_total / 1024).toFixed(0)} GB`)));
  if ($('tab-jobs').classList.contains('active')) {
    renderJobList();
    updateJobDetail();
  }
}

function renderJobList() {
  const filter = $('jobFilter').value;
  const list = $('jobList');
  const jobs = S.jobs.filter(j => filter === 'all' || (filter === 'active') === ['queued', 'starting', 'running', 'stopping'].includes(j.status));
  if (!S.selectedJob && jobs.length) S.selectedJob = jobs[0].id;
  list.replaceChildren(...jobs.map(job => {
    const fraction = jobFraction(job);
    const phase = PHASES[job.progress?.phase] || job.status;
    return h('button', { type: 'button', class: 'item job ' + job.status + (job.id === S.selectedJob ? ' active' : ''), onclick: () => { S.selectedJob = job.id; S.log = null; renderJobList(); updateJobDetail(true); } },
      h('div', { class: 'item-title' }, h('span', { class: 'dot' }), job.name),
      h('div', { class: 'item-sub' }, `${job.backend} · ${job.status === 'running' ? phase : job.status} · ${elapsed(job)}`),
      fraction !== null && job.status === 'running' ? h('div', { class: 'progress' }, h('div', { style: `width:${(fraction * 100).toFixed(1)}%` })) : null);
  }));
  if (!jobs.length) list.append(h('p', { class: 'muted pad' }, 'No jobs yet. Launch one from "New run".'));
}

function updateJobDetail(force = false) {
  const job = S.jobs.find(j => j.id === S.selectedJob);
  const panel = $('jobDetail');
  if (!job) { panel.replaceChildren(h('p', { class: 'muted' }, 'Select a job.')); return; }
  if (force || panel.dataset.job !== job.id) {
    panel.dataset.job = job.id;
    panel.replaceChildren(
      h('div', { id: 'jdHead' }), h('div', { id: 'jdProgress', class: 'card' }), h('div', { id: 'jdError' }),
      h('div', { class: 'two-col' }, h('div', { class: 'card' }, h('h4', {}, 'Loss per epoch'), h('canvas', { id: 'jdLoss', class: 'chart' })),
        h('div', { class: 'card' }, h('h4', {}, 'Results'), h('div', { id: 'jdMetrics' }))),
      h('div', { class: 'card' }, h('div', { class: 'log-head' }, h('h4', {}, 'Console output'),
        h('label', { class: 'check' }, h('input', { type: 'checkbox', id: 'logFollow', checked: true }), 'Follow')),
        h('pre', { id: 'jobLog', class: 'log' })));
    S.log = { jobId: job.id, next: null, lines: [], partial: '', done: false };
    pollLog();
  }
  const p = job.progress || {};
  const actions = [];
  if (['queued', 'running', 'starting'].includes(job.status)) actions.push(h('button', { type: 'button', class: 'danger', onclick: () => cancelJob(job) }, job.status === 'queued' ? 'Cancel' : 'Stop'));
  actions.push(h('button', { type: 'button', onclick: () => viewJobConfig(job) }, 'View config'));
  actions.push(h('button', { type: 'button', onclick: () => editJobConfig(job) }, 'Open in editor'));
  if ((job.outputs || {}).log_path_host) actions.push(h('button', { type: 'button', onclick: () => openResultsForJob(job) }, 'Results'));
  if (!['queued', 'running', 'starting', 'stopping'].includes(job.status)) actions.push(h('button', { type: 'button', onclick: () => removeJob(job) }, 'Remove'));
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
    if (job && !['queued', 'starting', 'running', 'stopping'].includes(job.status) && data.next >= data.size) log.done = true;
  } catch (error) {
    /* job removed or log not yet available */
  } finally {
    log.busy = false;
  }
}

async function cancelJob(job) {
  if (!confirm(`Stop "${job.name}"?`)) return;
  try { await api(`/api/jobs/${job.id}/cancel`, { method: 'POST', body: {} }); pollJobs(); } catch (error) { toast(error.message, 'error'); }
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
    setValues(data.values, data.extras, job.backend, `Config of job "${job.name}" (edit and launch again)`, job.source_config);
    showTab('editor');
  } catch (error) { toast(error.message, 'error'); }
}

function openResultsForJob(job) {
  S.results.pendingJob = job.id;
  showTab('results');
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
  } catch (error) { toast(error.message, 'error'); }
}

function renderResultList() {
  const query = $('resultFilter').value.trim().toLowerCase();
  const items = S.results.experiments.filter(e => !query || e.name.toLowerCase().includes(query));
  $('resultList').replaceChildren(...items.map(exp => {
    const jobNames = exp.jobs.map(id => S.jobs.find(j => j.id === id)?.name).filter(Boolean);
    const mae = exp.summary?.mae_bpm;
    return h('button', { type: 'button', class: 'item' + (exp.path === S.results.selected ? ' active' : ''), onclick: () => selectExperiment(exp.path) },
      h('div', { class: 'item-title' }, exp.name),
      h('div', { class: 'item-sub' }, exp.markers.map(m => m.replace('saved_', '').replace('PreTrainedModels', 'checkpoints')).join(' · '),
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
  if (name === 'Overview') {
    const exp = S.results.experiments.find(e => e.path === S.results.selected);
    const parts = [];
    for (const id of exp?.jobs || []) {
      const job = S.jobs.find(j => j.id === id);
      if (!job) continue;
      parts.push(h('div', { class: 'card' }, h('h4', {}, 'Launcher job: ', job.name), h('div', { class: 'muted' }, `${job.status} · ${fmtDate(job.created)}`), metricsTable(job.progress?.metrics)));
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

function renderPredictionControls(body, files) {
  const sources = files.filter(f => f.kind === 'pickle' || /window_results\.csv$/.test(f.name));
  if (!sources.length) { body.replaceChildren(h('p', { class: 'muted' }, 'No *_outputs.pickle or window_results.csv in this folder.')); return; }
  const source = h('select', {}, ...sources.map(f => h('option', { value: f.path }, f.name)));
  const windowInput = h('input', { type: 'number', min: '0', step: '1', value: '10', title: '0 = whole recording' });
  const stepInput = h('input', { type: 'number', min: '0', step: '1', value: '0', title: '0 = same as the window' });
  const diff = h('select', {}, h('option', { value: 'auto' }, 'auto'), h('option', { value: '1' }, 'yes'), h('option', { value: '0' }, 'no'));
  const out = h('div', { id: 'analysisOut' });
  const run = async () => {
    out.replaceChildren(h('p', { class: 'muted' }, 'Analysing…'));
    try {
      const params = { path: source.value, window: windowInput.value || '0', step: stepInput.value || '0', diff: diff.value };
      S.results.analysis = await api('/api/results/predictions', { params });
      S.results.analysisParams = params;
      renderAnalysis(out);
    } catch (error) { out.replaceChildren(h('div', { class: 'msg error' }, error.message)); }
  };
  body.replaceChildren(h('div', { class: 'toolbar' },
    h('label', {}, 'Source ', source), h('label', {}, 'Window (s) ', windowInput), h('label', {}, 'Step (s) ', stepInput),
    h('label', {}, 'Prediction is differenced ', diff), h('button', { type: 'button', class: 'primary', onclick: run }, 'Analyse')),
    h('p', { class: 'muted small' }, 'Heart rate is estimated per window with the toolbox method (integrate if differenced, detrend, 0.6–3.3 Hz band-pass, FFT peak).'), out);
  run();
}

function renderAnalysis(out) {
  const a = S.results.analysis;
  const o = a.overall || {};
  const cards = [['Windows', o.windows, 0], ['MAE (bpm)', o.mae], ['RMSE (bpm)', o.rmse], ['MAPE (%)', o.mape], ['Pearson', o.pearson, 3], ['Bias (bpm)', o.bias], ['SNR (dB)', o.snr]];
  const scatter = h('canvas', { class: 'chart' });
  const ba = h('canvas', { class: 'chart' });
  const filters = h('div', { class: 'toolbar' });
  const table = h('div', { class: 'table-wrap' });
  const detail = h('div', { id: 'recordingDetail' });
  out.replaceChildren(h('div', { class: 'cards' }, ...cards.map(([label, v, d]) => h('div', { class: 'metric' }, h('span', {}, label), h('b', {}, num(v, d ?? 2))))),
    h('div', { class: 'two-col' }, h('div', { class: 'card' }, h('h4', {}, 'Predicted vs ground-truth HR'), scatter), h('div', { class: 'card' }, h('h4', {}, 'Bland-Altman'), ba)),
    h('div', { class: 'card' }, h('h4', {}, 'Recordings'), filters, table), detail);
  const hasMeta = a.recordings.some(r => r.participant);
  const state = { participant: '', movement: '', view: '' };
  if (hasMeta) {
    for (const key of ['participant', 'movement', 'view']) {
      const values = [...new Set(a.recordings.map(r => r[key]))].sort();
      const select = h('select', { onchange: e => { state[key] = e.target.value; draw(); } }, h('option', { value: '' }, `All ${key}s`), ...values.map(v => h('option', { value: v }, v)));
      filters.append(select);
    }
  }
  const draw = () => {
    const rows = a.recordings.filter(r => Object.entries(state).every(([k, v]) => !v || r[k] === v));
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
      series: [{ type: 'points', x: mean, y: d, labels: pts.map(p => p.recording), color: '#3e63dd' }],
      hlines: [{ y: bias, color: '#e5484d', label: `bias ${num(bias)}` }, { y: bias + 1.96 * sd, dash: [4, 3], label: `+1.96 SD ${num(bias + 1.96 * sd)}` }, { y: bias - 1.96 * sd, dash: [4, 3], label: `−1.96 SD ${num(bias - 1.96 * sd)}` }] });
    const sortKey = S.results.sort;
    rows.sort((x, y) => sortKey === 'id' ? x.id.localeCompare(y.id) : (y[sortKey] ?? -Infinity) - (x[sortKey] ?? -Infinity));
    const cols = [['id', 'Recording'], ['windows', 'Windows'], ['duration', 'Duration (s)'], ['mean_gt', 'GT HR'], ['mean_pred', 'Pred HR'], ['mae', 'MAE'], ['rmse', 'RMSE'], ['bias', 'Bias'], ['pearson', 'r'], ['snr', 'SNR']];
    table.replaceChildren(h('table', { class: 'grid-table clickable' },
      h('tr', {}, ...cols.map(([k, label]) => h('th', { class: k === sortKey ? 'sorted' : '', onclick: () => { S.results.sort = k; draw(); } }, label))),
      ...rows.map(r => h('tr', { class: r.id === selected ? 'active' : '', onclick: () => selectRecording(r.id) },
        ...cols.map(([k]) => h('td', {}, k === 'id' ? r.id : num(r[k], k === 'windows' ? 0 : k === 'pearson' ? 3 : 1)))))));
  };
  S.results.redraw = draw;
  draw();
}

async function selectRecording(id) {
  S.results.recording = id;
  S.results.redraw?.();
  const a = S.results.analysis;
  const detail = $('recordingDetail');
  detail.replaceChildren(h('p', { class: 'muted' }, 'Loading recording…'));
  let data;
  if (a.source === 'csv') {
    data = { id, windows: a.windows_by_recording[id] || [], metrics: a.recordings.find(r => r.id === id) };
  } else {
    try {
      data = await api('/api/results/recording', { params: { ...S.results.analysisParams, id } });
    } catch (error) { detail.replaceChildren(h('div', { class: 'msg error' }, error.message)); return; }
  }
  const hr = h('canvas', { class: 'chart' });
  const parts = [h('h4', {}, `${id} — MAE ${num(data.metrics?.mae)} bpm`), hr];
  const center = data.windows.map(w => (w.start + w.end) / 2);
  Charts.draw(hr, { height: 230, xLabel: 'time (s)', yLabel: 'heart rate (bpm)', series: [
    { label: 'ground truth', x: center, y: data.windows.map(w => w.gt), color: '#222', markers: true },
    { label: 'predicted', x: center, y: data.windows.map(w => w.pred), color: '#e5484d', markers: true }] });
  if (data.prediction) {
    const wave = h('canvas', { class: 'chart' });
    const span = h('select', {}, ...[10, 30, 60, 0].map(s => h('option', { value: s }, s ? `${s} s` : 'all')));
    const offset = h('input', { type: 'range', min: '0', max: String(Math.max(0, Math.floor(data.duration))), value: '0', step: '1' });
    const step = data.duration / data.prediction.length;
    const t = data.prediction.map((_, i) => i * step);
    const drawWave = () => {
      const s = +span.value, o = +offset.value;
      const series = [{ label: 'prediction (processed)', x: t, y: data.prediction, color: '#e5484d' }];
      if (data.label) series.unshift({ label: 'label (processed)', x: t, y: data.label, color: '#222' });
      Charts.draw(wave, { height: 200, xLabel: 'time (s)', series, xMin: s ? o : undefined, xMax: s ? Math.min(data.duration, o + s) : undefined });
    };
    span.addEventListener('change', drawWave);
    offset.addEventListener('input', drawWave);
    span.value = '30';
    const spectrum = h('canvas', { class: 'chart' });
    parts.push(h('div', { class: 'toolbar' }, h('label', {}, 'Span ', span), h('label', { class: 'grow' }, 'Start ', offset)), wave,
      h('h4', {}, 'Prediction spectrum (whole recording)'), spectrum);
    detail.replaceChildren(h('div', { class: 'card' }, ...parts));
    drawWave();
    const gtMean = data.metrics?.mean_gt;
    Charts.draw(spectrum, { height: 150, xLabel: 'Hz', yLabel: 'power', legend: false, series: [{ x: data.spectrum.freqs, y: data.spectrum.power, color: '#3e63dd' }],
      vlines: Number.isFinite(gtMean) ? [{ x: gtMean / 60, color: '#222', dash: [4, 3] }] : [] });
    if (data.label_channels) {
      for (const [name, values] of Object.entries(data.label_channels).slice(1)) {
        const c = h('canvas', { class: 'chart' });
        detail.firstChild.append(h('h4', {}, name), c);
        const x = values.map((_, i) => i * data.duration / values.length);
        Charts.draw(c, { height: 130, xLabel: 'time (s)', legend: false, series: [{ x, y: values, color: '#30a46c' }] });
      }
    }
  } else {
    detail.replaceChildren(h('div', { class: 'card' }, ...parts, h('p', { class: 'muted small' }, 'Waveforms need the output pickle.')));
  }
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
  $('newBlank').addEventListener('click', () => newBlank($('blankMode').value));
  document.querySelectorAll('input[name="backend"]').forEach(r => r.addEventListener('change', () => setBackend(r.value)));
  $('convertPaths').addEventListener('click', convertPaths);
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
