// Run with: node --test tools/tests/test_launcher_sync.cjs
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Lightweight DOM fixture: exercise the shipped UI without dependencies,
// a running launcher, or modifying its bootstrap/init behavior in production.
class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.events = {};
    this.className = '';
    this.text = '';
    this.classList = {
      add: c => { this.className += ' ' + c; },
      remove: c => { this.className = this.className.split(' ').filter(x => x !== c).join(' '); },
      contains: c => this.className.split(' ').includes(c),
      toggle: (c, on) => on ? this.classList.add(c) : this.classList.remove(c),
    };
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(c => c.textContent).join(''); }
  setAttribute(key, value) { this.attrs[key] = String(value); if (key === 'disabled') this.disabled = true; }
  addEventListener(event, fn) { this.events[event] = fn; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = []; this.text = ''; this.append(...children); }
  querySelectorAll(selector) {
    const tags = selector.split(',').map(s => s.trim().toUpperCase());
    return this.children.flatMap(c => [...(tags.includes(c.tagName) ? [c] : []), ...c.querySelectorAll(selector)]);
  }
}

function fixture() {
  const elements = new Map();
  const document = {
    querySelector: () => ({ content: 'test-token' }),
    getElementById: id => {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement: tag => new Element(tag),
    createTextNode: text => { const el = new Element('#text'); el.textContent = text; return el; },
    querySelectorAll: () => [],
  };
  const context = vm.createContext({ document, Node: Element, console, setTimeout, clearTimeout });
  const file = path.join(__dirname, '../../launcher/static/app.js');
  const source = fs.readFileSync(file, 'utf8');
  const init = source.lastIndexOf('\ninit().catch(');
  assert.ok(init > 0, 'Only omit application startup in the test fixture');
  vm.runInContext(source.slice(0, init), context, { filename: file });
  vm.runInContext('renderForm = () => updateSyncControls(); scheduleValidate = () => {};', context);
  const api = vm.runInContext(`({ S, onFieldChange, syncFields, preprocessingSyncField, syncTarget,
    syncTrainingValue, updateSyncControls, applySyncLinks, previewSyncLinks,
    togglePreprocessingSync, syncSettingsDialog, renderField, setValues,
    closeModal, previewYaml, saveConfigDialog, applyModelDefaults })`, context);
  const specs = [
    ['FS', 'int', 30], ['DATA_FORMAT', 'enum', 'NDCHW'],
    ['PREPROCESS.RESIZE.W', 'int', 72], ['PREPROCESS.DATA_TYPE', 'multi', ['Raw']],
    ['PREPROCESS.DO_CHUNK', 'bool', true], ['PREPROCESS.LABEL_TYPE', 'enum', 'Raw'],
    ['PREPROCESS.RESIZE.H', 'int', 72], ['PREPROCESS.CHUNK_LENGTH', 'int', 160],
    ['PREPROCESS.CROP_FACE.LARGE_BOX_COEF', 'float', 1.5],
    ['PREPROCESS.IBVP.DATA_MODE', 'enum', 'RGB'],
    ['DO_PREPROCESS', 'bool', false], ['VHRM.CROP_MODE_BY_VIEW', 'yamllist', [['front', 'face']]],
    ['VHRM.VIEWS', 'list', ['front']], ['DATASET', 'enum', 'vHRM'],
    ['DATA_PATH', 'path', '/data'], ['CACHED_PATH', 'path', '/cache'],
    ['BEGIN', 'float', 0], ['END', 'float', 1], ['VHRM.PARTICIPANTS', 'list', []],
    ['VHRM.EXPERIMENTS', 'list', []], ['EXP_DATA_NAME', 'str', 'test'],
  ];
  api.S.fields = ['TRAIN.DATA', 'VALID.DATA', 'TEST.DATA'].flatMap(split => specs.map(([suffix, type, value]) => ({
    key: `${split}.${suffix}`, type, label: suffix, default: value, section: split,
    options: ['Raw', 'DiffNormalized', 'NDCHW', 'RGB'],
  })));
  api.S.fieldMap = Object.fromEntries(api.S.fields.map(f => [f.key, f]));
  api.S.values = Object.fromEntries(api.S.fields.map(f => [f.key, structuredClone(f.default)]));
  api.S.values.TOOLBOX_MODE = 'train_and_test';
  api.S.values['TEST.USE_LAST_EPOCH'] = true;
  api.S.boot = { schema: { model_profiles: {} } };
  return { ...api, elements, context, el: document.getElementById };
}

function click(el) { el.events.click(); }
function button(f, label) {
  const found = f.el('modalFoot').querySelectorAll('button').find(b => b.textContent === label);
  assert.ok(found, `Modal has a ${label} button`);
  return found;
}
const trainFs = 'TRAIN.DATA.FS', validFs = 'VALID.DATA.FS', testFs = 'TEST.DATA.FS';

test('links start off; eligibility excludes identities, participants, paths and split boundaries', () => {
  const f = fixture();
  f.updateSyncControls();
  assert.equal(f.el('linkPreprocess').checked, false);
  assert.equal(f.el('linkPreprocess').indeterminate, false);
  assert.equal(f.el('syncSettings').disabled, false);
  const keys = new Set(f.syncFields().map(field => field.key));
  for (const suffix of ['DATASET', 'DATA_PATH', 'CACHED_PATH', 'BEGIN', 'END', 'VHRM.PARTICIPANTS', 'VHRM.EXPERIMENTS', 'EXP_DATA_NAME']) {
    assert.equal(keys.has(`TRAIN.DATA.${suffix}`), false, suffix);
  }
  assert.ok(keys.has('TRAIN.DATA.VHRM.VIEWS'));
  f.S.fieldMap[testFs].type = 'float';
  assert.equal(f.syncFields().some(field => field.key === trainFs), false, 'Only compatible fields are offered');
});

test('quick preset previews before changing values; cancellation leaves it off', () => {
  const f = fixture();
  f.S.values[validFs] = 10;
  f.el('linkPreprocess').checked = true;
  f.togglePreprocessingSync();
  assert.equal(f.el('modalTitle').textContent, 'Preview training sync');
  assert.ok(f.el('modalBody').textContent.includes(validFs));
  assert.equal(f.S.values[validFs], 10);
  assert.equal(Object.keys(f.S.syncLinks).length, 0);
  assert.equal(f.el('linkPreprocess').checked, false);
  click(button(f, 'Cancel'));
  assert.equal(f.S.values[validFs], 10);
});

test('quick preset applies to both splits including hidden fields; source edits propagate', () => {
  const f = fixture();
  f.el('linkPreprocess').checked = true;
  f.togglePreprocessingSync();
  click(button(f, 'Apply links'));
  assert.equal(f.el('linkPreprocess').checked, true);
  assert.equal(f.el('linkPreprocess').indeterminate, false);
  assert.equal(Object.keys(f.S.syncLinks).length, f.syncFields().filter(f.preprocessingSyncField).length * 2);
  f.onFieldChange(trainFs, 60);
  assert.equal(f.S.values[validFs], 60, 'Hidden validation values also follow training');
  assert.equal(f.S.values[testFs], 60);
  assert.equal(f.S.syncLinks['TEST.DATA.PREPROCESS.IBVP.DATA_MODE'], 'TRAIN.DATA.PREPROCESS.IBVP.DATA_MODE');
});

test('custom links target only the selected split and deep-copy lists', () => {
  const f = fixture();
  const source = 'TRAIN.DATA.VHRM.CROP_MODE_BY_VIEW', target = 'TEST.DATA.VHRM.CROP_MODE_BY_VIEW';
  f.applySyncLinks({ [testFs]: trainFs, [target]: source });
  assert.equal(f.el('linkPreprocess').indeterminate, true);
  f.onFieldChange(trainFs, 55);
  assert.equal(f.S.values[testFs], 55);
  assert.equal(f.S.values[validFs], 30);
  const value = [['front', { mode: 'roi', coords: [1, 2] }]];
  f.onFieldChange(source, value);
  value[0][1].coords[0] = 99;
  assert.equal(f.S.values[target][0][1].coords[0], 1);
});

test('linked controls are disabled and protected; unlink retains the value', () => {
  const f = fixture();
  const transforms = 'TEST.DATA.PREPROCESS.DATA_TYPE';
  f.applySyncLinks({ [testFs]: trainFs, [transforms]: 'TRAIN.DATA.PREPROCESS.DATA_TYPE' });
  for (const key of [testFs, transforms]) {
    const field = f.renderField(f.S.fieldMap[key]);
    assert.ok(field.classList.contains('linked-field'));
    assert.ok(field.querySelectorAll('input, select, textarea').every(input => input.disabled));
  }
  f.onFieldChange(testFs, 999);
  assert.equal(f.S.values[testFs], 30);
  const field = f.renderField(f.S.fieldMap[testFs]);
  click(field.querySelectorAll('button').find(b => b.textContent === 'Unlink'));
  assert.equal(f.S.syncLinks[testFs], undefined);
  assert.equal(f.S.values[testFs], 30);
  f.onFieldChange(testFs, 45);
  assert.equal(f.S.values[testFs], 45);
  assert.equal(f.S.values[trainFs], 30);
});

test('turning off the quick preset keeps non-preset custom links and values', () => {
  const f = fixture();
  const custom = 'TEST.DATA.VHRM.VIEWS';
  f.applySyncLinks({ [custom]: 'TRAIN.DATA.VHRM.VIEWS' });
  f.el('linkPreprocess').checked = true;
  f.togglePreprocessingSync();
  click(button(f, 'Apply links'));
  f.onFieldChange(trainFs, 50);
  f.el('linkPreprocess').checked = false;
  f.togglePreprocessingSync();
  assert.deepEqual(Object.keys(f.S.syncLinks), [custom]);
  assert.equal(f.el('linkPreprocess').indeterminate, true);
  assert.equal(f.S.values[testFs], 50);
  f.onFieldChange(trainFs, 60);
  assert.equal(f.S.values[testFs], 50);
});

test('custom picker supports independent targets, Back, preview and Apply', () => {
  const f = fixture();
  f.S.values[testFs] = 12;
  f.syncSettingsDialog();
  const boxes = f.el('modalBody').querySelectorAll('input');
  const target = boxes.find(b => b.attrs['aria-label'] === 'FS: sync test with training');
  target.checked = true;
  target.events.change();
  click(button(f, 'Preview changes'));
  assert.equal(f.S.values[testFs], 12);
  click(button(f, 'Back'));
  assert.equal(f.el('modalBody').querySelectorAll('input').find(b => b.attrs['aria-label'] === target.attrs['aria-label']).checked, true);
  click(button(f, 'Preview changes'));
  click(button(f, 'Apply links'));
  assert.deepEqual(Object.keys(f.S.syncLinks), [testFs]);
  assert.equal(f.S.values[testFs], 30);
});

test('custom preset shortcut and Clear all change only the draft until applied', () => {
  const f = fixture();
  f.applySyncLinks({ [testFs]: trainFs });
  f.syncSettingsDialog();
  click(f.el('modalBody').querySelectorAll('button').find(b => b.textContent === 'Select preprocessing settings'));
  assert.equal(Object.keys(f.S.syncLinks).length, 1);
  const boxes = f.el('modalBody').querySelectorAll('input');
  assert.equal(boxes.filter(b => b.checked).length, f.syncFields().filter(f.preprocessingSyncField).length * 2);
  click(f.el('modalBody').querySelectorAll('button').find(b => b.textContent === 'Clear all'));
  assert.ok(boxes.every(b => !b.checked));
  assert.equal(Object.keys(f.S.syncLinks).length, 1);
  click(button(f, 'Preview changes'));
  click(button(f, 'Apply links'));
  assert.equal(Object.keys(f.S.syncLinks).length, 0);
  assert.equal(f.S.values[testFs], 30);
  assert.equal(f.el('linkPreprocess').indeterminate, false);
});

test('loading another config resets links without overwriting loaded values', () => {
  const f = fixture();
  f.applySyncLinks({ [testFs]: trainFs });
  const loaded = { ...f.S.values, [testFs]: 17 };
  f.setValues(loaded, {}, 'local', 'Loaded config', 'config.yaml');
  assert.equal(Object.keys(f.S.syncLinks).length, 0);
  assert.equal(f.S.values[testFs], 17);
  assert.equal(f.el('linkPreprocess').checked, false);
});

test('leaving training mode clears links and enables independent test editing', () => {
  const f = fixture();
  f.applySyncLinks({ [testFs]: trainFs });
  f.onFieldChange('TOOLBOX_MODE', 'only_test');
  assert.equal(Object.keys(f.S.syncLinks).length, 0);
  assert.equal(f.el('syncSettings').disabled, true);
  f.onFieldChange(testFs, 20);
  f.onFieldChange('TOOLBOX_MODE', 'train_and_test');
  assert.equal(f.S.values[testFs], 20);
});

test('applying model defaults keeps linked hidden validation consistent', () => {
  const f = fixture();
  const source = 'TRAIN.DATA.PREPROCESS.RESIZE.W', target = 'VALID.DATA.PREPROCESS.RESIZE.W';
  f.applySyncLinks({ [target]: source });
  f.S.boot.schema.model_profiles.Physnet = { format: 'NDCHW', types: ['Raw'], size: 128, label: 'Raw', chunk: 64 };
  vm.runInContext('toast = () => {};', f.context);
  f.applyModelDefaults('Physnet');
  assert.equal(f.S.values[source], 128);
  assert.equal(f.S.values[target], 128);
});

test('YAML preview and save send ordinary values/extras, never link metadata', async () => {
  const f = fixture();
  f.applySyncLinks({ [testFs]: trainFs });
  f.S.extras = { EXTRA: 'preserved' };
  f.S.boot.save_locations = { local: 'configs' };
  f.S.backend = 'local';
  const requests = [];
  f.context.captureRequest = (url, options) => {
    requests.push({ url, body: JSON.parse(JSON.stringify(options.body)) });
    if (url.endsWith('/save')) return { path: 'configs/example.yaml', configs: [] };
    return { yaml: 'TEST: {}' };
  };
  vm.runInContext('api = async (url, options) => captureRequest(url, options); populateConfigSelect = () => {}; selectConfig = () => {}; toast = () => {};', f.context);
  await f.previewYaml();
  f.saveConfigDialog();
  await button(f, 'Save').events.click();
  assert.equal(requests.length, 2);
  for (const { body } of requests) {
    assert.equal(body.values[testFs], 30);
    assert.deepEqual(body.extras, { EXTRA: 'preserved' });
    assert.equal(JSON.stringify(body).includes('syncLinks'), false);
  }
});