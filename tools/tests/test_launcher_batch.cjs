// Run with: node --test tools/tests/test_launcher_batch.cjs
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Minimal DOM, as in test_launcher_crossval.cjs: runs the shipped app.js without its startup.
// Elements created with an id are found by getElementById, as in a browser.
let registry = new Map();

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.events = {};
    this.className = '';
    this.text = '';
    this.dataset = {};
    this.classList = {
      add: c => { this.className += ' ' + c; },
      remove: c => { this.className = this.className.split(' ').filter(x => x !== c).join(' '); },
      contains: c => this.className.split(' ').includes(c),
      toggle: (c, on) => on ? this.classList.add(c) : this.classList.remove(c),
    };
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(c => c.textContent).join(''); }
  setAttribute(key, value) { this.attrs[key] = String(value); if (key === 'id') registry.set(String(value), this); }
  addEventListener(event, fn) { this.events[event] = fn; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = []; this.text = ''; this.append(...children); }
  querySelectorAll() { return []; }
  all(predicate) { return this.children.flatMap(c => [...(predicate(c) ? [c] : []), ...c.all(predicate)]); }
  byClass(name) { return this.all(c => c.className.split(' ').includes(name)); }
  byTag(tag) { return this.all(c => c.tagName === tag.toUpperCase()); }
}

function fixture() {
  const elements = registry = new Map();
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
  const fetch = async () => { throw new Error('offline'); };
  const context = vm.createContext({ document, Node: Element, console, setTimeout, clearTimeout, Date, fetch, confirm: () => true });
  const file = path.join(__dirname, '../../launcher/static/app.js');
  const source = fs.readFileSync(file, 'utf8');
  const init = source.lastIndexOf('\ninit().catch(');
  assert.ok(init > 0, 'Only omit application startup in the test fixture');
  vm.runInContext(source.slice(0, init), context, { filename: file });
  vm.runInContext('renderForm = () => {}; scheduleValidate = () => {}; updateJobDetail = () => {};', context);
  const api = vm.runInContext(`({ S, renderJobList, renderBatchSelect, renderBatchChecks, updateBatchDetail, batchProgress, headlineMae,
    continueDialog, editQueuedDialog, renderContinueRows, renderEditRows, renderEditParameters, editRequest, continueRequest })`, context);
  const fields = [
    { key: 'TRAIN.EPOCHS', type: 'int', label: 'Epochs', section: 'Training' },
    { key: 'TRAIN.EARLY_STOPPING.ENABLED', type: 'bool', label: 'Enable early stopping', section: 'Training', group: 'Early stopping' },
    { key: 'TRAIN.EARLY_STOPPING.PATIENCE', type: 'int', label: 'Patience (epochs)', section: 'Training', group: 'Early stopping' },
    { key: 'WANDB.MODE', type: 'enum', label: 'W&B mode', section: 'Tracking', options: ['online', 'offline'] },
    { key: 'DEVICE', type: 'enum', label: 'Device', section: 'Run', options: ['cuda:0', 'cpu'] },
  ];
  api.S.fields = fields;
  api.S.fieldMap = Object.fromEntries(fields.map(field => [field.key, field]));
  api.S.boot = { schema: { locked_when_queued: ['WANDB.'], mode_labels: { train_and_test: 'Training', unsupervised_method: 'Unsupervised' } }, platform: 'nt',
    settings: { default_backend: 'docker', max_parallel_jobs: 2 }, configs: [
      { path: 'configs/thesis/b1/UBFC/physnet_face.yaml', group: 'local', mode: 'train_and_test', model: 'Physnet', train: 'UBFC-rPPG', test: 'UBFC-rPPG', backend: 'local' },
      { path: 'configs/thesis/b1/UBFC/pos_face.yaml', group: 'local', mode: 'unsupervised_method', model: '', train: '', test: 'UBFC-rPPG', backend: 'local' },
      { path: 'configs/thesis/b1/PURE/physnet_face.yaml', group: 'local', mode: 'train_and_test', model: 'Physnet', train: 'PURE', test: 'PURE', backend: 'local' },
      { path: 'configs/other.yaml', group: 'local', mode: 'train_and_test', model: 'TSCAN', train: 'PURE', test: 'PURE', backend: 'docker' },
    ] };
  return { ...api, el: document.getElementById };
}

const newBatch = (f, selected = []) => {
  f.S.batch = { selected: new Set(selected), query: '', backend: null, answers: {}, secrets: {}, name: '', result: null,
    excluded: new Set(), filter: null, seq: 0, nodes: [] };
  return f.S.batch;
};
const checkbox = node => node.children.find(c => c.tagName === 'INPUT');
const folderBox = (f, name) => f.el('batchTree').byTag('summary').find(s => s.textContent.startsWith(name)).children[0];

test('a folder checkbox selects every config below it; partly selected folders are indeterminate', () => {
  const f = fixture();
  const b = newBatch(f);
  f.renderBatchSelect();
  assert.equal(f.el('batchCheck').disabled, true, 'nothing to check yet');
  const ubfc = folderBox(f, 'UBFC');
  ubfc.checked = true;
  ubfc.events.change();
  assert.deepEqual([...b.selected].sort(), ['configs/thesis/b1/UBFC/physnet_face.yaml', 'configs/thesis/b1/UBFC/pos_face.yaml']);
  assert.equal(f.el('batchCheck').textContent, 'Check 2 configs');
  assert.equal(f.el('batchCount').textContent, '2 configs selected');
  const b1 = folderBox(f, 'b1');
  assert.equal(b1.checked, false);
  assert.equal(b1.indeterminate, true, 'one of the two dataset folders is selected');
  assert.equal(ubfc.checked, true);
  assert.equal(b.radios.find(r => r.value === 'local').radio.checked, true, 'the backend follows the selected configs (local paths)');
  assert.equal(b.radios.find(r => r.value === 'docker').radio.checked, false);
  b1.checked = true;
  b1.events.change();
  assert.equal(b.selected.size, 3);
  assert.equal(b1.indeterminate, false);
});

test('the search keeps the selection and folders open on what is selected', () => {
  const f = fixture();
  const b = newBatch(f, ['configs/other.yaml']);
  b.query = 'pure';
  f.renderBatchSelect();
  const files = f.el('batchTree').byClass('batch-file').map(n => n.attrs.title);
  assert.deepEqual(files, ['configs/thesis/b1/PURE/physnet_face.yaml', 'configs/other.yaml']);
  assert.equal(checkbox(f.el('batchTree').byClass('batch-file')[1]).checked, true);
  assert.equal(f.el('batchCount').textContent, '1 config selected');
});

const result = () => ({
  name: 'thesis/b1', folder: 'configs/thesis/b1', launchable: 3, can_launch: true,
  counts: { ready: 2, warn: 1, blocked: 2 },
  checks: [{ id: 'policy', title: 'Run order', status: 'answered', message: '3 jobs', options: [{ value: 'queue', label: 'One after another' }] }],
  issues: [
    { status: 'error', message: 'folder of experiment JZC not found', configs: ['JZC/a', 'JZC/b'] },
    { status: 'error', message: 'Only one config', configs: ['JZC/b'] },
  ],
  configs: [
    { path: 'p/a', name: 'UBFC/a', status: 'ready', errors: [], warnings: [], model: 'Physnet', datasets: 'UBFC-rPPG', description: 'First' },
    { path: 'p/b', name: 'UBFC/b', status: 'warn', errors: [], warnings: ['usually 72 px'], model: 'Physnet', datasets: 'UBFC-rPPG' },
    { path: 'p/c', name: 'UBFC/c', status: 'ready', errors: [], warnings: [], model: 'POS', datasets: 'UBFC-rPPG' },
    { path: 'p/d', name: 'JZC/a', status: 'blocked', errors: ['folder of experiment JZC not found'], warnings: [] },
    { path: 'p/e', name: 'JZC/b', status: 'blocked', errors: ['folder of experiment JZC not found', 'Only one config'], warnings: [] },
  ],
});

test('the checks list the configs, the shared problems and launch the ready configs', () => {
  const f = fixture();
  const b = newBatch(f);
  b.result = result();
  f.el('batchLaunch');
  f.renderBatchChecks();
  const body = f.el('batchBody');
  assert.equal(f.el('batchWill').textContent, '3 of 5 configs will be launched');
  assert.equal(f.el('batchLaunch').textContent, 'Launch 3 jobs');
  assert.equal(f.el('batchLaunch').disabled, false);
  assert.deepEqual(body.byClass('issue').map(n => n.textContent), ['folder of experiment JZC not found2 configsJZC/a, JZC/b'],
    'only problems of several configs are listed together');
  const rows = body.byClass('batch-row');
  assert.equal(rows.length, 5);
  const boxes = rows.map(row => checkbox(row.children[0]));
  assert.deepEqual(boxes.map(box => 'disabled' in box.attrs), [false, false, false, true, true], 'blocked configs cannot be launched');
  assert.ok(rows[1].textContent.includes('Ready · 1 warning'));
  assert.ok(rows[4].textContent.includes('+1 more'));
  boxes[2].checked = false;
  boxes[2].events.change();
  assert.deepEqual([...b.excluded], ['p/c']);
  assert.equal(f.el('batchLaunch').textContent, 'Launch 2 jobs');
  // Filtering by status keeps the exclusions.
  body.byClass('batch-blocked')[0].events.click();
  assert.equal(b.filter, 'blocked');
  assert.equal(f.el('batchBody').byClass('batch-row').length, 2);
  assert.equal(f.el('batchLaunch').textContent, 'Launch 2 jobs');
});

test('an open question blocks the launch', () => {
  const f = fixture();
  const b = newBatch(f);
  b.result = result();
  b.result.checks[0].status = 'question';
  f.renderBatchChecks();
  assert.equal(f.el('batchLaunch').disabled, true);
});

const job = (id, index, status, extra = {}) => ({ id, name: `UBFC/${id}`, status, created: '2026-11-03T10:00:00', backend: 'local',
  batch: { id: 'B1', name: 'thesis/b1', index, count: 3 }, ...extra });
const entry = (index, status, extra = {}) => ({ path: `p/${index}`, name: `UBFC/${index}`, index, status, job: `j${index}`, attempts: 1, ...extra });

test('batch jobs are nested under their batch; long batches start collapsed', () => {
  const f = fixture();
  f.el('jobFilter').value = 'all';
  f.el('tab-jobs').className = '';
  f.S.batches = [{ id: 'B1', name: 'thesis/b1', created: '2026-11-03T10:00:00', status: 'running', active: true, backend: 'local',
    answers: { policy: 'queue' }, configs: [entry(0, 'succeeded'), entry(1, 'running'), entry(2, 'queued')] }];
  f.S.jobs = [job('j0', 0, 'succeeded'), job('j1', 1, 'running'), job('j2', 2, 'queued'),
    { id: 'solo', name: 'Single run', status: 'succeeded', created: '2026-11-02T09:00:00', backend: 'local' }];
  f.renderJobList();
  const list = f.el('jobList');
  assert.equal(list.children.length, 2, 'one batch row and one single job');
  assert.ok(list.children[0].textContent.includes('batch · 1/3 done'));
  assert.deepEqual(list.children[0].byClass('fold').map(n => n.textContent.slice(0, 6)), ['UBFC/j', 'UBFC/j', 'UBFC/j']);
  assert.equal(f.S.selectedBatch, 'B1', 'the newest entry is selected');
  f.S.batches[0].configs = Array.from({ length: 13 }, (_, i) => entry(i, 'queued'));
  f.renderJobList();
  assert.equal(list.children[0].byClass('fold').length, 0, 'more than 12 jobs: collapsed until opened');
});

test('the batch detail lists its jobs and relaunches the unfinished ones', () => {
  const f = fixture();
  const hour = 3600 * 1000;
  const now = Date.now();
  f.S.batches = [{ id: 'B1', name: 'thesis/b1', created: '2026-11-03T10:00:00', status: 'running', active: true, backend: 'local',
    folder: 'configs/thesis/b1', answers: { policy: 'parallel', wandb: 'existing' }, configs: [
      entry(0, 'succeeded', { started: new Date(now - 3 * hour).toISOString(), finished: new Date(now - 2 * hour).toISOString() }),
      entry(1, 'failed'), entry(2, 'queued'), entry(3, 'queued')] }];
  f.S.jobs = [job('j0', 0, 'succeeded', { summary: { mae_bpm: 2.345 } }), job('j1', 1, 'failed', { progress: { metrics: { FFT: { MAE: { value: 7.5 } } } } })];
  f.S.selectedBatch = 'B1';
  f.updateBatchDetail(true);
  const text = f.el('jobDetail').textContent;
  assert.ok(text.includes('1 of 4 succeeded'));
  assert.ok(text.includes('Run order: several at once'));
  assert.ok(text.includes('W&B: saved API key'));
  assert.ok(text.includes('2.35') && text.includes('7.50'), 'MAE from summary.json or the console metrics');
  const buttons = f.el('bdHead').byTag('button').map(n => n.textContent);
  assert.deepEqual(buttons, ['Stop remaining jobs', 'Relaunch 1 unfinished…']);
  // Two queued jobs of ~1 h, two at a time.
  const progress = f.batchProgress(f.S.batches[0]);
  assert.ok(Math.abs(progress.eta - 3600) < 5, `expected ~1 h left, got ${progress.eta}`);
});

test('the batch offers to continue its stopped trainings and to edit its queued jobs', () => {
  const f = fixture();
  f.S.batches = [{ id: 'B1', name: 'thesis/b1', created: '2026-11-03T10:00:00', status: 'running', active: true, backend: 'local',
    configs: [entry(0, 'succeeded'), entry(1, 'succeeded'), entry(2, 'queued'), entry(3, 'queued')] }];
  f.S.jobs = [job('j0', 0, 'succeeded', { can_continue: true }), job('j1', 1, 'succeeded', { can_continue: false }),
    job('j2', 2, 'queued'), job('j3', 3, 'queued')];
  f.S.selectedBatch = 'B1';
  f.updateBatchDetail(true);
  assert.deepEqual(f.el('bdHead').byTag('button').map(n => n.textContent), ['Continue 1 training…', 'Edit 2 queued…', 'Stop remaining jobs']);
});

const continuationRows = () => [
  { job: 'j0', name: 'JZC/rhythmformer', next_epoch: 5, epochs_total: 20, best_epoch: 0, min_epochs: 4, max_epochs: 15, errors: [] },
  { job: 'j1', name: 'RM/physnet', next_epoch: 9, epochs_total: 20, best_epoch: 4, min_epochs: null, max_epochs: 11, errors: [] },
  { job: 'j2', name: 'RM/other', next_epoch: 6, epochs_total: 20, best_epoch: 1, min_epochs: 0, max_epochs: 14, errors: ['A patience of 4 is already used up.'] },
];
const rowBoxes = f => f.el('changeRows').byTag('input');

test('the continue dialog shows what each job will train and sends only the chosen jobs', () => {
  const f = fixture();
  f.continueDialog([{ id: 'j0' }, { id: 'j1' }, { id: 'j2' }]);
  const c = f.S.changing;
  assert.equal(f.el('modalTitle').textContent, 'Continue 3 trainings');
  assert.deepEqual(JSON.parse(JSON.stringify(f.continueRequest(c).body.early_stopping)), { enabled: false }, 'removing early stopping is the default');
  c.mode = 'patience';
  assert.equal(f.continueRequest(c), null, 'no request without a patience');
  c.patience = '8';
  assert.equal(f.continueRequest(c).body.early_stopping.patience, '8');
  c.rows = continuationRows();
  f.renderContinueRows(c);
  const text = f.el('changeRows').textContent;
  assert.ok(text.includes('5 / 20') && text.includes('4–15 more epochs'), text);
  assert.ok(text.includes('up to 11 more epochs'), 'without the losses, only the maximum is known');
  assert.ok(text.includes('already used up'));
  const boxes = rowBoxes(f);
  assert.deepEqual(boxes.map(box => [box.checked, 'disabled' in box.attrs]), [[true, false], [true, false], [false, true]]);
  assert.equal(f.el('changeApply').textContent, 'Continue 2 trainings');
  assert.equal(f.el('changeApply').disabled, false);
  boxes[0].checked = false;
  boxes[0].events.change();
  assert.equal(f.el('changeApply').textContent, 'Continue 1 training');
  assert.deepEqual([...c.excluded], ['j0']);
  c.mode = 'remove';
  f.renderContinueRows(c);
  assert.ok(f.el('changeRows').textContent.includes('15 more epochs'));
  assert.ok(!f.el('changeRows').textContent.includes('4–15'));
});

test('the edit dialog offers the editable parameters and shows the change per job', () => {
  const f = fixture();
  f.editQueuedDialog([{ id: 'j2' }, { id: 'j3' }, { id: 'j4' }]);
  const e = f.S.changing;
  assert.equal(f.el('modalTitle').textContent, 'Edit 3 queued jobs');
  const picker = f.el('editParameters').byTag('select')[0];
  const offered = picker.byTag('option').map(o => o.attrs.value);
  assert.ok(offered.includes('TRAIN.EARLY_STOPPING.PATIENCE'));
  assert.ok(!offered.includes('WANDB.MODE'), 'locked parameters are not offered');
  assert.equal(picker.value, 'TRAIN.EARLY_STOPPING.PATIENCE');
  assert.equal(f.editRequest(e), null, 'no request without a value');
  f.renderEditRows(e);
  assert.ok(f.el('changeRows').textContent.includes('Enter a value'));
  picker.value = 'TRAIN.EARLY_STOPPING.ENABLED';
  picker.events.change();
  const value = f.el('editParameters').byTag('select')[1];
  assert.deepEqual(value.byTag('option').map(o => o.attrs.value), ['', 'true', 'false'], 'a switch offers true and false');
  e.changes[0] = { key: 'TRAIN.EARLY_STOPPING.PATIENCE', value: '8' };
  f.renderEditParameters(e);
  assert.deepEqual(JSON.parse(JSON.stringify(f.editRequest(e).body.changes)), { 'TRAIN.EARLY_STOPPING.PATIENCE': '8' });
  e.rows = [
    { job: 'j2', name: 'baseline/physnet', status: 'change', before: { 'TRAIN.EARLY_STOPPING.PATIENCE': 4 }, after: { 'TRAIN.EARLY_STOPPING.PATIENCE': 8 }, errors: [] },
    { job: 'j3', name: 'baseline/pos', status: 'unused', before: {}, after: {}, errors: [] },
    { job: 'j4', name: 'started', status: 'error', before: {}, after: {}, errors: ['Already started.'] },
  ];
  f.renderEditRows(e);
  const text = f.el('changeRows').textContent;
  assert.ok(text.includes('Patience (epochs): 4 → 8'), text);
  assert.ok(text.includes('does not use it') && text.includes('Already started.'));
  assert.deepEqual(rowBoxes(f).map(box => box.checked), [true, false, false]);
  assert.equal(f.el('changeApply').textContent, 'Apply to 1 job');
});
